-- 0005_api_append.sql: the SQL API begins. Errors, validation, and the write path.
--
-- From here on the database is the implementation and every client is a caller:
-- the Go package, a Python service and psql all reach the same functions, and
-- every rule a row must satisfy is checked here, by the function that writes it,
-- before the CHECK constraints underneath get a chance to say it less clearly.
--
-- Rendered as a Go text/template, like every migration, though nothing here
-- uses a substitution.
--
-- Conventions every later API migration follows:
--
--   * A public function is documented with COMMENT ON FUNCTION and is part of
--     the contract. A function whose name starts with an underscore is not, and
--     may change in any migration.
--
--   * Every function is SECURITY INVOKER and pins its search_path to the one
--     the migration ran under -- the extension schema, then mempher -- so an
--     unqualified vector type or btree_gin operator resolves the same way
--     whoever calls it, and an object in a caller's schema cannot shadow one
--     of ours.
--
--   * No function reads a clock it was not handed. Every timestamp a row keeps
--     is a parameter, defaulting to now() for interactive callers, which is
--     what keeps a replay byte-identical.
--
--   * A caller's mistake raises one of the SQLSTATEs below, in class MP. The
--     message is for a human; the code is the contract.
--
--     MP001 invalid scope id            MP011 invalid validity window
--     MP002 invalid episode content     MP012 invalid extraction
--     MP003 invalid role                MP013 invalid argument
--     MP004 invalid source              MP020 not found
--     MP005 invalid actor               MP021 vector dimension mismatch
--     MP006 invalid binding             MP022 fact windows conflict
--     MP007 invalid channel             MP023 job not leased by this worker
--     MP008 invalid job kind            MP024 job is not dead
--     MP009 invalid job state           MP025 the same work is already queued
--     MP010 invalid fact

-- ---------------------------------------------------------------------------
-- Internal helpers
-- ---------------------------------------------------------------------------

-- Raises a caller error. A function rather than a RAISE at each site so the
-- class stays in one place and every message reads the same way.
CREATE FUNCTION mempher._fail(p_code text, p_message text, p_hint text DEFAULT NULL)
    RETURNS void
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_hint IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = p_code, MESSAGE = p_message;
    END IF;
    RAISE EXCEPTION USING ERRCODE = p_code, MESSAGE = p_message, HINT = p_hint;
END $$;

-- A scope id is 1 to 200 bytes and holds no control character: it ends up in
-- log lines and operator queries, where an embedded newline is at best
-- confusing. The range covers C0, DEL and C1, which is what Go's
-- unicode.IsControl means by a control character.
CREATE FUNCTION mempher._check_scope(p_scope text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_scope IS NULL OR p_scope = '' THEN
        PERFORM mempher._fail('MP001', 'scope id is empty');
    ELSIF octet_length(p_scope) > 200 THEN
        PERFORM mempher._fail('MP001', format(
            'scope id is %s bytes, limit is 200', octet_length(p_scope)));
    ELSIF p_scope ~ '[\x01-\x1f\x7f-\x9f]' THEN
        PERFORM mempher._fail('MP001', 'scope id holds a control character');
    END IF;
END $$;

-- A binding is a flat object of at most 32 string entries, each key 1 to 64
-- bytes and each value at most 512. NULL is not a binding; the caller means
-- '{}' and is told so rather than guessed at.
CREATE FUNCTION mempher._check_binding(p_binding jsonb, p_where text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad text;
BEGIN
    IF p_binding IS NULL OR jsonb_typeof(p_binding) <> 'object' THEN
        PERFORM mempher._fail('MP006', format('%s: binding is not a JSON object', p_where));
    END IF;
    IF (SELECT count(*) FROM jsonb_object_keys(p_binding)) > 32 THEN
        PERFORM mempher._fail('MP006', format(
            '%s: binding has %s keys, limit is 32',
            p_where, (SELECT count(*) FROM jsonb_object_keys(p_binding))));
    END IF;
    SELECT key INTO v_bad
    FROM jsonb_each(p_binding)
    WHERE jsonb_typeof(value) <> 'string'
       OR octet_length(key) NOT BETWEEN 1 AND 64
       OR octet_length(value #>> '{}') > 512
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP006', format(
            '%s: binding entry %L must be a string of at most 512 bytes under a key of 1 to 64 bytes',
            p_where, v_bad));
    END IF;
END $$;

-- One element of mempher.append's p_episodes. p_where names it in a message,
-- because "episode 37 of 64" is what a caller needs to find it.
CREATE FUNCTION mempher._check_new_episode(p_episode jsonb, p_where text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_unknown text;
BEGIN
    IF jsonb_typeof(p_episode) IS DISTINCT FROM 'object' THEN
        PERFORM mempher._fail('MP013', format('%s is not a JSON object', p_where));
    END IF;
    -- An unknown key is a typo, and a typo such as "occurredAt" would otherwise
    -- quietly date the episode to now.
    SELECT k INTO v_unknown
    FROM jsonb_object_keys(p_episode) AS k
    WHERE k NOT IN ('content', 'role', 'actor', 'source', 'occurred_at', 'binding')
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format('%s has unknown key %L', p_where, v_unknown),
            'an episode takes content, role, actor, source, occurred_at and binding');
    END IF;

    IF jsonb_typeof(p_episode->'content') IS DISTINCT FROM 'string'
       OR p_episode->>'content' = '' THEN
        PERFORM mempher._fail('MP002', format('%s: content is empty', p_where));
    ELSIF octet_length(p_episode->>'content') > 1048576 THEN
        PERFORM mempher._fail('MP002', format(
            '%s: content is %s bytes, limit is 1048576',
            p_where, octet_length(p_episode->>'content')));
    END IF;

    IF jsonb_typeof(p_episode->'role') IS DISTINCT FROM 'string'
       OR p_episode->>'role' NOT IN ('user', 'assistant', 'system', 'tool', 'observation') THEN
        PERFORM mempher._fail('MP003', format('%s: role %s is not one of user, assistant, system, tool, observation',
            p_where, coalesce(p_episode->'role', 'null'::jsonb)));
    END IF;

    IF p_episode ? 'actor' AND (jsonb_typeof(p_episode->'actor') <> 'string'
       OR octet_length(p_episode->>'actor') > 200) THEN
        PERFORM mempher._fail('MP005', format('%s: actor must be a string of at most 200 bytes', p_where));
    END IF;

    IF jsonb_typeof(p_episode->'source') IS DISTINCT FROM 'string'
       OR octet_length(p_episode->>'source') NOT BETWEEN 1 AND 100 THEN
        PERFORM mempher._fail('MP004', format('%s: source must be a string of 1 to 100 bytes', p_where));
    END IF;

    IF p_episode ? 'occurred_at' AND (jsonb_typeof(p_episode->'occurred_at') <> 'string'
       OR NOT pg_input_is_valid(p_episode->>'occurred_at', 'timestamptz')) THEN
        PERFORM mempher._fail('MP013', format(
            '%s: occurred_at %s is not a timestamp', p_where, p_episode->'occurred_at'));
    END IF;

    IF p_episode ? 'binding' THEN
        PERFORM mempher._check_binding(p_episode->'binding', p_where);
    END IF;
END $$;

-- The job kinds a write may imply, deduplicated in the order given. Unknown
-- kinds are rejected here rather than by the CHECK on mempher.jobs, which would
-- report a constraint name instead of the kind.
CREATE FUNCTION mempher._check_job_kinds(p_kinds text[])
    RETURNS text[]
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad text;
BEGIN
    IF p_kinds IS NULL THEN
        RETURN '{}';
    END IF;
    SELECT k INTO v_bad FROM unnest(p_kinds) AS k
    WHERE k IS NULL OR k NOT IN ('encode', 'extract') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP008', format('job kind %L is not one of encode, extract', v_bad));
    END IF;
    RETURN ARRAY(
        SELECT k FROM unnest(p_kinds) WITH ORDINALITY AS u(k, ord)
        GROUP BY k ORDER BY min(ord));
END $$;

-- ---------------------------------------------------------------------------
-- mempher.append
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.append(
    p_scope        text,
    p_episodes     jsonb,
    p_at           timestamptz DEFAULT now(),
    p_jobs         text[]      DEFAULT ARRAY['encode'],
    p_max_attempts int         DEFAULT 5
)
    RETURNS TABLE (id uuid, seq bigint)
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
DECLARE
    v_n     int;
    v_kinds text[];
    v_base  bigint;
    v_ids   uuid[];
    v_seqs  bigint[];
    v_elem  jsonb;
    v_ord   bigint;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    IF p_at IS NULL THEN
        PERFORM mempher._fail('MP013', 'at is null');
    END IF;
    IF jsonb_typeof(p_episodes) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP013', 'episodes is not a JSON array');
    END IF;
    v_n := jsonb_array_length(p_episodes);
    IF v_n NOT BETWEEN 1 AND 64 THEN
        PERFORM mempher._fail('MP013', format('append holds %s episodes, must be 1 to 64', v_n),
            'a batch is one round trip and one encode job; split a longer one');
    END IF;
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;
    v_kinds := mempher._check_job_kinds(p_jobs);

    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_episodes) WITH ORDINALITY AS a(e, o)
    LOOP
        PERFORM mempher._check_new_episode(v_elem, format('episode %s of %s', v_ord, v_n));
    END LOOP;

    -- Allocating the sequence and creating the scope are one upsert, so seq is
    -- dense and gap-free without a second round trip or an explicit lock.
    -- Concurrent appends to one scope serialise on this row, which is the
    -- right semantics for a totally ordered log.
    INSERT INTO mempher.scopes AS s (id, last_seq, created_at)
    VALUES (p_scope, v_n, p_at)
    ON CONFLICT (id) DO UPDATE SET last_seq = s.last_seq + v_n
    RETURNING s.last_seq - v_n INTO v_base;

    WITH inserted AS (
        INSERT INTO mempher.episodes
            (scope_id, seq, content, role, actor, source, occurred_at, ingested_at, binding)
        SELECT p_scope,
               v_base + a.o,
               a.e->>'content',
               a.e->>'role',
               coalesce(a.e->>'actor', ''),
               a.e->>'source',
               coalesce((a.e->>'occurred_at')::timestamptz, p_at),
               p_at,
               coalesce(a.e->'binding', '{}'::jsonb)
        FROM jsonb_array_elements(p_episodes) WITH ORDINALITY AS a(e, o)
        RETURNING id, seq
    )
    SELECT array_agg(i.id ORDER BY i.seq), array_agg(i.seq ORDER BY i.seq)
    INTO v_ids, v_seqs
    FROM inserted i;

    -- One job per kind for the whole batch: a turn that produced several
    -- episodes is embedded in one model call.
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    SELECT k, p_scope, v_ids, p_at, p_max_attempts, p_at, p_at
    FROM unnest(v_kinds) AS k
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING;

    RETURN QUERY SELECT u.id, u.seq FROM unnest(v_ids, v_seqs) AS u(id, seq);
END $$;

COMMENT ON FUNCTION mempher.append(text, jsonb, timestamptz, text[], int) IS
    'Appends 1 to 64 episodes to a scope, atomically, with the jobs they imply. '
    'p_episodes is a JSON array of objects: content, role and source are required; '
    'actor, occurred_at (defaults to p_at) and binding (a flat object of strings) are optional. '
    'p_at is the ingest time. p_jobs names the projections to queue, one job each for '
    'the whole batch: encode, and extract when L1 is on. Returns each episode''s id and seq, in order.';
