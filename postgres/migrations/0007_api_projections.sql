-- 0007_api_projections.sql: writing the projections, and reading L1.
--
-- A worker produces two kinds of output, vectors and an extraction, and each is
-- one call here. What the output means for the projection -- which episode row
-- a vector's scope comes from, when a claim supersedes another, what an
-- extraction that has already been applied does the second time -- is decided
-- by these functions, not by the worker.
--
-- Rendered as a Go text/template: {{.VectorDimensions}} and
-- {{.TextSearchConfig}} are the deployment's, as in 0001. Conventions and error
-- codes are those of 0005_api_append.sql.

-- ---------------------------------------------------------------------------
-- Internal helpers
-- ---------------------------------------------------------------------------

-- Matches any one of a text's lexemes, stemmed under the schema's own
-- configuration exactly as the indexed columns were. A conversational question
-- has its words spread across a conversation, not gathered in one turn, so the
-- channels that use this want everything that shares anything, best overlap
-- first; requiring every word left 83% of LongMemEval questions matching
-- nothing. quote_literal makes each lexeme a literal rather than something the
-- tsquery parser could read as an operator, and text that is only stop words is
-- the empty query, which matches nothing and ranks everything zero.
CREATE FUNCTION mempher._any_lexeme_query(p_text text)
    RETURNS tsquery
    LANGUAGE sql
    STABLE
    SET search_path FROM CURRENT
RETURN coalesce(
    (SELECT string_agg(quote_literal(lexeme), ' | ')
     FROM unnest(to_tsvector('{{.TextSearchConfig}}'::regconfig, coalesce(p_text, ''))))::tsquery,
    ''::tsquery);

-- A label that ends up in operator queries and log lines: 1 to p_max bytes, no
-- control character, and for a slug no whitespace either.
CREATE FUNCTION mempher._check_label(
    p_value text, p_what text, p_code text, p_max int, p_slug boolean DEFAULT false)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_value IS NULL OR p_value = '' THEN
        PERFORM mempher._fail(p_code, format('%s is empty', p_what));
    ELSIF octet_length(p_value) > p_max THEN
        PERFORM mempher._fail(p_code, format('%s is %s bytes, limit is %s',
            p_what, octet_length(p_value), p_max));
    ELSIF p_value ~ '[\x01-\x1f\x7f-\x9f]' THEN
        PERFORM mempher._fail(p_code, format('%s holds a control character', p_what));
    ELSIF p_slug AND p_value ~ '\s' THEN
        PERFORM mempher._fail(p_code, format('%s %L holds whitespace', p_what, p_value),
            'a predicate is identity: "lives_in" and "lives in" must not be two relations');
    END IF;
END $$;

-- A JSON timestamp field, or NULL when it is absent and p_required is false.
CREATE FUNCTION mempher._json_time(
    p_object jsonb, p_key text, p_where text, p_code text, p_required boolean)
    RETURNS timestamptz
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_value jsonb := p_object->p_key;
BEGIN
    IF v_value IS NULL OR v_value = 'null'::jsonb THEN
        IF p_required THEN
            PERFORM mempher._fail(p_code, format('%s: %s is required', p_where, p_key));
        END IF;
        RETURN NULL;
    END IF;
    IF jsonb_typeof(v_value) <> 'string'
       OR NOT pg_input_is_valid(v_value #>> '{}', 'timestamptz')
       OR NOT isfinite((v_value #>> '{}')::timestamptz) THEN
        PERFORM mempher._fail(p_code, format('%s: %s %s is not a finite timestamp', p_where, p_key, v_value));
    END IF;
    RETURN (v_value #>> '{}')::timestamptz;
END $$;

-- The assertions of an extraction as typed rows, once apply_extraction has
-- checked them. A function rather than a temporary table: one per extraction
-- would churn the system catalogue at the rate the workers run.
CREATE FUNCTION mempher._assertions(p_assert jsonb)
    RETURNS TABLE (ord bigint, subject text, predicate text, object text,
                   statement text, valid tstzrange, confidence real)
    LANGUAGE sql
    STABLE
    SET search_path FROM CURRENT
AS $$
    SELECT a.o, a.e->>'subject', a.e->>'predicate', a.e->>'object', a.e->>'statement',
           tstzrange((a.e->>'valid_from')::timestamptz, (a.e->>'valid_to')::timestamptz, '[)'),
           coalesce((a.e->'confidence')::real, 1)
    FROM jsonb_array_elements(p_assert) WITH ORDINALITY AS a(e, o)
$$;

-- ---------------------------------------------------------------------------
-- Encodings
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.put_encodings(
    p_model    text,
    p_episodes uuid[],
    p_vectors  vector[],
    p_at       timestamptz DEFAULT now(),
    p_surprise real[]      DEFAULT NULL
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_n       int := coalesce(cardinality(p_episodes), 0);
    v_bad     int;
    v_missing uuid;
    v_count   int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    IF v_n = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'name at least one episode, and no null');
    END IF;
    IF coalesce(cardinality(p_vectors), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s vectors for %s episodes',
            coalesce(cardinality(p_vectors), 0), v_n));
    END IF;
    IF p_surprise IS NOT NULL AND cardinality(p_surprise) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s surprise values for %s episodes',
            cardinality(p_surprise), v_n));
    END IF;
    -- Caught here, by position, rather than by the column's typmod, which
    -- reports the width and not which vector or which model produced it.
    SELECT o INTO v_bad FROM unnest(p_vectors) WITH ORDINALITY AS u(v, o)
    WHERE v IS NULL OR vector_dims(v) <> {{.VectorDimensions}} LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP021', format(
            'vector %s from model %L has %s dimensions but the schema was migrated for %s',
            v_bad, p_model, vector_dims(p_vectors[v_bad]), {{.VectorDimensions}}));
    END IF;
    SELECT o INTO v_bad FROM unnest(p_surprise) WITH ORDINALITY AS u(s, o)
    WHERE s IS NULL OR s < 0 OR s > 1 LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format('surprise %s is %s, must be in [0,1]',
            v_bad, p_surprise[v_bad]));
    END IF;

    -- scope_id and ingested_at come from the episode row, never the caller:
    -- they are denormalised so the vector index can filter without a join,
    -- and deriving them here is what keeps them from disagreeing with L0.
    -- The upsert is what makes an encode job safe to retry.
    INSERT INTO mempher.episode_encodings
        (episode_id, model, scope_id, ingested_at, content_vec, surprise, encoded_at)
    SELECT e.id, p_model, e.scope_id, e.ingested_at, u.v, coalesce(u.s, 0), p_at
    FROM unnest(p_episodes, p_vectors, p_surprise) AS u(id, v, s)
    JOIN mempher.episodes e ON e.id = u.id
    ON CONFLICT (episode_id, model) DO UPDATE
    SET content_vec = excluded.content_vec,
        surprise    = excluded.surprise,
        encoded_at  = excluded.encoded_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;

    IF v_count < (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) THEN
        SELECT x INTO v_missing FROM unnest(p_episodes) AS x
        WHERE NOT EXISTS (SELECT 1 FROM mempher.episodes e WHERE e.id = x) LIMIT 1;
        PERFORM mempher._fail('MP020', format('episode %s does not exist', v_missing));
    END IF;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.put_encodings(text, uuid[], vector[], timestamptz, real[]) IS
    'Writes one vector per episode for a model, replacing any the episode already had '
    'for that model, and returns how many were written. The arrays are parallel. An '
    'episode that does not exist fails the whole call with MP020.';

-- ---------------------------------------------------------------------------
-- Facts: applying an extraction
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.apply_extraction(
    p_scope     text,
    p_extractor text,
    p_episodes  uuid[],
    p_assert    jsonb       DEFAULT '[]',
    p_retract   jsonb       DEFAULT '[]',
    p_at        timestamptz DEFAULT now()
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_elem     jsonb;
    v_ord      bigint;
    v_where    text;
    v_from     timestamptz;
    v_to       timestamptz;
    v_conf     jsonb;
    v_found    int;
    v_marked   int;
    v_fact     uuid;
    v_lower    timestamptz;
    v_clash    record;
    v_unknown  text;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP012', 200);
    PERFORM mempher._check_at(p_at);
    IF p_episodes IS NULL OR cardinality(p_episodes) = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP012', 'an extraction names the episodes it read, and no null');
    END IF;
    IF jsonb_typeof(p_assert) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP012', 'assert is not a JSON array');
    ELSIF jsonb_array_length(p_assert) > 64 THEN
        PERFORM mempher._fail('MP012', format('extraction asserts %s facts, limit is 64',
            jsonb_array_length(p_assert)));
    END IF;
    IF jsonb_typeof(p_retract) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP012', 'retract is not a JSON array');
    ELSIF jsonb_array_length(p_retract) > 64 THEN
        PERFORM mempher._fail('MP012', format('extraction retracts %s facts, limit is 64',
            jsonb_array_length(p_retract)));
    END IF;

    -- Every assertion is checked before anything is written.
    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_assert) WITH ORDINALITY AS a(e, o)
    LOOP
        v_where := format('assertion %s', v_ord);
        IF jsonb_typeof(v_elem) <> 'object' THEN
            PERFORM mempher._fail('MP012', format('%s is not a JSON object', v_where));
        END IF;
        SELECT k INTO v_unknown FROM jsonb_object_keys(v_elem) AS k
        WHERE k NOT IN ('subject', 'predicate', 'object', 'statement', 'valid_from', 'valid_to', 'confidence')
        LIMIT 1;
        IF FOUND THEN
            PERFORM mempher._fail('MP012', format('%s has unknown key %L', v_where, v_unknown),
                'an assertion takes subject, predicate, object, statement, valid_from, valid_to and confidence');
        END IF;
        IF jsonb_typeof(v_elem->'subject') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'predicate') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'object') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'statement') IS DISTINCT FROM 'string' THEN
            PERFORM mempher._fail('MP010', format(
                '%s: subject, predicate, object and statement must all be strings', v_where));
        END IF;
        PERFORM mempher._check_label(v_elem->>'subject', v_where || ': subject', 'MP010', 200);
        PERFORM mempher._check_label(v_elem->>'predicate', v_where || ': predicate', 'MP010', 100, true);
        IF v_elem->>'object' = '' OR octet_length(v_elem->>'object') > 1000 THEN
            PERFORM mempher._fail('MP010', format('%s: object must be 1 to 1000 bytes', v_where));
        END IF;
        IF btrim(v_elem->>'statement') = '' OR octet_length(v_elem->>'statement') > 4000 THEN
            PERFORM mempher._fail('MP010', format('%s: statement must be 1 to 4000 bytes, not blank', v_where));
        END IF;
        v_conf := coalesce(v_elem->'confidence', '1'::jsonb);
        IF jsonb_typeof(v_conf) <> 'number' OR v_conf::numeric < 0 OR v_conf::numeric > 1 THEN
            PERFORM mempher._fail('MP010', format('%s: confidence %s must be a number in [0,1]', v_where, v_conf));
        END IF;
        v_from := mempher._json_time(v_elem, 'valid_from', v_where, 'MP011', true);
        v_to := mempher._json_time(v_elem, 'valid_to', v_where, 'MP011', false);
        IF v_to IS NOT NULL AND v_to <= v_from THEN
            PERFORM mempher._fail('MP011', format(
                '%s: the window ends (%s) at or before it starts (%s)', v_where, v_to, v_from));
        END IF;
    END LOOP;

    -- The same claim twice over the same window is one claim, and the first
    -- wording of it wins. Over two windows that overlap it is a contradiction
    -- the model made within one answer, which the database would reject anyway,
    -- less legibly.
    SELECT a.subject, a.predicate, a.object INTO v_clash
    FROM mempher._assertions(p_assert) a JOIN mempher._assertions(p_assert) b
      ON a.ord < b.ord AND (a.subject, a.predicate, a.object) = (b.subject, b.predicate, b.object)
     AND a.valid && b.valid AND a.valid <> b.valid
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP022', format(
            '%L %L %L is asserted over two overlapping windows in one extraction',
            v_clash.subject, v_clash.predicate, v_clash.object));
    END IF;

    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_retract) WITH ORDINALITY AS a(e, o)
    LOOP
        v_where := format('retraction %s', v_ord);
        IF jsonb_typeof(v_elem) <> 'object'
           OR jsonb_typeof(v_elem->'fact') IS DISTINCT FROM 'string'
           OR NOT pg_input_is_valid(v_elem->>'fact', 'uuid') THEN
            PERFORM mempher._fail('MP012', format('%s must name a fact by its uuid', v_where));
        END IF;
        PERFORM mempher._json_time(v_elem, 'at', v_where, 'MP012', true);
    END LOOP;

    -- The markers say which episodes this extractor has read. They are what
    -- makes a retried job a no-op rather than a second application: if every
    -- episode was already marked, the extraction has been applied, and its
    -- facts are returned as they stand.
    WITH found AS (
        SELECT e.id, e.scope_id, e.ingested_at
        FROM mempher.episodes e
        WHERE e.id = ANY(p_episodes) AND e.scope_id = p_scope
    ), marked AS (
        INSERT INTO mempher.fact_extractions
            (episode_id, extractor, scope_id, ingested_at, fact_count, extracted_at)
        SELECT f.id, p_extractor, f.scope_id, f.ingested_at,
               (SELECT count(*) FROM (SELECT DISTINCT subject, predicate, object, valid
                                      FROM mempher._assertions(p_assert)) d), p_at
        FROM found f
        ON CONFLICT (episode_id, extractor) DO NOTHING
        RETURNING episode_id
    )
    SELECT (SELECT count(*) FROM found), (SELECT count(*) FROM marked)
    INTO v_found, v_marked;

    IF v_found < (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) THEN
        PERFORM mempher._fail('MP020', format('%s of %s episodes are not in scope %L',
            (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) - v_found,
            (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x), p_scope));
    END IF;

    IF v_marked > 0 THEN
        -- Retraction closes a window; it never deletes. A fact that had
        -- already stopped being true by then is left as it is, which is what
        -- makes replaying an old extraction harmless.
        FOR v_elem IN SELECT e FROM jsonb_array_elements(p_retract) AS a(e)
        LOOP
            v_fact := (v_elem->>'fact')::uuid;
            v_to := (v_elem->>'at')::timestamptz;
            UPDATE mempher.facts
            SET valid = tstzrange(lower(valid), v_to, '[)'), updated_at = p_at
            WHERE id = v_fact AND scope_id = p_scope
              AND lower(valid) < v_to
              AND (upper_inf(valid) OR upper(valid) > v_to);
            IF FOUND THEN
                CONTINUE;
            END IF;
            SELECT lower(valid) INTO v_lower FROM mempher.facts WHERE id = v_fact AND scope_id = p_scope;
            IF NOT FOUND THEN
                PERFORM mempher._fail('MP020', format('retract fact %s: no such fact in scope %L', v_fact, p_scope));
            ELSIF v_to <= v_lower THEN
                PERFORM mempher._fail('MP011', format(
                    'fact %s stops being true at %s, but only became true at %s', v_fact, v_to, v_lower));
            END IF;
        END LOOP;

        -- A claim already held over exactly this window gains this
        -- extraction's episodes as provenance and its latest wording; any
        -- other claim is a new row. A new row that overlaps a window the
        -- scope already believes is the conflict WITHOUT OVERLAPS exists to
        -- catch.
        BEGIN
            WITH a AS (
                SELECT DISTINCT ON (x.subject, x.predicate, x.object, x.valid) x.*
                FROM mempher._assertions(p_assert) x
                ORDER BY x.subject, x.predicate, x.object, x.valid, x.ord
            ), updated AS (
                UPDATE mempher.facts f
                SET statement   = a.statement,
                    confidence  = a.confidence,
                    episode_ids = (SELECT array_agg(DISTINCT e ORDER BY e) FROM unnest(f.episode_ids || p_episodes) AS e),
                    updated_at  = p_at
                FROM a
                WHERE f.scope_id = p_scope AND f.extractor = p_extractor
                  AND (f.subject, f.predicate, f.object) = (a.subject, a.predicate, a.object)
                  AND f.valid = a.valid
                RETURNING a.ord
            )
            INSERT INTO mempher.facts
                (scope_id, extractor, subject, predicate, object, statement, valid,
                 confidence, episode_ids, asserted_at, updated_at)
            SELECT p_scope, p_extractor, a.subject, a.predicate, a.object, a.statement, a.valid,
                   a.confidence, p_episodes, p_at, p_at
            FROM a
            WHERE a.ord NOT IN (SELECT u.ord FROM updated u);
        EXCEPTION WHEN exclusion_violation THEN
            PERFORM mempher._fail('MP022',
                'an assertion overlaps a window the scope already believes for the same claim',
                'retract the existing fact first, or assert the window it does not cover');
        END;
    END IF;

    RETURN QUERY
    SELECT * FROM mempher.facts
    WHERE scope_id = p_scope AND extractor = p_extractor AND episode_ids && p_episodes
    ORDER BY subject, predicate, object, lower(valid);
END $$;

COMMENT ON FUNCTION mempher.apply_extraction(text, text, uuid[], jsonb, jsonb, timestamptz) IS
    'Records one extraction of some episodes atomically: their read markers, the '
    'retractions (each {"fact": uuid, "at": timestamp} closes that fact''s window), and the '
    'assertions (each {"subject", "predicate", "object", "statement", "valid_from", '
    '"valid_to"?, "confidence"?}). Applying the same extraction twice changes nothing the '
    'second time. Returns every fact the extraction''s episodes support.';

-- ---------------------------------------------------------------------------
-- Facts: reading L1
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.search_facts(
    p_scope          text,
    p_extractor      text,
    p_at             timestamptz DEFAULT now(),
    p_as_of          timestamptz DEFAULT now(),
    p_text           text        DEFAULT NULL,
    p_limit          int         DEFAULT 200,
    p_subjects       text[]      DEFAULT NULL,
    p_predicates     text[]      DEFAULT NULL,
    p_min_confidence real        DEFAULT 0,
    p_include_closed boolean     DEFAULT false
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_query tsquery;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    PERFORM mempher._check_at(p_as_of, 'as_of');
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_min_confidence IS NULL OR p_min_confidence < 0 OR p_min_confidence > 1 THEN
        PERFORM mempher._fail('MP013', format('min_confidence is %s, must be in [0,1]', p_min_confidence));
    END IF;
    IF octet_length(p_text) > 1048576 THEN
        PERFORM mempher._fail('MP002', 'query text is longer than 1048576 bytes');
    END IF;

    -- The text ranks, it never filters: relevance decides which facts survive
    -- the limit, not which are true. A fact the question does not mention is
    -- still a fact about the scope. Subject, predicate and object close the
    -- ordering, so equal ranks come back in the same order every run.
    v_query := mempher._any_lexeme_query(p_text);
    RETURN QUERY
    SELECT f.* FROM mempher.facts f
    WHERE f.scope_id = p_scope AND f.extractor = p_extractor
      AND f.asserted_at <= p_as_of
      AND CASE WHEN p_include_closed THEN lower(f.valid) <= p_at ELSE f.valid @> p_at END
      AND (p_subjects IS NULL OR cardinality(p_subjects) = 0 OR f.subject = ANY(p_subjects))
      AND (p_predicates IS NULL OR cardinality(p_predicates) = 0 OR f.predicate = ANY(p_predicates))
      AND f.confidence >= p_min_confidence
    ORDER BY ts_rank_cd(f.statement_tsv, v_query) DESC, f.subject, f.predicate, f.object, lower(f.valid)
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.search_facts(text, text, timestamptz, timestamptz, text, int, text[], text[], real, boolean) IS
    'Returns the facts one extractor holds about a scope that are true at p_at, as known '
    'by p_as_of, ranked by relevance to p_text when given. With p_include_closed, also '
    'those that began by p_at and have since stopped being true.';
