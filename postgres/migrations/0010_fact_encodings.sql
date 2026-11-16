-- 0010_fact_encodings.sql: facts are embedded, and recalled by meaning.
--
-- Until here a fact was ranked against a question by the words the two share,
-- so "where does she live" could not find "Ana moved to Lisbon". A fact's
-- statement now has a vector projection, as an episode's content has had since
-- 0001, and search_facts and recall rank by both.
--
-- The work is queued by the database: apply_extraction enqueues an
-- encode_facts job when it leaves a fact with no vector. A worker in any
-- language leases it, reads pending_fact_encodings, embeds the statements, and
-- hands them to put_fact_encodings.
--
-- Rendered as a Go text/template: {{.VectorDimensions}} is the deployment's, as
-- in 0001. Conventions and error codes are those of 0005_api_append.sql.

-- ---------------------------------------------------------------------------
-- The projection
-- ---------------------------------------------------------------------------

-- Keyed by (fact, model) like episode_encodings, so a change of model is new
-- rows rather than an UPDATE. A row here always encodes the statement the fact
-- holds now: put_fact_encodings refuses a vector made from an older statement,
-- and the triggers below remove the vector when the statement changes or the
-- fact goes.
--
-- There is no ANN index. A fact search is confined to one scope and one
-- extractor, and already reads every fact that is true at the instant asked
-- about, because the question ranks facts and never filters them. The distance
-- is computed over that same set, exactly.
--
-- There is no foreign key to mempher.facts either, and that is deliberate: L1
-- is repaired by truncating it and replaying L0, and PostgreSQL refuses to
-- truncate a table a foreign key points at. The triggers give what the key
-- would have: no vector outlives its fact.
CREATE TABLE mempher.fact_encodings (
    fact_id       uuid        NOT NULL,
    model         text        NOT NULL,
    statement_vec vector({{.VectorDimensions}}) NOT NULL,
    encoded_at    timestamptz NOT NULL,

    CONSTRAINT fact_encodings_pkey      PRIMARY KEY (fact_id, model),
    CONSTRAINT fact_encodings_model_len CHECK (length(model) BETWEEN 1 AND 200)
);

CREATE FUNCTION mempher._drop_fact_encodings()
    RETURNS trigger
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        TRUNCATE mempher.fact_encodings;
    ELSE
        DELETE FROM mempher.fact_encodings WHERE fact_id = OLD.id;
    END IF;
    RETURN NULL;
END $$;

-- A re-asserted claim may arrive with a reworded statement. The old vector
-- then describes text the fact no longer holds, and the fact is pending again.
CREATE TRIGGER facts_statement_changed
    AFTER UPDATE OF statement ON mempher.facts
    FOR EACH ROW
    WHEN (OLD.statement IS DISTINCT FROM NEW.statement)
    EXECUTE FUNCTION mempher._drop_fact_encodings();

-- Erasure, by mempher.forget or by hand.
CREATE TRIGGER facts_deleted
    AFTER DELETE ON mempher.facts
    FOR EACH ROW
    EXECUTE FUNCTION mempher._drop_fact_encodings();

-- Dropping the projection to rebuild it.
CREATE TRIGGER facts_truncated
    AFTER TRUNCATE ON mempher.facts
    FOR EACH STATEMENT
    EXECUTE FUNCTION mempher._drop_fact_encodings();


-- ---------------------------------------------------------------------------
-- The job kind
-- ---------------------------------------------------------------------------

ALTER TABLE mempher.jobs DROP CONSTRAINT jobs_kind_valid;
ALTER TABLE mempher.jobs ADD CONSTRAINT jobs_kind_valid
    CHECK (kind IN ('encode', 'extract', 'encode_facts'));

CREATE OR REPLACE FUNCTION mempher._check_job_kinds(p_kinds text[])
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
    WHERE k IS NULL OR k NOT IN ('encode', 'extract', 'encode_facts') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP008', format(
            'job kind %L is not one of encode, extract, encode_facts', v_bad));
    END IF;
    RETURN ARRAY(
        SELECT k FROM unnest(p_kinds) WITH ORDINALITY AS u(k, ord)
        GROUP BY k ORDER BY min(ord));
END $$;

-- pending_episodes answers for the two kinds whose work is per episode. It is
-- replaced only to say so: before this migration every kind was one of them.
CREATE OR REPLACE FUNCTION mempher.pending_episodes(
    p_kind      text,
    p_target    text,
    p_scope     text   DEFAULT NULL,
    p_after_seq bigint DEFAULT 0,
    p_limit     int    DEFAULT 100
)
    RETURNS TABLE (scope_id text, seq bigint, id uuid)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    IF p_kind IS NULL OR p_kind NOT IN ('encode', 'extract') THEN
        PERFORM mempher._fail('MP008', format(
            'pending_episodes answers for encode and extract, not %s',
            coalesce(quote_literal(p_kind), 'null')),
            'facts with no vector are listed by pending_fact_encodings');
    END IF;
    PERFORM mempher._check_label(p_target,
        CASE p_kind WHEN 'encode' THEN 'model' ELSE 'extractor' END, 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    IF p_after_seq IS NULL OR p_after_seq < 0 THEN
        PERFORM mempher._fail('MP013', format('after_seq is %s, must be at least 0', p_after_seq));
    ELSIF p_after_seq > 0 AND p_scope IS NULL THEN
        PERFORM mempher._fail('MP013', 'after_seq needs a scope, because seq is only ordered within one');
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;

    IF p_kind = 'encode' THEN
        RETURN QUERY
        SELECT e.scope_id, e.seq, e.id FROM mempher.episodes e
        WHERE (p_scope IS NULL OR e.scope_id = p_scope) AND e.seq > p_after_seq
          AND NOT EXISTS (SELECT 1 FROM mempher.episode_encodings x
                          WHERE x.episode_id = e.id AND x.model = p_target)
        ORDER BY e.scope_id, e.seq
        LIMIT p_limit;
    ELSE
        RETURN QUERY
        SELECT e.scope_id, e.seq, e.id FROM mempher.episodes e
        WHERE (p_scope IS NULL OR e.scope_id = p_scope) AND e.seq > p_after_seq
          AND NOT EXISTS (SELECT 1 FROM mempher.fact_extractions x
                          WHERE x.episode_id = e.id AND x.extractor = p_target)
        ORDER BY e.scope_id, e.seq
        LIMIT p_limit;
    END IF;
END $$;


-- ---------------------------------------------------------------------------
-- Writing and finding the work
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.pending_fact_encodings(
    p_model    text,
    p_scope    text   DEFAULT NULL,
    p_episodes uuid[] DEFAULT NULL,
    p_after    uuid   DEFAULT NULL,
    p_limit    int    DEFAULT 100
)
    RETURNS TABLE (id uuid, scope_id text, statement text)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    IF array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'pending_fact_encodings names a null episode');
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;

    RETURN QUERY
    SELECT f.id, f.scope_id, f.statement FROM mempher.facts f
    WHERE (p_scope IS NULL OR f.scope_id = p_scope)
      AND (p_episodes IS NULL OR cardinality(p_episodes) = 0 OR f.episode_ids && p_episodes)
      AND (p_after IS NULL OR f.id > p_after)
      AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x
                      WHERE x.fact_id = f.id AND x.model = p_model)
    ORDER BY f.id
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.pending_fact_encodings(text, text, uuid[], uuid, int) IS
    'Lists facts with no vector by model p_model, in id order, with the statement to '
    'embed. p_scope and p_episodes narrow it to one scope and to facts those episodes '
    'support, which is what an encode_facts job asks for. Page with p_after.';

CREATE FUNCTION mempher.put_fact_encodings(
    p_model      text,
    p_facts      uuid[],
    p_statements text[],
    p_vectors    vector[],
    p_at         timestamptz DEFAULT now()
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_n     int := coalesce(cardinality(p_facts), 0);
    v_bad   int;
    v_count int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    IF v_n = 0 OR array_position(p_facts, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'name at least one fact, and no null');
    END IF;
    IF (SELECT count(DISTINCT x) FROM unnest(p_facts) AS x) <> v_n THEN
        PERFORM mempher._fail('MP013', 'a fact is named twice');
    END IF;
    IF coalesce(cardinality(p_statements), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s statements for %s facts',
            coalesce(cardinality(p_statements), 0), v_n));
    END IF;
    IF coalesce(cardinality(p_vectors), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s vectors for %s facts',
            coalesce(cardinality(p_vectors), 0), v_n));
    END IF;
    SELECT o INTO v_bad FROM unnest(p_vectors) WITH ORDINALITY AS u(v, o)
    WHERE v IS NULL OR vector_dims(v) <> {{.VectorDimensions}} LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP021', format(
            'vector %s from model %L has %s dimensions but the schema was migrated for %s',
            v_bad, p_model, vector_dims(p_vectors[v_bad]), {{.VectorDimensions}}));
    END IF;

    -- The statement is compared under a row lock. The caller read it before
    -- its model call, and since then an extraction may have reworded the fact
    -- or an erasure removed it. The lock makes that change either visible
    -- here, so the vector is skipped, or wait until this row exists for the
    -- change's trigger to remove.
    WITH held AS (
        SELECT f.id, f.statement FROM mempher.facts f
        WHERE f.id = ANY(p_facts)
        ORDER BY f.id
        FOR SHARE
    )
    INSERT INTO mempher.fact_encodings (fact_id, model, statement_vec, encoded_at)
    SELECT h.id, p_model, u.v, p_at
    FROM unnest(p_facts, p_statements, p_vectors) AS u(id, s, v)
    JOIN held h ON h.id = u.id AND h.statement = u.s
    ON CONFLICT (fact_id, model) DO UPDATE
    SET statement_vec = excluded.statement_vec,
        encoded_at    = excluded.encoded_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.put_fact_encodings(text, uuid[], text[], vector[], timestamptz) IS
    'Writes one vector per fact for a model and returns how many were written. The '
    'arrays are parallel, and p_statements is the text each vector was made from: a fact '
    'that has gone, or whose statement has changed since it was read, is skipped rather '
    'than given a vector of text it does not hold.';

CREATE FUNCTION mempher.enqueue_fact_encodings(
    p_model        text,
    p_scope        text        DEFAULT NULL,
    p_at           timestamptz DEFAULT now(),
    p_run_after    timestamptz DEFAULT NULL,
    p_max_attempts int         DEFAULT 5
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_count int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    PERFORM mempher._check_at(p_at);
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;

    -- A job names the first episode that supports a pending fact, not all of
    -- them: a fact many episodes agree on would otherwise name a list too long
    -- for the index that collapses duplicate jobs. The job embeds every
    -- pending fact that episode supports, so the grouping loses nothing.
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    SELECT DISTINCT 'encode_facts', f.scope_id, ARRAY[f.episode_ids[1]],
           coalesce(p_run_after, p_at), p_max_attempts, p_at, p_at
    FROM mempher.facts f
    WHERE (p_scope IS NULL OR f.scope_id = p_scope)
      AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x
                      WHERE x.fact_id = f.id AND x.model = p_model)
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.enqueue_fact_encodings(text, text, timestamptz, timestamptz, int) IS
    'Queues encode_facts jobs for every fact with no vector by model p_model, in one '
    'scope or in all of them, and returns how many jobs it added. This is the backfill: '
    'for facts extracted before this migration, after a change of embedding model, or '
    'when a job died. Running it twice adds nothing the second time.';


-- ---------------------------------------------------------------------------
-- apply_extraction queues the work
-- ---------------------------------------------------------------------------

-- The function from 0007 keeps its body and becomes internal. The public name
-- now wraps it, which is how a forward-only migration adds a step to a
-- function without restating it.
ALTER FUNCTION mempher.apply_extraction(text, text, uuid[], jsonb, jsonb, timestamptz)
    RENAME TO _apply_extraction;

COMMENT ON FUNCTION mempher._apply_extraction(text, text, uuid[], jsonb, jsonb, timestamptz) IS
    'Internal: the body of mempher.apply_extraction, without the job it queues.';

CREATE FUNCTION mempher.apply_extraction(
    p_scope     text,
    p_extractor text,
    p_episodes  uuid[],
    p_assert    jsonb       DEFAULT '[]',
    p_retract   jsonb       DEFAULT '[]',
    p_at        timestamptz DEFAULT now(),
    p_encode    boolean     DEFAULT true
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM mempher._apply_extraction(p_scope, p_extractor, p_episodes, p_assert, p_retract, p_at);

    -- "No vector" is asked of every model at once, because the database does
    -- not know which one the deployment embeds with. It is exact for the usual
    -- deployment, which has one; a second model is enqueue_fact_encodings' job.
    IF coalesce(p_encode, true) AND EXISTS (
        SELECT 1 FROM mempher.facts f
        WHERE f.scope_id = p_scope AND f.episode_ids && p_episodes
          AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x WHERE x.fact_id = f.id)
    ) THEN
        INSERT INTO mempher.jobs
            (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
        VALUES ('encode_facts', p_scope, p_episodes, p_at, 5, p_at, p_at)
        ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
            DO NOTHING;
    END IF;
END $$;

COMMENT ON FUNCTION mempher.apply_extraction(text, text, uuid[], jsonb, jsonb, timestamptz, boolean) IS
    'Records one extraction of some episodes atomically: their read markers, the '
    'retractions (each {"fact": uuid, "at": timestamp} closes that fact''s window), and the '
    'assertions (each {"subject", "predicate", "object", "statement", "valid_from", '
    '"valid_to"?, "confidence"?}). Applying the same extraction twice changes nothing the '
    'second time. Returns every fact the extraction''s episodes support, and, unless '
    'p_encode is false, queues an encode_facts job when one of them has no vector.';


-- ---------------------------------------------------------------------------
-- Ranking facts by meaning
-- ---------------------------------------------------------------------------

-- The weights a fusion runs with: the caller's, checked, with the lexical
-- channel at a quarter of a vote unless they say otherwise. That default was
-- calibrated on LongMemEval for episodes, where an equal vote let keyword
-- overlap outrank meaning; facts use it unmeasured.
CREATE FUNCTION mempher._fusion_weights(p_weights jsonb)
    RETURNS jsonb
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_weights jsonb := coalesce(p_weights, '{}');
    v_bad     text;
BEGIN
    IF jsonb_typeof(v_weights) <> 'object' THEN
        PERFORM mempher._fail('MP013', 'weights is not a JSON object');
    END IF;
    SELECT key INTO v_bad FROM jsonb_each(v_weights)
    WHERE key NOT IN ('semantic', 'lexical', 'fact')
       OR jsonb_typeof(value) <> 'number' OR value::float8 < 0
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format(
            'weight %L must name a channel and be a number at least 0', v_bad));
    END IF;
    IF NOT v_weights ? 'lexical' THEN
        v_weights := v_weights || '{"lexical": 0.25}';
    END IF;
    RETURN v_weights;
END $$;

CREATE FUNCTION mempher._check_fusion_k(p_fusion_k float8)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_fusion_k IS NULL OR p_fusion_k < 0 OR p_fusion_k = 'Infinity' OR p_fusion_k = 'NaN' THEN
        PERFORM mempher._fail('MP013', format('fusion_k is %s, must be a finite number at least 0', p_fusion_k));
    END IF;
END $$;

-- A query's embedding comes with the model that made it and has the width the
-- schema was migrated for.
CREATE FUNCTION mempher._check_query_embedding(p_embedding vector, p_model text)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF vector_dims(p_embedding) <> {{.VectorDimensions}} THEN
        PERFORM mempher._fail('MP021', format(
            'query vector has %s dimensions but the schema was migrated for %s',
            vector_dims(p_embedding), {{.VectorDimensions}}));
    END IF;
END $$;

-- search_facts gains parameters, which CREATE OR REPLACE cannot add, and a
-- second overload would make every short call ambiguous.
DROP FUNCTION mempher.search_facts(text, text, timestamptz, timestamptz, text, int, text[], text[], real, boolean);

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
    p_include_closed boolean     DEFAULT false,
    p_embedding      vector      DEFAULT NULL,
    p_model          text        DEFAULT NULL,
    p_weights        jsonb       DEFAULT '{}',
    p_fusion_k       float8      DEFAULT 60
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_query   tsquery;
    v_weights jsonb;
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
    IF p_embedding IS NOT NULL THEN
        PERFORM mempher._check_query_embedding(p_embedding, p_model);
    END IF;
    PERFORM mempher._check_fusion_k(p_fusion_k);
    v_weights := mempher._fusion_weights(p_weights);
    v_query := mempher._any_lexeme_query(p_text);

    -- The question ranks, it never filters: relevance decides which facts
    -- survive the limit, not which are true. A fact the question does not
    -- mention is still a fact about the scope.
    --
    -- Two rankings vote, by reciprocal rank as recall fuses episodes: nearness
    -- of the statement's vector to p_embedding, and the words the statement
    -- shares with p_text. A fact with no vector yet, or sharing no word, has no
    -- vote from that ranking. With no embedding the order is the lexical one.
    -- Subject, predicate and object close the ordering, so equal scores come
    -- back in the same order every run.
    RETURN QUERY
    WITH candidates AS (
        SELECT f AS fact,
               ts_rank_cd(f.statement_tsv, v_query) AS lexical,
               x.statement_vec <=> p_embedding AS distance
        FROM mempher.facts f
        LEFT JOIN mempher.fact_encodings x
               ON p_embedding IS NOT NULL AND x.fact_id = f.id AND x.model = p_model
        WHERE f.scope_id = p_scope AND f.extractor = p_extractor
          AND f.asserted_at <= p_as_of
          AND CASE WHEN p_include_closed THEN lower(f.valid) <= p_at ELSE f.valid @> p_at END
          AND (p_subjects IS NULL OR cardinality(p_subjects) = 0 OR f.subject = ANY(p_subjects))
          AND (p_predicates IS NULL OR cardinality(p_predicates) = 0 OR f.predicate = ANY(p_predicates))
          AND f.confidence >= p_min_confidence
    ), scored AS (
        SELECT c.fact, c.lexical,
               CASE WHEN c.distance IS NOT NULL
                    THEN coalesce((v_weights->>'semantic')::float8, 1)
                         / (p_fusion_k + rank() OVER (ORDER BY c.distance))
                    ELSE 0 END
             + CASE WHEN c.lexical > 0
                    THEN (v_weights->>'lexical')::float8
                         / (p_fusion_k + rank() OVER (ORDER BY c.lexical DESC))
                    ELSE 0 END AS score
        FROM candidates c
    )
    SELECT (s.fact).* FROM scored s
    ORDER BY s.score DESC, s.lexical DESC,
             (s.fact).subject, (s.fact).predicate, (s.fact).object, lower((s.fact).valid)
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.search_facts(text, text, timestamptz, timestamptz, text, int, text[], text[],
                                         real, boolean, vector, text, jsonb, float8) IS
    'Returns the facts one extractor holds about a scope that are true at p_at, as known '
    'by p_as_of. With p_include_closed, also those that began by p_at and have since '
    'stopped being true. The question orders them and never excludes one: by meaning '
    'when p_embedding is given (the question embedded by p_model), by shared words when '
    'p_text is, and by both fused by reciprocal rank, weighted by p_weights (lexical '
    'defaults to 0.25).';


-- ---------------------------------------------------------------------------
-- recall
-- ---------------------------------------------------------------------------

-- Restated from 0008 with three changes: the fact channel passes the query's
-- embedding to search_facts, the embedding is checked whenever it is given
-- rather than only for the semantic channel, and the checks that search_facts
-- now shares are the helpers above.
CREATE OR REPLACE FUNCTION mempher.recall(
    p_scope               text,
    p_query               text,
    p_embedding           vector      DEFAULT NULL,
    p_model               text        DEFAULT NULL,
    p_extractor           text        DEFAULT NULL,
    p_channels            text[]      DEFAULT NULL,
    p_limit               int         DEFAULT 10,
    p_candidates          int         DEFAULT 50,
    p_fact_limit          int         DEFAULT 32,
    p_max_tokens          int         DEFAULT NULL,
    p_as_of               timestamptz DEFAULT now(),
    p_occurred_from       timestamptz DEFAULT NULL,
    p_occurred_to         timestamptz DEFAULT NULL,
    p_binding             jsonb       DEFAULT NULL,
    p_roles               text[]      DEFAULT NULL,
    p_subjects            text[]      DEFAULT NULL,
    p_min_fact_confidence real        DEFAULT 0,
    p_weights             jsonb       DEFAULT '{}',
    p_fusion_k            float8      DEFAULT 60,
    p_bytes_per_token     int         DEFAULT 4
)
    RETURNS jsonb
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
    -- Without iterative index scans an HNSW search filtered to one scope
    -- returns fewer neighbours than asked for, and on a large database can
    -- return none at all: the index finds the nearest vectors anywhere, and the
    -- scope filter then discards them. strict_order keeps scanning until the
    -- limit is met, in distance order.
    SET hnsw.iterative_scan = strict_order
AS $$
DECLARE
    v_channels    text[];
    v_bad         text;
    v_weights     jsonb;
    v_candidates  int;
    v_tsq         tsquery;
    v_t0          timestamptz;
    v_reports     jsonb := '[]';
    v_sem_ids     uuid[]   := '{}';
    v_sem_scores  float8[] := '{}';
    v_lex_ids     uuid[]   := '{}';
    v_lex_scores  float8[] := '{}';
    v_fact_ids    uuid[]   := '{}';
    v_facts       jsonb    := '[]';
    v_episodes    jsonb    := '[]';
    v_fact_tokens bigint   := 0;
    v_ep_tokens   bigint   := 0;
    v_truncated   boolean  := false;
BEGIN
    -- ---------------------------------------------------------------- checks
    PERFORM mempher._check_scope(p_scope);
    IF octet_length(p_query) > 1048576 THEN
        PERFORM mempher._fail('MP002', format('recall query is %s bytes, limit is 1048576',
            octet_length(p_query)));
    END IF;
    PERFORM mempher._check_at(p_as_of, 'as_of');

    SELECT c INTO v_bad FROM unnest(p_channels) AS c
    WHERE c IS NULL OR c NOT IN ('semantic', 'lexical', 'fact') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP007', format('channel %s is not one of semantic, lexical, fact',
            coalesce(quote_literal(v_bad), 'null')));
    END IF;
    IF p_channels IS NULL OR cardinality(p_channels) = 0 THEN
        -- Every channel the arguments make possible.
        v_channels := array_remove(ARRAY[
            CASE WHEN p_embedding IS NOT NULL THEN 'semantic' END,
            CASE WHEN btrim(p_query) <> '' THEN 'lexical' END,
            CASE WHEN p_extractor IS NOT NULL THEN 'fact' END], NULL);
        IF cardinality(v_channels) = 0 THEN
            PERFORM mempher._fail('MP002', 'nothing to recall with: give a query, an embedding, or both');
        END IF;
    ELSE
        v_channels := ARRAY(SELECT c FROM unnest(p_channels) WITH ORDINALITY AS u(c, o)
                            GROUP BY c ORDER BY min(o));
    END IF;
    -- Words are what the lexical channel matches; the fact channel only
    -- ranks by them, and the semantic channel has its embedding instead.
    IF 'lexical' = ANY(v_channels) AND coalesce(btrim(p_query), '') = '' THEN
        PERFORM mempher._fail('MP002', 'the lexical channel needs a query');
    END IF;

    IF 'semantic' = ANY(v_channels) AND p_embedding IS NULL THEN
        PERFORM mempher._fail('MP013', 'the semantic channel needs the query''s embedding');
    END IF;
    -- Checked whichever channels run, because the fact channel ranks by it too.
    IF p_embedding IS NOT NULL THEN
        PERFORM mempher._check_query_embedding(p_embedding, p_model);
    END IF;
    IF 'fact' = ANY(v_channels) THEN
        IF p_extractor IS NULL THEN
            PERFORM mempher._fail('MP013', 'the fact channel needs the extractor whose facts to read');
        END IF;
        PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    END IF;

    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_candidates IS NULL OR p_candidates < 1 THEN
        PERFORM mempher._fail('MP013', format('candidates is %s, must be positive', p_candidates));
    END IF;
    IF p_fact_limit IS NULL OR p_fact_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('fact_limit is %s, must be positive', p_fact_limit));
    END IF;
    IF p_max_tokens IS NOT NULL AND p_max_tokens < 1 THEN
        PERFORM mempher._fail('MP013', format('max_tokens is %s; leave it null for no budget', p_max_tokens));
    END IF;
    IF p_bytes_per_token IS NULL OR p_bytes_per_token < 1 THEN
        PERFORM mempher._fail('MP013', format('bytes_per_token is %s, must be positive', p_bytes_per_token));
    END IF;
    IF p_min_fact_confidence IS NULL OR p_min_fact_confidence < 0 OR p_min_fact_confidence > 1 THEN
        PERFORM mempher._fail('MP013', format('min_fact_confidence is %s, must be in [0,1]', p_min_fact_confidence));
    END IF;
    PERFORM mempher._check_fusion_k(p_fusion_k);
    -- An inverted window matches nothing, which is far more likely to be
    -- swapped arguments than an intentional empty question.
    IF p_occurred_from IS NOT NULL AND p_occurred_to IS NOT NULL AND p_occurred_to < p_occurred_from THEN
        PERFORM mempher._fail('MP013', format('the event-time window ends (%s) before it starts (%s)',
            p_occurred_to, p_occurred_from));
    END IF;
    IF p_binding IS NOT NULL THEN
        PERFORM mempher._check_binding(p_binding, 'recall');
    END IF;
    SELECT r INTO v_bad FROM unnest(p_roles) AS r
    WHERE r IS NULL OR r NOT IN ('user', 'assistant', 'system', 'tool', 'observation') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP003', format('role filter %s is not a role', coalesce(quote_literal(v_bad), 'null')));
    END IF;
    FOR v_bad IN SELECT s FROM unnest(p_subjects) AS s LOOP
        PERFORM mempher._check_label(v_bad, 'subject', 'MP010', 200);
    END LOOP;

    v_weights := mempher._fusion_weights(p_weights);

    -- A channel is asked for at least as many candidates as will be returned,
    -- or a limit above the depth could never be met.
    v_candidates := greatest(p_candidates, p_limit);
    v_tsq := mempher._any_lexeme_query(p_query);

    -- -------------------------------------------------------------- channels
    IF 'semantic' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        SELECT coalesce(array_agg(c.id ORDER BY c.score DESC, c.id DESC), '{}'),
               coalesce(array_agg(c.score ORDER BY c.score DESC, c.id DESC), '{}')
        INTO v_sem_ids, v_sem_scores
        FROM (
            SELECT e.id, 1 - (enc.content_vec <=> p_embedding) AS score
            FROM mempher.episode_encodings enc
            JOIN mempher.episodes e ON e.id = enc.episode_id
            WHERE enc.model = p_model AND enc.scope_id = p_scope AND enc.ingested_at <= p_as_of
              AND (p_occurred_from IS NULL OR e.occurred_at >= p_occurred_from)
              AND (p_occurred_to IS NULL OR e.occurred_at <= p_occurred_to)
              AND (p_binding IS NULL OR e.binding @> p_binding)
              AND (p_roles IS NULL OR cardinality(p_roles) = 0 OR e.role = ANY(p_roles))
            ORDER BY enc.content_vec <=> p_embedding
            LIMIT v_candidates
        ) c;
        v_reports := v_reports || jsonb_build_object('channel', 'semantic',
            'candidates', cardinality(v_sem_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    IF 'lexical' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        SELECT coalesce(array_agg(c.id ORDER BY c.score DESC, c.id DESC), '{}'),
               coalesce(array_agg(c.score ORDER BY c.score DESC, c.id DESC), '{}')
        INTO v_lex_ids, v_lex_scores
        FROM (
            SELECT e.id, ts_rank_cd(e.content_tsv, v_tsq)::float8 AS score
            FROM mempher.episodes e
            WHERE e.scope_id = p_scope AND e.ingested_at <= p_as_of AND e.content_tsv @@ v_tsq
              AND (p_occurred_from IS NULL OR e.occurred_at >= p_occurred_from)
              AND (p_occurred_to IS NULL OR e.occurred_at <= p_occurred_to)
              AND (p_binding IS NULL OR e.binding @> p_binding)
              AND (p_roles IS NULL OR cardinality(p_roles) = 0 OR e.role = ANY(p_roles))
            ORDER BY score DESC, e.seq DESC
            LIMIT v_candidates
        ) c;
        v_reports := v_reports || jsonb_build_object('channel', 'lexical',
            'candidates', cardinality(v_lex_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    -- Facts are not ranked against episodes: they are what the scope currently
    -- believes, true at the end of the event-time window or else at as_of, and
    -- the question only decides which survive the limit. It is asked by
    -- meaning as well as by words whenever its embedding was given.
    IF 'fact' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        WITH f AS (
            SELECT x.*, x.ordinality AS pos
            FROM mempher.search_facts(p_scope, p_extractor, coalesce(p_occurred_to, p_as_of), p_as_of,
                                      p_query, p_fact_limit, p_subjects, NULL, p_min_fact_confidence,
                                      false, p_embedding, p_model, v_weights, p_fusion_k)
                 WITH ORDINALITY AS x
        ), costed AS (
            SELECT f.*, (octet_length(f.statement) + p_bytes_per_token - 1) / p_bytes_per_token AS tokens,
                   sum((octet_length(f.statement) + p_bytes_per_token - 1) / p_bytes_per_token)
                       OVER (ORDER BY f.pos) AS spent
            FROM f
        )
        SELECT coalesce(jsonb_agg(jsonb_build_object(
                   'id', c.id, 'scope', c.scope_id, 'extractor', c.extractor,
                   'subject', c.subject, 'predicate', c.predicate, 'object', c.object,
                   'statement', c.statement,
                   'valid_from', lower(c.valid), 'valid_to', upper(c.valid),
                   'confidence', c.confidence, 'episodes', to_jsonb(c.episode_ids),
                   'asserted_at', c.asserted_at, 'updated_at', c.updated_at,
                   'tokens', c.tokens) ORDER BY c.pos)
                   FILTER (WHERE p_max_tokens IS NULL OR c.spent <= p_max_tokens), '[]'),
               coalesce(max(c.spent) FILTER (WHERE p_max_tokens IS NULL OR c.spent <= p_max_tokens), 0),
               bool_or(p_max_tokens IS NOT NULL AND c.spent > p_max_tokens),
               coalesce(array_agg(c.id), '{}')
        INTO v_facts, v_fact_tokens, v_truncated, v_fact_ids
        FROM costed c;
        v_truncated := coalesce(v_truncated, false);
        v_reports := v_reports || jsonb_build_object('channel', 'fact',
            'candidates', cardinality(v_fact_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    -- ---------------------------------------------------------------- fusion
    -- Reciprocal rank fusion: an episode scores the sum, over the channels
    -- that found it, of its channel's weight over K plus its rank there. Ranks
    -- rather than scores, because a cosine similarity and a ts_rank_cd are not
    -- on any common scale. Ties go to the newer episode.
    WITH hits AS (
        SELECT 'semantic'::text AS channel, u.id, u.rank::int AS rank, u.score
        FROM unnest(v_sem_ids, v_sem_scores) WITH ORDINALITY AS u(id, score, rank)
        UNION ALL
        SELECT 'lexical', u.id, u.rank::int, u.score
        FROM unnest(v_lex_ids, v_lex_scores) WITH ORDINALITY AS u(id, score, rank)
    ), fused AS (
        SELECT h.id,
               sum(coalesce((v_weights->>h.channel)::float8, 1) / (p_fusion_k + h.rank)) AS score,
               jsonb_agg(jsonb_build_object('channel', h.channel, 'rank', h.rank, 'score', h.score)
                         ORDER BY h.channel) AS hits
        FROM hits h
        GROUP BY h.id
    ), ranked AS (
        SELECT f.*, row_number() OVER (ORDER BY f.score DESC, f.id DESC) AS pos
        FROM fused f
    ), costed AS (
        SELECT r.*, e.scope_id, e.seq, e.content, e.role, e.actor, e.source,
               e.occurred_at, e.ingested_at, e.binding,
               (octet_length(e.content) + p_bytes_per_token - 1) / p_bytes_per_token AS tokens,
               sum((octet_length(e.content) + p_bytes_per_token - 1) / p_bytes_per_token)
                   OVER (ORDER BY r.pos) AS spent
        FROM ranked r
        JOIN mempher.episodes e ON e.id = r.id
        WHERE r.pos <= p_limit
    )
    SELECT coalesce(jsonb_agg(jsonb_build_object(
               'id', c.id, 'scope', c.scope_id, 'seq', c.seq, 'content', c.content,
               'role', c.role, 'actor', c.actor, 'source', c.source,
               'occurred_at', c.occurred_at, 'ingested_at', c.ingested_at, 'binding', c.binding,
               'score', c.score, 'hits', c.hits, 'tokens', c.tokens) ORDER BY c.pos)
               FILTER (WHERE p_max_tokens IS NULL OR v_fact_tokens + c.spent <= p_max_tokens), '[]'),
           coalesce(max(c.spent) FILTER (WHERE p_max_tokens IS NULL OR v_fact_tokens + c.spent <= p_max_tokens), 0),
           v_truncated
               OR coalesce(bool_or(p_max_tokens IS NOT NULL AND v_fact_tokens + c.spent > p_max_tokens), false)
               OR (SELECT count(*) FROM ranked) > p_limit
    INTO v_episodes, v_ep_tokens, v_truncated
    FROM costed c;

    RETURN jsonb_build_object(
        'scope',     p_scope,
        'as_of',     p_as_of,
        'facts',     v_facts,
        'episodes',  v_episodes,
        'channels',  v_reports,
        'tokens',    v_fact_tokens + v_ep_tokens,
        'truncated', v_truncated);
END $$;

COMMENT ON FUNCTION mempher.recall(text, text, vector, text, text, text[], int, int, int, int,
                                   timestamptz, timestamptz, timestamptz, jsonb, text[], text[],
                                   real, jsonb, float8, int) IS
    'Recalls what a scope holds that bears on p_query, as of p_as_of, as one JSON document: '
    '{"facts": [...], "episodes": [...], "channels": [...], "tokens": n, "truncated": bool}. '
    'Channels default to every one the arguments allow: semantic needs p_embedding (the '
    'query embedded by p_model), lexical needs p_query, fact needs p_extractor and ranks '
    'facts by p_embedding as well as p_query when it is given. Episodes are '
    'fused by reciprocal rank, weighted by p_weights (lexical defaults to 0.25), and each '
    'carries the channels that found it. p_max_tokens, counted at p_bytes_per_token, is '
    'spent on facts first, then on episodes in order; what does not fit is left out and '
    'the result says it was truncated.';
