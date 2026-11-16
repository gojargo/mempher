-- 0008_api_recall.sql: the read path in one call.
--
-- mempher.recall runs every retrieval channel the caller asked for, fuses the
-- episode channels by reciprocal rank, reads what the scope believes from L1,
-- spends a token budget on both -- facts first -- and returns the result as one
-- JSON document, with the provenance of every hit. The only part of reading
-- that happens outside the database is embedding the question, because that is
-- a model call.
--
-- Rendered as a Go text/template: {{.VectorDimensions}} is the deployment's, as
-- in 0001. Conventions and error codes are those of 0005_api_append.sql.

CREATE FUNCTION mempher.recall(
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

    IF 'semantic' = ANY(v_channels) THEN
        IF p_embedding IS NULL THEN
            PERFORM mempher._fail('MP013', 'the semantic channel needs the query''s embedding');
        END IF;
        PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
        IF vector_dims(p_embedding) <> {{.VectorDimensions}} THEN
            PERFORM mempher._fail('MP021', format(
                'query vector has %s dimensions but the schema was migrated for %s',
                vector_dims(p_embedding), {{.VectorDimensions}}));
        END IF;
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
    IF p_fusion_k IS NULL OR p_fusion_k < 0 OR p_fusion_k = 'Infinity' OR p_fusion_k = 'NaN' THEN
        PERFORM mempher._fail('MP013', format('fusion_k is %s, must be a finite number at least 0', p_fusion_k));
    END IF;
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

    -- The lexical channel is a quarter of a vote unless the caller says
    -- otherwise: calibrated on LongMemEval, where an equal vote let keyword
    -- overlap outrank meaning.
    v_weights := coalesce(p_weights, '{}');
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
    -- the question only decides which survive the limit.
    IF 'fact' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        WITH f AS (
            SELECT x.*, x.ordinality AS pos
            FROM mempher.search_facts(p_scope, p_extractor, coalesce(p_occurred_to, p_as_of), p_as_of,
                                      p_query, p_fact_limit, p_subjects, NULL, p_min_fact_confidence)
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
    'query embedded by p_model), lexical needs p_query, fact needs p_extractor. Episodes are '
    'fused by reciprocal rank, weighted by p_weights (lexical defaults to 0.25), and each '
    'carries the channels that found it. p_max_tokens, counted at p_bytes_per_token, is '
    'spent on facts first, then on episodes in order; what does not fit is left out and '
    'the result says it was truncated.';
