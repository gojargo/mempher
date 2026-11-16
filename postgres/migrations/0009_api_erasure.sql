-- 0009_api_erasure.sql: erasure, and what L0 says is still owed.
--
-- mempher.forget is the one function here that destroys anything, and the only
-- one the append-only trigger on mempher.episodes lets through: the declaration
-- that opens the trigger is a SET clause on the function itself, so it holds
-- for exactly the duration of the call and cannot leak into the rest of the
-- caller's transaction.
--
-- mempher.pending_episodes is the safety net behind the queue: it asks L0 and a
-- projection table, never the queue's own history, which episodes still lack
-- an encoding or an extraction.
--
-- Conventions and error codes are those of 0005_api_append.sql.

CREATE FUNCTION mempher.forget(p_scope text, p_episodes uuid[] DEFAULT NULL)
    RETURNS TABLE (episodes int, encodings int, extractions int, facts int, jobs int,
                   scope_removed boolean)
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
    SET mempher.erasing = 'on'
AS $$
#variable_conflict use_column
DECLARE
    v_whole boolean := p_episodes IS NULL OR cardinality(p_episodes) = 0;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    IF NOT v_whole THEN
        IF cardinality(p_episodes) > 1000 THEN
            PERFORM mempher._fail('MP013', format('forget names %s episodes, limit is 1000',
                cardinality(p_episodes)), 'erase a longer list in several calls, or the whole scope');
        END IF;
        IF array_position(p_episodes, NULL) IS NOT NULL THEN
            PERFORM mempher._fail('MP013', 'forget names a null episode');
        END IF;
    END IF;

    -- Everything derived goes before what it was derived from. A fact is
    -- deleted rather than closed, because a closed window still says what it
    -- said; if what remains still supports the claim, the next extraction
    -- derives it again. The counts are the receipt.
    IF v_whole THEN
        DELETE FROM mempher.facts f WHERE f.scope_id = p_scope;
        GET DIAGNOSTICS facts = ROW_COUNT;
        DELETE FROM mempher.fact_extractions x WHERE x.scope_id = p_scope;
        GET DIAGNOSTICS extractions = ROW_COUNT;
        DELETE FROM mempher.episode_encodings c WHERE c.scope_id = p_scope;
        GET DIAGNOSTICS encodings = ROW_COUNT;
        DELETE FROM mempher.jobs j WHERE j.scope_id = p_scope;
        GET DIAGNOSTICS jobs = ROW_COUNT;
        DELETE FROM mempher.episodes e WHERE e.scope_id = p_scope;
        GET DIAGNOSTICS episodes = ROW_COUNT;
        DELETE FROM mempher.scopes s WHERE s.id = p_scope;
        scope_removed := FOUND;
    ELSE
        DELETE FROM mempher.facts f WHERE f.scope_id = p_scope AND f.episode_ids && p_episodes;
        GET DIAGNOSTICS facts = ROW_COUNT;
        DELETE FROM mempher.fact_extractions x WHERE x.scope_id = p_scope AND x.episode_id = ANY(p_episodes);
        GET DIAGNOSTICS extractions = ROW_COUNT;
        DELETE FROM mempher.episode_encodings c WHERE c.scope_id = p_scope AND c.episode_id = ANY(p_episodes);
        GET DIAGNOSTICS encodings = ROW_COUNT;
        DELETE FROM mempher.jobs j WHERE j.scope_id = p_scope AND j.episode_ids && p_episodes;
        GET DIAGNOSTICS jobs = ROW_COUNT;
        DELETE FROM mempher.episodes e WHERE e.scope_id = p_scope AND e.id = ANY(p_episodes);
        GET DIAGNOSTICS episodes = ROW_COUNT;
        scope_removed := false;
    END IF;
    RETURN NEXT;
END $$;

COMMENT ON FUNCTION mempher.forget(text, uuid[]) IS
    'Erases a whole scope, or the named episodes of one (at most 1000), together with '
    'every encoding, extraction marker, fact and job derived from them, in one '
    'transaction, and returns what went. Erasing what is not there removes nothing and '
    'is not an error. The only way rows leave mempher.episodes.';

CREATE FUNCTION mempher.pending_episodes(
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
    PERFORM mempher._check_job_kinds(ARRAY[p_kind]);
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

COMMENT ON FUNCTION mempher.pending_episodes(text, text, text, bigint, int) IS
    'Lists episodes with no encoding by model p_target (kind encode), or no extraction '
    'by extractor p_target (kind extract), in scope and seq order: the work L0 implies '
    'whatever the queue remembers. Page within a scope with p_after_seq.';
