-- 0006_api_queue.sql: the job queue as functions.
--
-- The queue's whole state machine lives here now: claiming with SKIP LOCKED,
-- the lease a worker must still hold to finish, the backoff between attempts,
-- the dead state an exhausted job lands in, and reclaiming a dead worker's
-- leases. A worker in any language is a loop around lease_jobs and one of
-- succeed_job or fail_job; it decides nothing about the queue itself.
--
-- Conventions and error codes are those of 0005_api_append.sql. Reading the
-- queue needs no function: mempher.jobs is a plain table, and selecting from it
-- is part of the contract.

-- ---------------------------------------------------------------------------
-- Internal helpers
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher._check_worker(p_worker text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_worker IS NULL OR p_worker = '' THEN
        PERFORM mempher._fail('MP013', 'worker is empty',
            'name the worker, so a lease says who holds it');
    END IF;
END $$;

CREATE FUNCTION mempher._check_at(p_at timestamptz, p_name text DEFAULT 'at')
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_at IS NULL OR NOT isfinite(p_at) THEN
        PERFORM mempher._fail('MP013', format('%s is %s, must be a finite timestamp',
            p_name, coalesce(p_at::text, 'null')));
    END IF;
END $$;

-- Says why a job the caller named could not be moved: it does not exist, or it
-- is not in the state the caller assumed. Costs a read, but only on a path that
-- is about to be investigated anyway.
CREATE FUNCTION mempher._explain_lost_lease(p_job uuid, p_worker text, p_op text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_state text;
    v_by    text;
BEGIN
    SELECT state, leased_by INTO v_state, v_by FROM mempher.jobs WHERE id = p_job;
    IF NOT FOUND THEN
        PERFORM mempher._fail('MP020', format('%s job %s: no such job', p_op, p_job));
    END IF;
    PERFORM mempher._fail('MP023', format(
        '%s job %s: worker %L does not hold it (state %s, held by %L)',
        p_op, p_job, p_worker, v_state, coalesce(v_by, '')),
        'the lease expired and the job was reclaimed, or another worker finished it');
END $$;

-- ---------------------------------------------------------------------------
-- Writing to the queue
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.enqueue(
    p_kind         text,
    p_scope        text,
    p_episodes     uuid[],
    p_at           timestamptz DEFAULT now(),
    p_run_after    timestamptz DEFAULT NULL,
    p_max_attempts int         DEFAULT 5
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_job_kinds(ARRAY[p_kind]);
    PERFORM mempher._check_at(p_at);
    IF p_episodes IS NULL OR cardinality(p_episodes) = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'a job must name at least one episode, and no null');
    END IF;
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM mempher.scopes WHERE id = p_scope) THEN
        PERFORM mempher._fail('MP020', format('scope %L has no episodes yet', p_scope));
    END IF;

    -- Duplicate work collapses onto the job already holding it. The loop covers
    -- the one race there is: the outstanding job finishing between the
    -- conflicting insert and the read, which frees the slot for this one.
    LOOP
        INSERT INTO mempher.jobs
            (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
        VALUES (p_kind, p_scope, p_episodes, coalesce(p_run_after, p_at), p_max_attempts, p_at, p_at)
        ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
            DO NOTHING
        RETURNING * INTO v_job;
        IF FOUND THEN
            RETURN v_job;
        END IF;
        SELECT * INTO v_job FROM mempher.jobs
        WHERE kind = p_kind AND scope_id = p_scope AND episode_ids = p_episodes
          AND state IN ('pending', 'running');
        IF FOUND THEN
            RETURN v_job;
        END IF;
    END LOOP;
END $$;

COMMENT ON FUNCTION mempher.enqueue(text, text, uuid[], timestamptz, timestamptz, int) IS
    'Queues one job of a kind over episodes of a scope. Work identical to a job still '
    'pending or running collapses onto it, and that job is returned instead.';

-- ---------------------------------------------------------------------------
-- Running the queue
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.lease_jobs(
    p_worker text,
    p_limit  int         DEFAULT 1,
    p_kinds  text[]      DEFAULT NULL,
    p_lease  interval    DEFAULT '5 minutes',
    p_at     timestamptz DEFAULT now()
)
    RETURNS SETOF mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_kinds text[];
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_lease IS NULL OR p_lease <= interval '0' THEN
        PERFORM mempher._fail('MP013', format('lease is %s, must be positive', p_lease));
    END IF;
    v_kinds := mempher._check_job_kinds(p_kinds);

    -- The attempt is counted here, at lease time, rather than when it fails: a
    -- worker that dies mid-job never reports, and its attempt must still count
    -- or a job that kills its worker would be retried for ever.
    RETURN QUERY
    WITH claimed AS (
        UPDATE mempher.jobs j SET
            state        = 'running',
            attempts     = j.attempts + 1,
            leased_until = p_at + p_lease,
            leased_by    = p_worker,
            updated_at   = p_at
        WHERE j.id IN (
            SELECT c.id FROM mempher.jobs c
            WHERE c.state = 'pending' AND c.run_after <= p_at AND c.attempts < c.max_attempts
              AND (cardinality(v_kinds) = 0 OR c.kind = ANY(v_kinds))
            ORDER BY c.run_after, c.id
            LIMIT p_limit
            FOR UPDATE SKIP LOCKED)
        RETURNING j.*
    )
    SELECT * FROM claimed ORDER BY run_after, id;
END $$;

COMMENT ON FUNCTION mempher.lease_jobs(text, int, text[], interval, timestamptz) IS
    'Claims up to p_limit ready jobs for one worker, oldest first, skipping jobs other '
    'workers hold. NULL or empty kinds means any kind. Each claim counts as an attempt. Finish '
    'each with succeed_job or fail_job before p_at + p_lease, or reclaim_jobs takes it back.';

CREATE FUNCTION mempher.succeed_job(
    p_job    uuid,
    p_worker text,
    p_at     timestamptz DEFAULT now()
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    UPDATE mempher.jobs SET
        state        = 'done',
        leased_until = NULL,
        leased_by    = NULL,
        updated_at   = p_at
    WHERE id = p_job AND state = 'running' AND leased_by = p_worker
    RETURNING * INTO v_job;
    IF NOT FOUND THEN
        PERFORM mempher._explain_lost_lease(p_job, p_worker, 'succeed');
    END IF;
    RETURN v_job;
END $$;

COMMENT ON FUNCTION mempher.succeed_job(uuid, text, timestamptz) IS
    'Marks a job the worker holds done. Raises MP023 if the worker no longer holds it.';

CREATE FUNCTION mempher.fail_job(
    p_job      uuid,
    p_worker   text,
    p_error    text,
    p_at       timestamptz DEFAULT now(),
    p_retry_at timestamptz DEFAULT NULL
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    IF p_retry_at IS NOT NULL AND NOT isfinite(p_retry_at) THEN
        PERFORM mempher._fail('MP013', 'retry_at must be a finite timestamp');
    END IF;

    -- The attempt was counted at lease time, so a job that has used its last
    -- one goes dead rather than back to pending: the alternative is a job that
    -- retries for ever, which is how a broken extractor becomes a bill.
    --
    -- Backoff doubles from ten seconds to a ceiling of ten minutes, unless the
    -- caller knows better (a provider's Retry-After, say). The exponent is
    -- capped before it is used, so a job with a large max_attempts cannot
    -- overflow the interval.
    UPDATE mempher.jobs SET
        state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
        run_after    = CASE
                           WHEN attempts >= max_attempts THEN run_after
                           ELSE coalesce(p_retry_at,
                                p_at + least(interval '10 seconds' * power(2, least(greatest(attempts, 1) - 1, 10)),
                                             interval '10 minutes'))
                       END,
        leased_until = NULL,
        leased_by    = NULL,
        last_error   = CASE
                           WHEN length(p_error) > 2000 THEN left(p_error, 2000) || '... (truncated)'
                           ELSE p_error
                       END,
        updated_at   = p_at
    WHERE id = p_job AND state = 'running' AND leased_by = p_worker
    RETURNING * INTO v_job;
    IF NOT FOUND THEN
        PERFORM mempher._explain_lost_lease(p_job, p_worker, 'fail');
    END IF;
    RETURN v_job;
END $$;

COMMENT ON FUNCTION mempher.fail_job(uuid, text, text, timestamptz, timestamptz) IS
    'Records that the worker''s attempt at a job failed. The job returns to pending at '
    'p_retry_at, or after a backoff doubling from 10 seconds to 10 minutes when that is '
    'NULL, unless the attempt was its last, in which case it is dead. p_error is kept, '
    'truncated to 2000 characters.';

CREATE FUNCTION mempher.reclaim_jobs(p_at timestamptz DEFAULT now())
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_count int;
BEGIN
    PERFORM mempher._check_at(p_at);
    UPDATE mempher.jobs SET
        state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
        leased_until = NULL,
        leased_by    = NULL,
        last_error   = 'lease expired; the worker holding it is presumed dead',
        updated_at   = p_at
    WHERE state = 'running' AND leased_until < p_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.reclaim_jobs(timestamptz) IS
    'Releases every lease that expired before p_at and returns how many. A job with '
    'attempts left returns to pending; one without goes dead. Safe to call from every worker.';

-- ---------------------------------------------------------------------------
-- Operating the queue
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.retry_job(p_job uuid, p_at timestamptz DEFAULT now())
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job   mempher.jobs;
    v_state text;
BEGIN
    PERFORM mempher._check_at(p_at);
    -- The attempts are reset because they were spent on a cause that has since
    -- been dealt with; a revived job that could only try once more would die
    -- again on the first flake. last_error stays, so the row still says what
    -- went wrong.
    BEGIN
        UPDATE mempher.jobs SET
            state        = 'pending',
            attempts     = 0,
            run_after    = p_at,
            leased_until = NULL,
            leased_by    = NULL,
            updated_at   = p_at
        WHERE id = p_job AND state = 'dead'
        RETURNING * INTO v_job;
    EXCEPTION WHEN unique_violation THEN
        PERFORM mempher._fail('MP025', format(
            'retry job %s: the same work is already pending or running', p_job));
    END;
    IF FOUND THEN
        RETURN v_job;
    END IF;
    SELECT state INTO v_state FROM mempher.jobs WHERE id = p_job;
    IF NOT FOUND THEN
        PERFORM mempher._fail('MP020', format('retry job %s: no such job', p_job));
    END IF;
    PERFORM mempher._fail('MP024', format('retry job %s: it is %s, not dead', p_job, v_state));
    RETURN NULL;
END $$;

COMMENT ON FUNCTION mempher.retry_job(uuid, timestamptz) IS
    'Returns one dead job to pending with its attempts reset, once its cause is dealt '
    'with. Raises MP024 if it is not dead and MP025 if the same work is already queued.';

CREATE FUNCTION mempher.purge_jobs(
    p_before timestamptz,
    p_states text[] DEFAULT ARRAY['done', 'dead'],
    p_limit  int    DEFAULT 1000
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad   text;
    v_count int;
BEGIN
    PERFORM mempher._check_at(p_before, 'before');
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_states IS NULL OR cardinality(p_states) = 0 THEN
        PERFORM mempher._fail('MP009', 'states is empty; name done, dead or both');
    END IF;
    -- Deleting a job still owed would lose the work with no record it was lost.
    SELECT s INTO v_bad FROM unnest(p_states) AS s
    WHERE s IS NULL OR s NOT IN ('done', 'dead') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP009', format(
            'cannot purge %s jobs: only done and dead jobs are finished', coalesce(v_bad, 'null')));
    END IF;

    -- A bare DELETE cannot be bounded, and a purge that cannot be bounded holds
    -- locks for as long as the history is deep.
    DELETE FROM mempher.jobs
    WHERE id IN (
        SELECT id FROM mempher.jobs
        WHERE state = ANY(p_states) AND updated_at < p_before
        ORDER BY id
        LIMIT p_limit);
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.purge_jobs(timestamptz, text[], int) IS
    'Deletes up to p_limit finished (done or dead) jobs last updated before p_before, '
    'and returns how many went. Call it in a loop until it returns less than p_limit.';

CREATE FUNCTION mempher.job_stats(p_scope text DEFAULT NULL)
    RETURNS TABLE (kind text, state text, jobs bigint, oldest timestamptz)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    RETURN QUERY
    SELECT j.kind, j.state, count(*), min(j.created_at)
    FROM mempher.jobs j
    WHERE p_scope IS NULL OR j.scope_id = p_scope
    GROUP BY j.kind, j.state
    ORDER BY j.kind, j.state;
END $$;

COMMENT ON FUNCTION mempher.job_stats(text) IS
    'Counts the queue by kind and state, with the creation time of the oldest job in '
    'each, across every scope or within one.';
