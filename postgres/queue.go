// The deferred-work port: claiming jobs, finishing them, and recovering the ones
// a dead worker was holding.
//
// The queue's state machine is the SQL API's (0001_init.sql): every method
// that moves a job is one call to a mempher.* function, and only listing and
// reading single jobs are plain queries against mempher.jobs.

package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/ops"
)

// Defaults for the operational queries, used when the corresponding field is
// zero.
const (
	// DefaultJobsLimit is how many jobs [Store.Jobs] returns when a query
	// does not say.
	DefaultJobsLimit = 100
	// DefaultPurgeLimit is how many rows one [Store.Purge] call deletes when
	// a request does not say. It is deliberately modest: a purge is a
	// maintenance loop of short transactions, not one long one.
	DefaultPurgeLimit = 1000
)

// jobColumns is the read shape of a job, in the order scanJob expects. Every
// queue function returns rows of mempher.jobs, so it reads those too.
const jobColumns = `id, kind, scope_id, episode_ids, state, attempts, max_attempts,
	run_after, leased_until, leased_by, last_error, created_at, updated_at`

// Enqueue adds one job, or returns the outstanding one it would have
// duplicated.
func (s *Store) Enqueue(
	ctx context.Context,
	job mempher.NewJob,
	now time.Time,
) (mempher.Job, error) {
	if now.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: enqueue: now must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	for i, id := range job.Episodes {
		if id.IsZero() {
			return mempher.Job{}, fmt.Errorf("mempher/postgres: enqueue: episode %d is unset: %w",
				i, mempher.ErrInvalidConfig)
		}
	}
	var runAfter *time.Time
	if !job.RunAfter.IsZero() {
		runAfter = &job.RunAfter
	}
	maxAttempts := job.MaxAttempts
	if maxAttempts == 0 {
		maxAttempts = mempher.DefaultMaxAttempts
	}
	enqueued, err := scanJob(s.pool.QueryRow(ctx,
		`SELECT `+jobColumns+` FROM mempher.enqueue($1, $2, $3, $4, $5, $6)`,
		string(job.Kind), string(job.Scope), episodeUUIDs(job.Episodes),
		now, runAfter, int32(maxAttempts)))
	if err != nil {
		return mempher.Job{}, apiError(fmt.Sprintf("enqueue %s job", job.Kind), err)
	}
	return enqueued, nil
}

// Lease atomically claims ready jobs for one worker, oldest first.
func (s *Store) Lease(ctx context.Context, req mempher.LeaseRequest) ([]mempher.Job, error) {
	if req.Now.IsZero() {
		return nil, fmt.Errorf("mempher/postgres: lease: the Now field must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	limit := req.Limit
	if limit == 0 {
		limit = 1
	}
	duration := req.Duration
	if duration == 0 {
		duration = mempher.DefaultLeaseDuration
	}
	var kinds []string
	if len(req.Kinds) > 0 {
		kinds = make([]string, len(req.Kinds))
		for i, kind := range req.Kinds {
			kinds[i] = string(kind)
		}
	}
	op := fmt.Sprintf("lease for %q", req.Worker)
	rows, err := s.pool.Query(ctx,
		`SELECT `+jobColumns+` FROM mempher.lease_jobs($1, $2, $3, $4, $5)`,
		string(req.Worker), int32(limit), kinds, duration, req.Now)
	if err != nil {
		return nil, apiError(op, err)
	}
	defer rows.Close()
	out := make([]mempher.Job, 0, max(limit, 0))
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: %s: %w", op, err)
		}
		out = append(out, job)
	}
	if err := rows.Err(); err != nil {
		return nil, apiError(op, err)
	}
	return out, nil
}

// Succeed marks a leased job done, or returns [mempher.ErrJobNotLeased] if the
// worker no longer holds it.
func (s *Store) Succeed(
	ctx context.Context,
	id mempher.JobID,
	worker mempher.WorkerID,
	at time.Time,
) error {
	if at.IsZero() {
		return fmt.Errorf("mempher/postgres: succeed: the time must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	if err := s.exec(ctx, `SELECT 1 FROM mempher.succeed_job($1, $2, $3)`,
		uuid.UUID(id), string(worker), at); err != nil {
		return apiError(fmt.Sprintf("succeed job %s", id), err)
	}
	return nil
}

// Fail records that an attempt failed. The job returns to pending at retryAt, or
// after the queue's own backoff when retryAt is zero, unless the attempt was its
// last.
func (s *Store) Fail(
	ctx context.Context,
	id mempher.JobID,
	worker mempher.WorkerID,
	cause error,
	at, retryAt time.Time,
) error {
	if at.IsZero() {
		return fmt.Errorf("mempher/postgres: fail: the time must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	var message *string
	if cause != nil {
		text := cause.Error()
		message = &text
	}
	var retry *time.Time
	if !retryAt.IsZero() {
		retry = &retryAt
	}
	if err := s.exec(ctx, `SELECT 1 FROM mempher.fail_job($1, $2, $3, $4, $5)`,
		uuid.UUID(id), string(worker), message, at, retry); err != nil {
		return apiError(fmt.Sprintf("fail job %s", id), err)
	}
	return nil
}

// Reclaim releases jobs whose leases expired before now, and reports how many.
func (s *Store) Reclaim(ctx context.Context, now time.Time) (int, error) {
	if now.IsZero() {
		return 0, fmt.Errorf("mempher/postgres: reclaim: now is unset: %w",
			mempher.ErrInvalidConfig)
	}
	var released int32
	if err := s.pool.QueryRow(ctx, `SELECT mempher.reclaim_jobs($1)`, now).Scan(&released); err != nil {
		return 0, apiError("reclaim", err)
	}
	return int(released), nil
}

// Job returns one job by id.
func (s *Store) Job(ctx context.Context, id mempher.JobID) (mempher.Job, error) {
	if id.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job: id is unset: %w",
			mempher.ErrNotFound)
	}
	job, err := scanJob(s.pool.QueryRow(ctx,
		`SELECT `+jobColumns+` FROM mempher.jobs WHERE id = $1`, uuid.UUID(id)))
	if errors.Is(err, pgx.ErrNoRows) {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job %s: %w", id, mempher.ErrNotFound)
	}
	if err != nil {
		return mempher.Job{}, apiError(fmt.Sprintf("job %s", id), err)
	}
	return job, nil
}

// exec runs a statement whose rows do not matter, only whether it succeeded.
func (s *Store) exec(ctx context.Context, sql string, args ...any) error {
	_, err := s.pool.Exec(ctx, sql, args...)
	return err //nolint:wrapcheck // every caller wraps it through apiError
}

// Jobs lists jobs matching q, oldest first.
//
// Ordering is by id, which is a uuidv7 and therefore the order the jobs were
// created in. That makes the cursor a single column and keeps a listing stable
// while jobs around it change state.
func (s *Store) Jobs(ctx context.Context, q ops.JobQuery) ([]mempher.Job, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultJobsLimit
	}

	b := &queryBuilder{}
	var sql strings.Builder
	sql.WriteString(`SELECT ` + jobColumns + ` FROM mempher.jobs WHERE true`)
	if q.Scope != "" {
		fmt.Fprintf(&sql, "\n  AND scope_id = %s", b.arg(string(q.Scope)))
	}
	if len(q.Kinds) > 0 {
		kinds := make([]string, len(q.Kinds))
		for i, kind := range q.Kinds {
			kinds[i] = string(kind)
		}
		fmt.Fprintf(&sql, "\n  AND kind = ANY(%s)", b.arg(kinds))
	}
	if len(q.States) > 0 {
		states := make([]string, len(q.States))
		for i, state := range q.States {
			states[i] = string(state)
		}
		fmt.Fprintf(&sql, "\n  AND state = ANY(%s)", b.arg(states))
	}
	if !q.After.IsZero() {
		fmt.Fprintf(&sql, "\n  AND id > %s", b.arg(uuid.UUID(q.After)))
	}
	fmt.Fprintf(&sql, "\nORDER BY id\nLIMIT %s", b.arg(limit))

	rows, err := s.pool.Query(ctx, sql.String(), b.args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: jobs: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	defer rows.Close()

	out := make([]mempher.Job, 0, min(limit, 128))
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
		}
		out = append(out, job)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	return out, nil
}

// scanJob reads one job in [jobColumns] order.
func scanJob(row scanner) (mempher.Job, error) {
	var (
		job         mempher.Job
		id          uuid.UUID
		kind        string
		state       string
		episodeIDs  []uuid.UUID
		attempts    int32
		maxAttempts int32
		leasedUntil *time.Time
		leasedBy    *string
		lastError   *string
	)
	if err := row.Scan(&id, &kind, &job.Scope, &episodeIDs, &state, &attempts, &maxAttempts,
		&job.RunAfter, &leasedUntil, &leasedBy, &lastError,
		&job.CreatedAt, &job.UpdatedAt); err != nil {
		// Wrapped, not replaced: callers test for pgx.ErrNoRows through it.
		return mempher.Job{}, fmt.Errorf("scan job: %w", err)
	}

	job.ID = mempher.JobID(id)
	job.Kind = mempher.JobKind(kind)
	job.State = mempher.JobState(state)
	if !job.Kind.Valid() {
		return mempher.Job{}, fmt.Errorf("job %s has kind %q: %w",
			job.ID, kind, mempher.ErrInvalidJobKind)
	}
	if !job.State.Valid() {
		return mempher.Job{}, fmt.Errorf("job %s has state %q: %w",
			job.ID, state, mempher.ErrInvalidConfig)
	}

	job.Episodes = make([]mempher.EpisodeID, len(episodeIDs))
	for i, episodeID := range episodeIDs {
		job.Episodes[i] = mempher.EpisodeID(episodeID)
	}
	job.Attempts = int(attempts)
	job.MaxAttempts = int(maxAttempts)
	job.RunAfter = job.RunAfter.UTC()
	job.CreatedAt = job.CreatedAt.UTC()
	job.UpdatedAt = job.UpdatedAt.UTC()
	if leasedUntil != nil {
		job.LeasedUntil = leasedUntil.UTC()
	}
	job.LeasedBy = mempher.WorkerID(deref(leasedBy))
	job.LastError = deref(lastError)
	return job, nil
}

// episodeUUIDs converts episode ids for the driver.
func episodeUUIDs(ids []mempher.EpisodeID) []uuid.UUID {
	out := make([]uuid.UUID, len(ids))
	for i, id := range ids {
		out[i] = uuid.UUID(id)
	}
	return out
}

// Stats counts the queue by kind and state, and reports the age of the oldest
// job in each bucket.
//
// It is an aggregate over the whole table, so it costs what the table's history
// costs. That is the argument for [Store.Purge]: a queue nobody prunes makes
// this slower every day, for rows that only ever get counted.
func (s *Store) Stats(
	ctx context.Context,
	scope mempher.ScopeID,
) ([]ops.JobCount, error) {
	var scopeArg *string
	if scope != "" {
		name := string(scope)
		scopeArg = &name
	}
	rows, err := s.pool.Query(ctx,
		`SELECT kind, state, jobs, oldest FROM mempher.job_stats($1)`, scopeArg)
	if err != nil {
		return nil, apiError("job stats", err)
	}
	defer rows.Close()
	out := make([]ops.JobCount, 0, 8)
	for rows.Next() {
		var (
			count  ops.JobCount
			kind   string
			state  string
			jobs   int64
			oldest *time.Time
		)
		if err := rows.Scan(&kind, &state, &jobs, &oldest); err != nil {
			return nil, fmt.Errorf("mempher/postgres: job stats: scan bucket: %w", err)
		}
		count.Kind = mempher.JobKind(kind)
		count.State = mempher.JobState(state)
		count.Jobs = int(jobs)
		if oldest != nil {
			count.Oldest = oldest.UTC()
		}
		out = append(out, count)
	}
	if err := rows.Err(); err != nil {
		return nil, apiError("job stats", err)
	}
	return out, nil
}

// Retry returns one dead job to pending at time now, with its attempts reset.
// Reviving work the queue is already holding is refused with
// [mempher.ErrJobOutstanding] rather than merged.
func (s *Store) Retry(
	ctx context.Context,
	id mempher.JobID,
	now time.Time,
) (mempher.Job, error) {
	if id.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry: id is unset: %w",
			mempher.ErrNotFound)
	}
	if now.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry: now is unset: %w",
			mempher.ErrInvalidConfig)
	}
	job, err := scanJob(s.pool.QueryRow(ctx,
		`SELECT `+jobColumns+` FROM mempher.retry_job($1, $2)`, uuid.UUID(id), now))
	if err != nil {
		return mempher.Job{}, apiError(fmt.Sprintf("retry job %s", id), err)
	}
	return job, nil
}

// Purge deletes finished jobs matching req and reports how many rows went.
//
// It is the one DELETE the queue allows. It is permitted because the queue is a
// work list rather than a layer of memory: nothing is derived from a done job,
// and a dead one has already been read by whoever it was left for.
func (s *Store) Purge(ctx context.Context, req ops.PurgeRequest) (int, error) {
	if err := req.Validate(); err != nil {
		return 0, fmt.Errorf("mempher/postgres: purge: %w", err)
	}
	limit := req.Limit
	if limit == 0 {
		limit = DefaultPurgeLimit
	}
	states := req.States
	if len(states) == 0 {
		states = []mempher.JobState{mempher.JobStateDone, mempher.JobStateDead}
	}
	names := make([]string, len(states))
	for i, state := range states {
		names[i] = string(state)
	}
	var purged int32
	if err := s.pool.QueryRow(ctx, `SELECT mempher.purge_jobs($1, $2, $3)`,
		req.Before, names, int32(limit)).Scan(&purged); err != nil {
		return 0, apiError("purge", err)
	}
	return int(purged), nil
}
