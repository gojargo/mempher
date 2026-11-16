// L0: appending episodes, reading them back, and replaying them in order.

package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/gojargo/mempher"
)

// episodeColumns is the read shape of an episode, in the order scanEpisode
// expects.
const episodeColumns = `id, scope_id, seq, content, role, actor, source,
	occurred_at, ingested_at, binding`
// appendSQL is the whole of the write path: mempher.append validates the batch,
// allocates its sequence numbers, inserts it and queues the jobs it implies, in
// one statement. See 0005_api_append.sql.
const appendSQL = `SELECT id, seq FROM mempher.append($1, $2::jsonb, $3, $4, $5)`

// appendEpisode is one element of mempher.append's episodes argument.
type appendEpisode struct {
	Content    string          `json:"content"`
	Role       mempher.Role    `json:"role"`
	Actor      mempher.Actor   `json:"actor"`
	Source     mempher.Source  `json:"source"`
	OccurredAt time.Time       `json:"occurred_at"`
	Binding    mempher.Binding `json:"binding"`
}

// Append inserts a batch of episodes and the jobs they imply, atomically,
// creating the scope if this is its first episode.
//
// A batch of one is the ordinary case and takes the same path, so the single and
// the batched append cannot drift apart.
func (s *Store) Append(ctx context.Context, cmd mempher.AppendCommand) (mempher.AppendResult, error) {
	if err := validateAppend(cmd); err != nil {
		return mempher.AppendResult{}, fmt.Errorf("mempher/postgres: append: %w", err)
	}

	first := cmd.Episodes[0]
	input := make([]appendEpisode, len(cmd.Episodes))
	for i, ep := range cmd.Episodes {
		binding := ep.Binding
		if binding == nil {
			binding = mempher.Binding{}
		}
		input[i] = appendEpisode{
			Content: ep.Content,
			Role:    ep.Role,
			Actor:   ep.Actor,
			Source:  ep.Source,
			// Truncated as pgx truncates a timestamptz parameter, so an
			// instant reaches the row the same way whichever path it took.
			OccurredAt: ep.OccurredAt.Truncate(time.Microsecond),
			Binding:    binding,
		}
	}
	payload, err := json.Marshal(input)
	if err != nil {
		return mempher.AppendResult{}, fmt.Errorf("mempher/postgres: append: encode episodes: %w", err)
	}
	kinds := make([]string, len(cmd.Jobs))
	maxAttempts := mempher.DefaultMaxAttempts
	for i, job := range cmd.Jobs {
		kinds[i] = string(job.Kind)
		if job.MaxAttempts > 0 {
			maxAttempts = job.MaxAttempts
		}
	}

	rows, err := s.pool.Query(ctx, appendSQL,
		string(first.Scope), payload, first.IngestedAt, kinds, int32(maxAttempts))
	if err != nil {
		return mempher.AppendResult{}, apiError(appendOp(first.Scope), err)
	}
	defer rows.Close()

	result := mempher.AppendResult{Episodes: make([]mempher.Episode, 0, len(cmd.Episodes))}
	for rows.Next() {
		var (
			id  uuid.UUID
			seq int64
		)
		if err := rows.Scan(&id, &seq); err != nil {
			return mempher.AppendResult{}, fmt.Errorf("mempher/postgres: append: scan: %w", err)
		}
		if len(result.Episodes) == len(cmd.Episodes) {
			return mempher.AppendResult{}, fmt.Errorf(
				"mempher/postgres: append: mempher.append returned more than the %d episodes given",
				len(cmd.Episodes))
		}
		ep := cmd.Episodes[len(result.Episodes)]
		result.Episodes = append(result.Episodes, mempher.Episode{
			ID:         mempher.EpisodeID(id),
			Scope:      ep.Scope,
			Seq:        seq,
			Content:    ep.Content,
			Role:       ep.Role,
			Actor:      ep.Actor,
			Source:     ep.Source,
			OccurredAt: ep.OccurredAt.Truncate(time.Microsecond).UTC(),
			IngestedAt: ep.IngestedAt.Truncate(time.Microsecond).UTC(),
			Binding:    ep.Binding.Clone(),
		})
	}
	if err := rows.Err(); err != nil {
		return mempher.AppendResult{}, apiError(appendOp(first.Scope), err)
	}
	if len(result.Episodes) != len(cmd.Episodes) {
		return mempher.AppendResult{}, fmt.Errorf(
			"mempher/postgres: append: mempher.append returned %d of %d episodes, "+
				"which should be impossible", len(result.Episodes), len(cmd.Episodes))
	}
	return result, nil
}

func appendOp(scope mempher.ScopeID) string {
	return fmt.Sprintf("append to scope %q", scope)
}

// Episode returns one episode by id within a scope.
//
// The scope is part of the predicate rather than checked afterwards, so an
// episode addressed from the wrong scope is indistinguishable from one that does
// not exist.
func (s *Store) Episode(
	ctx context.Context,
	scope mempher.ScopeID,
	id mempher.EpisodeID,
) (mempher.Episode, error) {
	if err := scope.Validate(); err != nil {
		return mempher.Episode{}, fmt.Errorf("mempher/postgres: episode: %w", err)
	}
	if id.IsZero() {
		return mempher.Episode{}, fmt.Errorf(
			"mempher/postgres: episode: id is unset: %w", mempher.ErrNotFound)
	}

	row := s.pool.QueryRow(ctx,
		`SELECT `+episodeColumns+` FROM mempher.episodes WHERE scope_id = $1 AND id = $2`,
		string(scope), uuid.UUID(id))

	ep, err := scanEpisode(row)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return mempher.Episode{}, fmt.Errorf(
			"mempher/postgres: episode %s in scope %q: %w", id, scope, mempher.ErrNotFound)
	case err != nil:
		return mempher.Episode{}, fmt.Errorf("mempher/postgres: episode %s: %w", id, err)
	}
	return ep, nil
}

// Replay returns up to limit episodes of a scope in Seq order, starting after
// afterSeq. The ordering is the point: consolidation has to see history as it
// happened.
func (s *Store) Replay(
	ctx context.Context,
	scope mempher.ScopeID,
	afterSeq int64,
	limit int,
) ([]mempher.Episode, error) {
	if err := scope.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay: %w", err)
	}
	if afterSeq < 0 {
		return nil, fmt.Errorf("mempher/postgres: replay: afterSeq is %d: %w",
			afterSeq, mempher.ErrInvalidConfig)
	}
	if limit <= 0 {
		return nil, fmt.Errorf("mempher/postgres: replay: limit is %d, must be positive: %w",
			limit, mempher.ErrInvalidConfig)
	}

	rows, err := s.pool.Query(ctx,
		`SELECT `+episodeColumns+` FROM mempher.episodes
		 WHERE scope_id = $1 AND seq > $2 ORDER BY seq LIMIT $3`,
		string(scope), afterSeq, limit)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
	}
	defer rows.Close()

	out := make([]mempher.Episode, 0, min(limit, 64))
	for rows.Next() {
		ep, err := scanEpisode(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
		}
		out = append(out, ep)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
	}
	return out, nil
}

// scanner is what both a single row and a row of a result set satisfy.
type scanner interface {
	Scan(dest ...any) error
}

// scanEpisode reads one episode in [episodeColumns] order.
//
// Timestamps come back as UTC. pgx returns timestamptz at the session's offset,
// which is the same instant but a different rendering, and normalising here makes
// round-trips and test comparisons predictable.
func scanEpisode(row scanner) (mempher.Episode, error) {
	var (
		ep      mempher.Episode
		id      uuid.UUID
		role    string
		binding mempher.Binding
	)
	if err := row.Scan(&id, &ep.Scope, &ep.Seq, &ep.Content, &role, &ep.Actor, &ep.Source,
		&ep.OccurredAt, &ep.IngestedAt, &binding); err != nil {
		// Wrapped, not replaced: callers test for pgx.ErrNoRows through it.
		return mempher.Episode{}, fmt.Errorf("scan episode: %w", err)
	}

	ep.ID = mempher.EpisodeID(id)
	ep.Role = mempher.Role(role)
	if !ep.Role.Valid() {
		return mempher.Episode{}, fmt.Errorf("episode %s has role %q: %w",
			ep.ID, role, mempher.ErrInvalidRole)
	}
	ep.OccurredAt = ep.OccurredAt.UTC()
	ep.IngestedAt = ep.IngestedAt.UTC()
	if len(binding) > 0 {
		ep.Binding = binding
	}
	return ep, nil
}

// validateAppend rejects a command the schema would reject anyway, so the error
// names the field instead of quoting a constraint.
func validateAppend(cmd mempher.AppendCommand) error {
	if len(cmd.Episodes) == 0 {
		return fmt.Errorf("a command must name at least one episode: %w",
			mempher.ErrInvalidConfig)
	}
	scope := cmd.Episodes[0].Scope
	at := cmd.Episodes[0].IngestedAt
	for i, ep := range cmd.Episodes {
		// The caller-facing rules live in one place; borrow them.
		probe := mempher.AppendRequest{
			Scope:   ep.Scope,
			Content: ep.Content,
			Role:    ep.Role,
			Actor:   ep.Actor,
			Source:  ep.Source,
			Binding: ep.Binding,
		}
		if err := probe.Validate(); err != nil {
			return fmt.Errorf("episode %d: %w", i, err)
		}
		if ep.OccurredAt.IsZero() || ep.IngestedAt.IsZero() {
			return fmt.Errorf(
				"episode %d: OccurredAt and IngestedAt must be resolved by the caller, "+
					"because a Store never reads a clock: %w", i, mempher.ErrInvalidConfig)
		}
		// One command is one instant of system time: it is the ingest
		// time of every episode and the creation time of every job.
		if !ep.IngestedAt.Equal(at) {
			return fmt.Errorf(
				"episode %d was ingested at %s, not %s: one command is one ingest instant: %w",
				i, ep.IngestedAt, at, mempher.ErrInvalidConfig)
		}
		// Sequencing takes one scope's row, so a command is one scope.
		if ep.Scope != scope {
			return fmt.Errorf(
				"episode %d names scope %q, not %q: one command is one scope: %w",
				i, ep.Scope, scope, mempher.ErrInvalidConfig)
		}
	}

	kinds := make(map[mempher.JobKind]struct{}, len(cmd.Jobs))
	attempts := 0
	for i, job := range cmd.Jobs {
		switch {
		case !job.Kind.Valid():
			return fmt.Errorf("job %d kind %q: %w",
				i, job.Kind, mempher.ErrInvalidJobKind)
		case len(job.Episodes) > 0:
			return fmt.Errorf(
				"job %d already names episodes, but the Store fills "+
					"that in with the id it mints: %w", i, mempher.ErrInvalidConfig)
		case job.MaxAttempts < 0:
			return fmt.Errorf("job %d MaxAttempts is %d: %w",
				i, job.MaxAttempts, mempher.ErrInvalidConfig)
		case !job.RunAfter.IsZero() && !job.RunAfter.Equal(at):
			return fmt.Errorf(
				"job %d runs after %s, but the jobs an append implies are ready "+
					"the moment it commits: %w", i, job.RunAfter, mempher.ErrInvalidConfig)
		case job.MaxAttempts > 0 && attempts > 0 && job.MaxAttempts != attempts:
			return fmt.Errorf(
				"job %d allows %d attempts and an earlier one %d, but the jobs of "+
					"one append share a retry budget: %w",
				i, job.MaxAttempts, attempts, mempher.ErrInvalidConfig)
		}
		if job.MaxAttempts > 0 {
			attempts = job.MaxAttempts
		}
		if _, dup := kinds[job.Kind]; dup {
			return fmt.Errorf(
				"two %q jobs for one episode would collapse into one: %w",
				job.Kind, mempher.ErrInvalidConfig)
		}
		kinds[job.Kind] = struct{}{}
	}
	return nil
}

// deref reads a nullable scan target, treating NULL as the zero value.
func deref[T any](p *T) T {
	if p == nil {
		var zero T
		return zero
	}
	return *p
}
