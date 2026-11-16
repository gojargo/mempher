// The vector projection of L1: finding facts with no vector, and writing them.

package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/gojargo/mempher"
)

// DefaultPendingFactEncodingsLimit is how many facts
// [Store.PendingFactEncodings] returns when the query sets no limit.
const DefaultPendingFactEncodingsLimit = 100

// PendingFactEncodings returns facts with no encoding for a model, in id order.
func (s *Store) PendingFactEncodings(
	ctx context.Context,
	q mempher.PendingFactEncodings,
) ([]mempher.PendingFact, error) {
	if q.Limit < 0 {
		return nil, fmt.Errorf("mempher/postgres: pending fact encodings: Limit is %d: %w",
			q.Limit, mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultPendingFactEncodingsLimit
	}
	var scope *string
	if q.Scope != "" {
		name := string(q.Scope)
		scope = &name
	}
	var after *uuid.UUID
	if !q.After.IsZero() {
		id := uuid.UUID(q.After)
		after = &id
	}

	rows, err := s.pool.Query(ctx,
		`SELECT id, scope_id, statement FROM mempher.pending_fact_encodings($1, $2, $3, $4, $5)`,
		string(q.Model), scope, episodeUUIDs(q.Episodes), after, int32(limit))
	if err != nil {
		return nil, apiError("pending fact encodings", err)
	}
	defer rows.Close()

	out := make([]mempher.PendingFact, 0, min(limit, 128))
	for rows.Next() {
		var (
			id   uuid.UUID
			fact mempher.PendingFact
		)
		if err := rows.Scan(&id, &fact.Scope, &fact.Statement); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending fact encodings: scan: %w", err)
		}
		fact.ID = mempher.FactID(id)
		out = append(out, fact)
	}
	if err := rows.Err(); err != nil {
		return nil, apiError("pending fact encodings", err)
	}
	return out, nil
}

// PutFactEncodings writes one vector per fact for a model, in one statement,
// and reports how many were written. See [mempher.FactStore].
func (s *Store) PutFactEncodings(
	ctx context.Context,
	model mempher.ModelID,
	encodings []mempher.FactEncoding,
	at time.Time,
) (int, error) {
	if at.IsZero() {
		return 0, fmt.Errorf(
			"mempher/postgres: put fact encodings: at must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	facts := make([]uuid.UUID, len(encodings))
	statements := make([]string, len(encodings))
	vectors := make([]string, len(encodings))
	for i, enc := range encodings {
		if err := s.validateVector(enc.Vector, model); err != nil {
			return 0, fmt.Errorf("mempher/postgres: put fact encodings: fact %s: %w", enc.Fact, err)
		}
		facts[i] = uuid.UUID(enc.Fact)
		statements[i] = enc.Statement
		vectors[i] = formatVector(enc.Vector)
	}

	var written int
	err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT mempher.put_fact_encodings($1, $2, $3, $4::%s[], $5)`, s.vectorType),
		string(model), facts, statements, vectors, at).Scan(&written)
	if err != nil {
		return 0, apiError(fmt.Sprintf("put %d fact encodings", len(encodings)), err)
	}
	return written, nil
}

// EnqueueFactEncodings queues an encode-facts job for every fact with no vector
// for a model, in one scope or, when scope is empty, in all of them, and
// reports how many jobs it added. See [ops.Ops.BackfillFactEncodings].
func (s *Store) EnqueueFactEncodings(
	ctx context.Context,
	model mempher.ModelID,
	scope mempher.ScopeID,
	now, runAfter time.Time,
) (int, error) {
	var only *string
	if scope != "" {
		name := string(scope)
		only = &name
	}
	var jobs int
	err := s.pool.QueryRow(ctx,
		`SELECT mempher.enqueue_fact_encodings($1, $2, $3, $4)`,
		string(model), only, now, optionalTime(runAfter)).Scan(&jobs)
	if err != nil {
		return 0, apiError("enqueue fact encodings", err)
	}
	return jobs, nil
}
