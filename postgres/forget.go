// Erasure: the one way content leaves L0.

package postgres

import (
	"context"
	"fmt"

	"github.com/google/uuid"

	"github.com/gojargo/mempher"
)

// Forget erases episodes and every row this store holds that derives from them.
//
// The whole erasure is one transaction, because a half-erased scope is not a
// smaller erasure but a failed one. It declares itself to the append-only
// trigger, removes each projection before its source, and takes the scope row
// last when the whole scope is going.
//
// An episode named from the wrong scope is not erased and is not an error, for
// the same reason reading one is [mempher.ErrNotFound] rather than a permission
// failure: a Store never crosses a scope. What actually went is in the result.
func (s *Store) Forget(
	ctx context.Context,
	req mempher.ForgetRequest,
) (mempher.ForgetResult, error) {
	if err := req.Validate(); err != nil {
		return mempher.ForgetResult{}, fmt.Errorf("mempher/postgres: forget: %w", err)
	}
	var episodes []uuid.UUID
	if len(req.Episodes) > 0 {
		episodes = episodeUUIDs(req.Episodes)
	}
	out := mempher.ForgetResult{Scope: req.Scope}
	var counts [5]int32
	err := s.pool.QueryRow(ctx,
		`SELECT episodes, encodings, extractions, facts, jobs, scope_removed FROM mempher.forget($1, $2)`,
		string(req.Scope), episodes,
	).Scan(&counts[0], &counts[1], &counts[2], &counts[3], &counts[4], &out.ScopeRemoved)
	if err != nil {
		return mempher.ForgetResult{}, apiError(fmt.Sprintf("forget in scope %q", req.Scope), err)
	}
	out.Episodes, out.Encodings, out.Extractions = int(counts[0]), int(counts[1]), int(counts[2])
	out.Facts, out.Jobs = int(counts[3]), int(counts[4])
	return out, nil
}
