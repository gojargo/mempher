// L1: applying an extraction, reading facts back, and finding episodes no
// extractor has read.

package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/ops"
)

// Defaults used when the corresponding query field is left zero.
const (
	// DefaultFactsLimit is how many facts [Store.Facts] returns when a query
	// does not say.
	DefaultFactsLimit = 200
	// DefaultPendingExtractionsLimit is how many ids
	// [Store.PendingExtractions] returns when a query does not say.
	DefaultPendingExtractionsLimit = 100
)

// pgerrcodeExclusionViolation is what a WITHOUT OVERLAPS key raises when two
// windows for one claim collide.
const pgerrcodeExclusionViolation = "23P01"

// factColumns is the read shape of a fact, in the order scanFact expects.
const factColumns = `id, scope_id, extractor, subject, predicate, object,
	statement, valid, confidence, episode_ids, asserted_at, updated_at`

// assertionJSON is one element of mempher.apply_extraction's assert argument.
type assertionJSON struct {
	Subject    mempher.Subject   `json:"subject"`
	Predicate  mempher.Predicate `json:"predicate"`
	Object     string            `json:"object"`
	Statement  string            `json:"statement"`
	ValidFrom  time.Time         `json:"valid_from"`
	ValidTo    *time.Time        `json:"valid_to,omitempty"`
	Confidence float32           `json:"confidence"`
}

// retractionJSON is one element of mempher.apply_extraction's retract argument.
type retractionJSON struct {
	Fact uuid.UUID `json:"fact"`
	At   time.Time `json:"at"`
}

// ApplyExtraction records one extraction atomically: the assertions, the
// retractions and the markers saying which episodes were read. It is one call to
// mempher.apply_extraction, which also decides what a repeated extraction does
// (nothing) and when two claims conflict.
func (s *Store) ApplyExtraction(
	ctx context.Context,
	cmd mempher.ExtractCommand,
) ([]mempher.Fact, error) {
	if err := cmd.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: %w", err)
	}
	assertions := make([]assertionJSON, len(cmd.Assert))
	for i, a := range cmd.Assert {
		assertions[i] = assertionJSON{
			Subject:    a.Subject,
			Predicate:  a.Predicate,
			Object:     a.Object,
			Statement:  a.Statement,
			ValidFrom:  a.Valid.From.Truncate(time.Microsecond),
			Confidence: a.Confidence,
		}
		if !a.Valid.IsOpen() {
			to := a.Valid.To.Truncate(time.Microsecond)
			assertions[i].ValidTo = &to
		}
	}
	retractions := make([]retractionJSON, len(cmd.Retract))
	for i, r := range cmd.Retract {
		retractions[i] = retractionJSON{Fact: uuid.UUID(r.Fact), At: r.At.Truncate(time.Microsecond)}
	}
	assertPayload, err := json.Marshal(assertions)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: encode assertions: %w", err)
	}
	retractPayload, err := json.Marshal(retractions)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: encode retractions: %w", err)
	}
	facts, err := scanFacts(ctx, s.pool,
		`SELECT `+factColumns+` FROM mempher.apply_extraction($1, $2, $3, $4::jsonb, $5::jsonb, $6)`,
		string(cmd.Scope), string(cmd.Extractor), episodeUUIDs(cmd.Episodes),
		assertPayload, retractPayload, cmd.At)
	if err != nil {
		return nil, s.wrapFactError("apply extraction", cmd.Scope, err)
	}
	return facts, nil
}

// Facts returns the facts matching q, through mempher.search_facts.
func (s *Store) Facts(ctx context.Context, q mempher.FactQuery) ([]mempher.Fact, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: facts: %w", err)
	}
	if q.AsOf.IsZero() {
		return nil, fmt.Errorf(
			"mempher/postgres: facts: AsOf must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	if q.At.IsZero() {
		return nil, fmt.Errorf(
			"mempher/postgres: facts: At must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultFactsLimit
	}
	var text *string
	if strings.TrimSpace(q.Text) != "" {
		text = &q.Text
	}
	subjects := make([]string, len(q.Subjects))
	for i, subject := range q.Subjects {
		subjects[i] = string(subject)
	}
	predicates := make([]string, len(q.Predicates))
	for i, predicate := range q.Predicates {
		predicates[i] = string(predicate)
	}
	facts, err := scanFacts(ctx, s.pool,
		`SELECT `+factColumns+` FROM mempher.search_facts($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
		string(q.Scope), string(q.Extractor), q.At, q.AsOf, text, int32(limit),
		subjects, predicates, q.MinConfidence, q.IncludeClosed)
	if err != nil {
		return nil, s.wrapFactError("facts", q.Scope, err)
	}
	return facts, nil
}

// factByIDSQL reads one fact within a scope. The scope is part of the predicate,
// not checked afterwards, so a fact addressed from the wrong scope is absent
// rather than refused.
const factByIDSQL = `SELECT ` + factColumns + `
FROM mempher.facts WHERE id = $1 AND scope_id = $2`

// Fact returns one fact by id within a scope, or [mempher.ErrNotFound].
func (s *Store) Fact(
	ctx context.Context,
	scope mempher.ScopeID,
	id mempher.FactID,
) (mempher.Fact, error) {
	if err := scope.Validate(); err != nil {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact: %w", err)
	}
	if id.IsZero() {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact: id is unset: %w",
			mempher.ErrInvalidConfig)
	}

	facts, err := scanFacts(ctx, s.pool, factByIDSQL, uuid.UUID(id), string(scope))
	if err != nil {
		return mempher.Fact{}, s.wrapFactError("fact", scope, err)
	}
	if len(facts) == 0 {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact %s in scope %q: %w",
			id, scope, mempher.ErrNotFound)
	}
	return facts[0], nil
}

// PendingExtractions returns ids of episodes no extraction marker covers, in Seq
// order.
func (s *Store) PendingExtractions(
	ctx context.Context,
	q ops.PendingExtractions,
) ([]mempher.EpisodeID, error) {
	if q.Limit < 0 {
		return nil, fmt.Errorf("mempher/postgres: pending extractions: Limit is %d: %w",
			q.Limit, mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultPendingExtractionsLimit
	}
	var scope *string
	if q.Scope != "" {
		name := string(q.Scope)
		scope = &name
	}
	rows, err := s.pool.Query(ctx,
		`SELECT id FROM mempher.pending_episodes('extract', $1, $2, $3, $4)`,
		string(q.Extractor), scope, q.AfterSeq, int32(limit))
	if err != nil {
		return nil, apiError("pending extractions", err)
	}
	defer rows.Close()
	out := make([]mempher.EpisodeID, 0, min(limit, 128))
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending extractions: scan: %w", err)
		}
		out = append(out, mempher.EpisodeID(id))
	}
	if err := rows.Err(); err != nil {
		return nil, apiError("pending extractions", err)
	}
	return out, nil
}

// querier is the part of pgx a fact read needs, so the same scan serves the pool
// and a transaction.
type querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// scanFacts reads a fact result set.
func scanFacts(ctx context.Context, q querier, sql string, args ...any) ([]mempher.Fact, error) {
	rows, err := q.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]mempher.Fact, 0, 16)
	for rows.Next() {
		fact, err := scanFact(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, fact)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read facts: %w", err)
	}
	return out, nil
}

// scanFact reads one row in the order of [factColumns].
func scanFact(rows pgx.Rows) (mempher.Fact, error) {
	var (
		f         mempher.Fact
		id        [16]byte
		subject   string
		predicate string
		valid     pgtype.Range[pgtype.Timestamptz]
		episodes  []uuid.UUID
	)
	if err := rows.Scan(&id, &f.Scope, &f.Extractor, &subject, &predicate, &f.Object,
		&f.Statement, &valid, &f.Confidence, &episodes, &f.AssertedAt, &f.UpdatedAt); err != nil {
		return mempher.Fact{}, fmt.Errorf("scan fact: %w", err)
	}

	f.ID = mempher.FactID(id)
	f.Subject = mempher.Subject(subject)
	f.Predicate = mempher.Predicate(predicate)
	f.AssertedAt = f.AssertedAt.UTC()
	f.UpdatedAt = f.UpdatedAt.UTC()

	validity, err := toValidity(valid)
	if err != nil {
		return mempher.Fact{}, fmt.Errorf("fact %s: %w", f.ID, err)
	}
	f.Valid = validity

	if len(episodes) > 0 {
		f.Episodes = make([]mempher.EpisodeID, len(episodes))
		for i, e := range episodes {
			f.Episodes[i] = mempher.EpisodeID(e)
		}
	}
	return f, nil
}

// toValidity converts the stored range to the Go window.
//
// The schema constrains the range to be non-empty, lower-bounded and half-open,
// so anything else here means a row was written by something other than this
// library, and reading it as a plausible window would hide that.
func toValidity(r pgtype.Range[pgtype.Timestamptz]) (mempher.Validity, error) {
	if r.LowerType != pgtype.Inclusive || !r.Lower.Valid {
		return mempher.Validity{}, fmt.Errorf(
			"validity window is not lower-bounded and inclusive: %w", mempher.ErrInvalidValidity)
	}
	out := mempher.Validity{From: r.Lower.Time.UTC()}
	switch r.UpperType {
	case pgtype.Unbounded:
		// Still true, which the Go type spells as a zero To.
	case pgtype.Exclusive:
		if !r.Upper.Valid {
			return mempher.Validity{}, fmt.Errorf(
				"validity window has an exclusive but absent upper bound: %w",
				mempher.ErrInvalidValidity)
		}
		out.To = r.Upper.Time.UTC()
	default:
		return mempher.Validity{}, fmt.Errorf(
			"validity window is not half-open: %w", mempher.ErrInvalidValidity)
	}
	return out, nil
}

// wrapFactError turns the conditions L1 has sentinels for into them, and names
// the scope on everything else.
func (s *Store) wrapFactError(op string, scope mempher.ScopeID, err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		if _, ok := apiErrors[pgErr.Code]; ok {
			return apiError(fmt.Sprintf("%s in scope %q", op, scope), err)
		}
	}
	switch {
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return fmt.Errorf("mempher/postgres: %s: run Migrate first: %w", op, ErrSchemaNotReady)
	case isCode(err, pgerrcodeExclusionViolation):
		return fmt.Errorf(
			"mempher/postgres: %s in scope %q: a claim is already believed over an "+
				"overlapping window; retract it before asserting a new one: %w",
			op, scope, mempher.ErrFactConflict)
	case isCode(err, pgerrcodeCheckViolation):
		return fmt.Errorf("mempher/postgres: %s in scope %q: %w: %w",
			op, scope, mempher.ErrInvalidFact, err)
	case isCode(err, pgerrcodeForeignKeyViolat):
		return fmt.Errorf("mempher/postgres: %s in scope %q: %w",
			op, scope, mempher.ErrNotFound)
	}
	return fmt.Errorf("mempher/postgres: %s in scope %q: %w", op, scope, err)
}

// The adapter serves every port in the library, so a signature drifting from an
// interface is a compile error here rather than a wiring error in New.
var (
	_ mempher.Store     = (*Store)(nil)
	_ mempher.Queue     = (*Store)(nil)
	_ mempher.FactStore = (*Store)(nil)
)
