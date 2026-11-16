// L1: entities, the names they go by, and the facts around them.

package postgres

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/gojargo/mempher"
)

// Defaults used when the corresponding query field is left zero.
const (
	// DefaultEntitiesLimit is how many entities [Store.Entities] returns when
	// a query does not say.
	DefaultEntitiesLimit = 200
	// DefaultRelatedFactsLimit is how many facts [Store.RelatedFacts] returns
	// when a query does not say.
	DefaultRelatedFactsLimit = 200
)

// entityColumns is the read shape of an entity, as mempher.list_entities
// returns it.
const entityColumns = `id, name, aliases, facts, created_at, updated_at`

// Entities lists the entities matching q, through mempher.list_entities.
func (s *Store) Entities(ctx context.Context, q mempher.EntityQuery) ([]mempher.Entity, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: entities: %w", err)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultEntitiesLimit
	}
	var text *string
	if strings.TrimSpace(q.MentionedIn) != "" {
		text = &q.MentionedIn
	}
	rows, err := s.pool.Query(ctx,
		`SELECT `+entityColumns+` FROM mempher.list_entities($1, $2, $3, $4, $5)`,
		string(q.Scope), string(q.Extractor), text, q.AsOf, int32(limit))
	if err != nil {
		return nil, s.wrapFactError("entities", q.Scope, err)
	}
	defer rows.Close()

	out := make([]mempher.Entity, 0, 16)
	for rows.Next() {
		var (
			e       = mempher.Entity{Scope: q.Scope, Extractor: q.Extractor}
			id      [16]byte
			name    string
			aliases []string
			facts   int64
		)
		if err := rows.Scan(&id, &name, &aliases, &facts, &e.CreatedAt, &e.UpdatedAt); err != nil {
			return nil, fmt.Errorf("mempher/postgres: entities: scan: %w", err)
		}
		e.ID, e.Name, e.Facts = mempher.EntityID(id), mempher.Subject(name), int(facts)
		e.CreatedAt, e.UpdatedAt = e.CreatedAt.UTC(), e.UpdatedAt.UTC()
		if len(aliases) > 0 {
			e.Aliases = make([]mempher.Subject, len(aliases))
			for i, alias := range aliases {
				e.Aliases[i] = mempher.Subject(alias)
			}
		}
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return nil, s.wrapFactError("entities", q.Scope, err)
	}
	return out, nil
}

// MergeEntities declares two names to be one entity, through
// mempher.merge_entities, and returns that entity as it now stands.
func (s *Store) MergeEntities(ctx context.Context, cmd mempher.MergeCommand) (mempher.Entity, error) {
	if err := cmd.Validate(); err != nil {
		return mempher.Entity{}, fmt.Errorf("mempher/postgres: merge entities: %w", err)
	}
	var id uuid.UUID
	err := s.pool.QueryRow(ctx, `SELECT mempher.merge_entities($1, $2, $3, $4, $5)`,
		string(cmd.Scope), string(cmd.Extractor), string(cmd.Alias), string(cmd.Into), cmd.At,
	).Scan(&id)
	if err != nil {
		return mempher.Entity{}, s.wrapFactError("merge entities", cmd.Scope, err)
	}
	// Read back by the name that stands, which leads to the entity whatever
	// it is called.
	entities, err := s.Entities(ctx, mempher.EntityQuery{
		Scope: cmd.Scope, Extractor: cmd.Extractor, MentionedIn: string(cmd.Into), AsOf: cmd.At,
	})
	if err != nil {
		return mempher.Entity{}, err
	}
	for _, e := range entities {
		if e.ID == mempher.EntityID(id) {
			return e, nil
		}
	}
	// Another merge took it between the two statements; the id is still the
	// answer to what was asked.
	return mempher.Entity{
		ID: mempher.EntityID(id), Scope: cmd.Scope, Extractor: cmd.Extractor,
	}, nil
}

// RelatedFacts walks the facts around the entities q names, through
// mempher.related_facts.
func (s *Store) RelatedFacts(ctx context.Context, q mempher.RelatedQuery) ([]mempher.RelatedFact, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: related facts: %w", err)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultRelatedFactsLimit
	}
	names := make([]string, len(q.Entities))
	for i, name := range q.Entities {
		names[i] = string(name)
	}
	rows, err := s.pool.Query(ctx,
		`SELECT hops, `+factColumns+` FROM mempher.related_facts($1, $2, $3, $4, $5, $6, $7)`,
		string(q.Scope), string(q.Extractor), names, int32(q.Hops), q.At, q.AsOf, int32(limit))
	if err != nil {
		return nil, s.wrapFactError("related facts", q.Scope, err)
	}
	defer rows.Close()

	out := make([]mempher.RelatedFact, 0, 16)
	for rows.Next() {
		var (
			f         mempher.Fact
			hops      int32
			id        [16]byte
			subject   string
			predicate string
			valid     pgtype.Range[pgtype.Timestamptz]
			episodes  []uuid.UUID
		)
		if err := rows.Scan(&hops, &id, &f.Scope, &f.Extractor, &subject, &predicate, &f.Object,
			&f.Statement, &valid, &f.Confidence, &episodes, &f.AssertedAt, &f.UpdatedAt); err != nil {
			return nil, fmt.Errorf("mempher/postgres: related facts: scan: %w", err)
		}
		fact, err := buildFact(f, id, subject, predicate, valid, episodes)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: related facts: %w", err)
		}
		out = append(out, mempher.RelatedFact{Fact: fact, Hops: int(hops)})
	}
	if err := rows.Err(); err != nil {
		return nil, s.wrapFactError("related facts", q.Scope, err)
	}
	return out, nil
}
