// L1: the entities facts are about, and the names they go by.

package mempher

import (
	"fmt"
	"strings"
	"time"
)

// Limits on entity resolution, enforced here and again by the store.
const (
	// MaxAliasesPerExtraction caps how many names one extraction may
	// declare, for the reason [MaxAssertionsPerExtraction] caps its facts.
	MaxAliasesPerExtraction = 64
	// MaxRelatedEntities caps how many entities a [RelatedQuery] may start
	// from.
	MaxRelatedEntities = 64
	// MaxRelatedHops is the furthest a [RelatedQuery] may walk.
	MaxRelatedHops = 8
)

// Entity is one thing a scope's facts are about, with every name it goes by.
//
// It exists because a triple is identity only while its subject is spelled one
// way. "Anna" in one episode and "the user's sister" in the next would be two
// subjects, so nothing would dedupe, nothing would supersede, and a question
// about one would miss what is known under the other. A [FactStore] therefore
// stores every fact under its entity's Name, whichever of the entity's names
// the [Assertion] used, and accepts any of them wherever a subject is asked
// for.
//
// Two names are the same name when they differ only in case or in runs of
// whitespace. Anything beyond that -- that "Anna" and "Anna Martin" are one
// person -- is a judgement, and is declared with an [Alias] rather than guessed.
//
// Like a [Fact] it belongs to one extractor: two extractors' opinions of who is
// who do not merge either.
type Entity struct {
	// ID is the uuidv7 surrogate key, minted by the database. It is gone
	// once the entity has been merged into another.
	ID EntityID
	// Scope is the partition this entity belongs to.
	Scope ScopeID
	// Extractor identifies whose view of the scope this is.
	Extractor ExtractorID
	// Name is the spelling its facts carry as their [Fact.Subject]: the
	// first one written.
	Name Subject
	// Aliases are the other names it goes by, sorted.
	Aliases []Subject
	// Facts is how many facts are about it, whatever their window.
	Facts int
	// CreatedAt is system time: when the entity was first named.
	CreatedAt time.Time
	// UpdatedAt is when it last gained a name or absorbed another entity.
	UpdatedAt time.Time
}

// Alias declares that two names are one entity: Name is another name for
// whatever Of names.
//
// An [Extractor] returns one when the episodes say so -- "I'm Alice", "my
// sister Anna" -- and an operator applies one with [FactStore.MergeEntities].
// Of is the side that stands. If Name already named a different entity, that
// entity is merged into Of's: its names move, its facts take the surviving
// subject, and where both held the same claim over overlapping windows the
// surviving entity's fact is the one kept.
//
// There is no inverse. A wrong merge is repaired the way any wrong extraction
// is: drop the projection and replay L0 through a fixed extractor.
type Alias struct {
	// Name is the additional name. Required.
	Name Subject
	// Of is a name the entity already goes by, or the name to create it
	// under. Required.
	Of Subject
}

// MergeCommand is one [Alias] applied outside an extraction.
type MergeCommand struct {
	// Scope is the partition to write to. Required, and must exist.
	Scope ScopeID
	// Extractor is whose view of the scope to change. Required.
	Extractor ExtractorID
	// Alias is the name to add, or the entity to merge away.
	Alias Subject
	// Into is the entity that stands.
	Into Subject
	// At is system time for the write, from the [Clock].
	At time.Time
}

// EntityQuery lists a scope's entities.
type EntityQuery struct {
	// Scope restricts the query to one partition. Required.
	Scope ScopeID
	// Extractor selects whose entities to read. Required.
	Extractor ExtractorID
	// MentionedIn, when non-empty, keeps only the entities this text names,
	// by any of their names, as whole words and whatever the case. It is how
	// a worker finds who an episode is talking about.
	MentionedIn string
	// AsOf is the system-time cut: entities and names learned after it are
	// not visible. Required.
	AsOf time.Time
	// Limit caps how many entities are returned. Zero means a store-chosen
	// default.
	Limit int
}

// RelatedQuery asks for the facts around some entities.
//
// A fact whose object is the name of an entity ties its subject to that
// entity, so a scope's facts are a graph, and this walks it: the facts that
// touch the named entities, then the facts of whatever those tie them to, out
// to Hops.
type RelatedQuery struct {
	// Scope restricts the query to one partition. Required.
	Scope ScopeID
	// Extractor selects whose facts to read. Required.
	Extractor ExtractorID
	// Entities are where the walk starts, each by any of its names. Required,
	// at most [MaxRelatedEntities]. A name nothing goes by contributes
	// nothing.
	Entities []Subject
	// Hops is how far to walk, at most [MaxRelatedHops]. Zero keeps to the
	// facts that touch the named entities themselves, as subject or object.
	Hops int
	// AsOf is the system-time cut, as in [FactQuery]. Required.
	AsOf time.Time
	// At is the event-time instant the facts must hold at. Required.
	At time.Time
	// Limit caps how many facts are returned, nearest first. Zero means a
	// store-chosen default.
	Limit int
}

// RelatedFact is a fact found by a [RelatedQuery], with how far it lay.
type RelatedFact struct {
	Fact
	// Hops is the distance of the nearer of the two entities the fact ties
	// from the entities the query named.
	Hops int
}

// validateEntityName holds a name to what a subject is held to, and reports
// the failure as an entity's rather than a fact's.
func validateEntityName(name Subject, what string) error {
	if err := name.Validate(); err != nil {
		return fmt.Errorf("mempher: %s: %w: %w", what, ErrInvalidEntity, err)
	}
	return nil
}

// Validate reports whether the alias can be applied.
func (a Alias) Validate() error {
	if err := validateEntityName(a.Name, "alias name"); err != nil {
		return err
	}
	return validateEntityName(a.Of, "alias target")
}

// Validate reports whether the merge can be applied.
func (c MergeCommand) Validate() error {
	if err := c.Scope.Validate(); err != nil {
		return err
	}
	switch {
	case c.Extractor == "":
		return fmt.Errorf("mempher: merge names no extractor: %w", ErrInvalidConfig)
	case c.At.IsZero():
		return fmt.Errorf("mempher: merge has no system time: %w", ErrInvalidConfig)
	}
	return Alias{Name: c.Alias, Of: c.Into}.Validate()
}

// Validate reports whether the entity query is usable.
func (q EntityQuery) Validate() error {
	if err := q.Scope.Validate(); err != nil {
		return err
	}
	switch {
	case q.Extractor == "":
		return fmt.Errorf("mempher: entity query names no extractor: %w", ErrInvalidConfig)
	case q.AsOf.IsZero():
		return fmt.Errorf("mempher: entity query has no AsOf: %w", ErrInvalidConfig)
	case q.Limit < 0:
		return fmt.Errorf("mempher: entity query limit is %d: %w", q.Limit, ErrInvalidConfig)
	case len(q.MentionedIn) > MaxContentLen:
		return fmt.Errorf("mempher: entity query text is %d bytes, limit is %d: %w",
			len(q.MentionedIn), MaxContentLen, ErrInvalidContent)
	}
	return nil
}

// Validate reports whether the related-facts query is usable.
func (q RelatedQuery) Validate() error {
	if err := q.Scope.Validate(); err != nil {
		return err
	}
	switch {
	case q.Extractor == "":
		return fmt.Errorf("mempher: related query names no extractor: %w", ErrInvalidConfig)
	case len(q.Entities) == 0:
		return fmt.Errorf("mempher: related query names no entity: %w", ErrInvalidConfig)
	case len(q.Entities) > MaxRelatedEntities:
		return fmt.Errorf("mempher: related query names %d entities, limit is %d: %w",
			len(q.Entities), MaxRelatedEntities, ErrInvalidConfig)
	case q.Hops < 0 || q.Hops > MaxRelatedHops:
		return fmt.Errorf("mempher: related query walks %d hops, must be 0 to %d: %w",
			q.Hops, MaxRelatedHops, ErrInvalidConfig)
	case q.AsOf.IsZero() || q.At.IsZero():
		return fmt.Errorf("mempher: related query needs AsOf and At: %w", ErrInvalidConfig)
	case q.Limit < 0:
		return fmt.Errorf("mempher: related query limit is %d: %w", q.Limit, ErrInvalidConfig)
	}
	for _, name := range q.Entities {
		if err := validateEntityName(name, "related entity"); err != nil {
			return err
		}
	}
	return nil
}

// blank reports whether a name is nothing but whitespace, which no entity can
// be found by.
func blank(s string) bool { return strings.TrimSpace(s) == "" }
