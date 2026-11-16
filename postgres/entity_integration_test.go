package postgres_test

import (
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/postgres"
)

// about builds a claim with a subject of the test's choosing.
func about(subject mempher.Subject, predicate mempher.Predicate, object string) mempher.Assertion {
	return mempher.Assertion{
		Subject:    subject,
		Predicate:  predicate,
		Object:     object,
		Statement:  string(subject) + " " + string(predicate) + " " + object,
		Valid:      mempher.Validity{From: y2023},
		Confidence: 0.9,
	}
}

// extraction applies one extraction of a fresh episode.
func extraction(
	t *testing.T,
	store *postgres.Store,
	scope mempher.ScopeID,
	aliases []mempher.Alias,
	assert ...mempher.Assertion,
) []mempher.Fact {
	t.Helper()
	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope:     scope,
		Extractor: testExtractor,
		Episodes:  []mempher.EpisodeID{seedEpisode(t, store, scope, "an episode")},
		Assert:    assert,
		Alias:     aliases,
		At:        epoch.Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}
	return facts
}

func TestApplyExtractionWritesUnderTheEntitysName(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	const scope = "entity:alias"

	extraction(t, store, scope, nil, about("the user's sister", "lives_in", "Paris"))
	facts := extraction(t, store, scope,
		[]mempher.Alias{{Name: "Anna", Of: "the user's sister"}},
		about("Anna", "plays", "cello"))
	if len(facts) != 1 || facts[0].Subject != "the user's sister" {
		t.Fatalf("facts = %+v, want the one about Anna stored under her sister's name", facts)
	}

	entities, err := store.Entities(t.Context(), mempher.EntityQuery{
		Scope: scope, Extractor: testExtractor, MentionedIn: "what does ANNA play?", AsOf: epoch.Add(time.Hour),
	})
	if err != nil {
		t.Fatalf("Entities: %v", err)
	}
	if len(entities) != 1 {
		t.Fatalf("got %d entities, want the one Anna names", len(entities))
	}
	got := entities[0]
	if got.Name != "the user's sister" || !slices.Equal(got.Aliases, []mempher.Subject{"Anna"}) ||
		got.Facts != 2 || got.ID.IsZero() {
		t.Errorf("entity = %+v, want the sister, also called Anna, with 2 facts", got)
	}

	// Before the alias was learned, the name led nowhere.
	earlier, err := store.Entities(t.Context(), mempher.EntityQuery{
		Scope: scope, Extractor: testExtractor, MentionedIn: "Anna", AsOf: epoch,
	})
	if err != nil {
		t.Fatalf("Entities: %v", err)
	}
	if len(earlier) != 0 {
		t.Errorf("as of before the alias: %+v, want nobody", earlier)
	}

	about, err := store.Facts(t.Context(), mempher.FactQuery{
		Scope: scope, Extractor: testExtractor, AsOf: epoch.Add(time.Hour), At: y2024,
		Subjects: []mempher.Subject{"anna"},
	})
	if err != nil {
		t.Fatalf("Facts: %v", err)
	}
	if len(about) != 2 {
		t.Errorf("%d facts about anna, want both of her sister's", len(about))
	}
}

func TestMergeEntities(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	const scope = "entity:merge"

	extraction(t, store, scope, nil, about("my sister", "lives_in", "Paris"))
	extraction(t, store, scope, nil, about("Anna", "lives_in", "Paris"), about("Anna", "plays", "cello"))

	cmd := mempher.MergeCommand{
		Scope: scope, Extractor: testExtractor, Alias: "Anna", Into: "my sister", At: epoch.Add(2 * time.Minute),
	}
	merged, err := store.MergeEntities(t.Context(), cmd)
	if err != nil {
		t.Fatalf("MergeEntities: %v", err)
	}
	if merged.Name != "my sister" || !slices.Equal(merged.Aliases, []mempher.Subject{"Anna"}) || merged.Facts != 2 {
		t.Errorf("merged = %+v, want my sister, also called Anna, with 2 facts", merged)
	}
	again, err := store.MergeEntities(t.Context(), cmd)
	if err != nil {
		t.Fatalf("MergeEntities again: %v", err)
	}
	if again.ID != merged.ID {
		t.Errorf("the second merge returned %s, the first %s", again.ID, merged.ID)
	}

	facts := currentFacts(t, store, scope)
	if len(facts) != 2 {
		t.Fatalf("got %d facts, want the duplicate folded into one and the other moved", len(facts))
	}
	for _, f := range facts {
		if f.Subject != "my sister" {
			t.Errorf("fact %s %s is still about %q", f.Predicate, f.Object, f.Subject)
		}
		if f.Predicate == "lives_in" && len(f.Episodes) != 2 {
			t.Errorf("the folded fact is supported by %d episodes, want both", len(f.Episodes))
		}
	}

	cmd.Scope = "entity:nowhere"
	if _, err := store.MergeEntities(t.Context(), cmd); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("merge in a scope that does not exist: %v, want ErrNotFound", err)
	}
	cmd.Scope, cmd.Alias = scope, " "
	if _, err := store.MergeEntities(t.Context(), cmd); !errors.Is(err, mempher.ErrInvalidEntity) {
		t.Errorf("merge of a blank name: %v, want ErrInvalidEntity", err)
	}
}

func TestRelatedFacts(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	const scope = "entity:graph"

	extraction(t, store, scope, []mempher.Alias{{Name: "Anna", Of: "the user's sister"}},
		about("the user", "sister", "Anna"),
		about("the user", "likes", "tea"),
		about("Anna", "works_at", "Acme"),
		about("Acme", "based_in", "Lyon"),
		about("Bob", "likes", "chess"))

	walk := func(hops int) []string {
		t.Helper()
		related, err := store.RelatedFacts(t.Context(), mempher.RelatedQuery{
			Scope: scope, Extractor: testExtractor, Entities: []mempher.Subject{"acme"},
			Hops: hops, AsOf: epoch.Add(time.Hour), At: y2024,
		})
		if err != nil {
			t.Fatalf("RelatedFacts: %v", err)
		}
		out := make([]string, len(related))
		for i, r := range related {
			out[i] = string(rune('0'+r.Hops)) + " " + string(r.Subject) + " " + string(r.Predicate)
		}
		return out
	}
	if got, want := walk(0), []string{"0 Acme based_in", "0 the user's sister works_at"}; !slices.Equal(got, want) {
		t.Errorf("0 hops = %q, want %q", got, want)
	}
	// The object "Anna" names the sister, so the user is one hop from Acme
	// through her, and the user's own facts two.
	want := []string{
		"0 Acme based_in", "0 the user's sister works_at", "1 the user sister", "2 the user likes",
	}
	if got := walk(2); !slices.Equal(got, want) {
		t.Errorf("2 hops = %q, want %q", got, want)
	}

	_, err := store.RelatedFacts(t.Context(), mempher.RelatedQuery{
		Scope: scope, Extractor: testExtractor, Entities: []mempher.Subject{"Acme"},
		Hops: mempher.MaxRelatedHops + 1, AsOf: epoch, At: y2024,
	})
	if !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("too many hops: %v, want ErrInvalidConfig", err)
	}
}

func TestForgetCountsEntities(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	const scope = "entity:forget"

	extraction(t, store, scope, nil, about("the user", "likes", "tea"), about("Bob", "likes", "chess"))
	result, err := store.Forget(t.Context(), mempher.ForgetRequest{Scope: scope})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if result.Entities != 2 || result.Facts != 2 {
		t.Errorf("forgot %d entities and %d facts, want 2 and 2", result.Entities, result.Facts)
	}
}
