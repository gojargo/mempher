package eval_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/eval"
	"github.com/gojargo/mempher/extract"
	"github.com/gojargo/mempher/internal/pgtest"
	"github.com/gojargo/mempher/memphertest"
	"github.com/gojargo/mempher/postgres"
)

// echo stands in for the model that writes answers: it answers with the
// excerpts it was shown, so an answer holds the expected one exactly when
// recall returned the episode that states it.
type echo struct {
	mu      sync.Mutex
	prompts []string
	fail    string
}

func (e *echo) Model() string { return "echo" }

func (e *echo) Complete(_ context.Context, p extract.Prompt) ([]byte, error) {
	e.mu.Lock()
	e.prompts = append(e.prompts, p.User)
	e.mu.Unlock()
	if e.fail != "" && strings.Contains(p.User, e.fail) {
		return nil, errors.New("the provider is down")
	}
	_, excerpts, _ := strings.Cut(p.User, "EXCERPTS")
	excerpts, _, _ = strings.Cut(excerpts, "QUESTION")
	return json.Marshal(map[string]string{"answer": excerpts}) //nolint:wrapcheck // a test double
}

// substring stands in for the judge: correct when the response holds the
// expected answer.
type substring struct{}

func (substring) Model() string { return "substring" }

func (substring) Complete(_ context.Context, p extract.Prompt) ([]byte, error) {
	_, rest, _ := strings.Cut(p.User, "EXPECTED ANSWER\n\n")
	expected, rest, _ := strings.Cut(rest, "\n\nRESPONSE\n\n")
	response, _, _ := strings.Cut(rest, "\nGRADING RULE")
	return json.Marshal(map[string]any{ //nolint:wrapcheck // a test double
		"reason":  "compared as text",
		"correct": strings.Contains(response, expected),
	})
}

func TestRunQA(t *testing.T) {
	pool := pgtest.Pool(t)
	embedder := memphertest.NewEmbedder(dims)
	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("postgres.New: %v", err)
	}
	memory, err := mempher.New(mempher.Config{Store: store, Embedder: embedder})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}

	day := time.Date(2024, 3, 1, 9, 0, 0, 0, time.UTC)
	said := func(content string) mempher.AppendRequest {
		return mempher.AppendRequest{
			Content: content, Role: mempher.RoleUser, Source: "test", OccurredAt: day,
		}
	}
	haystack := []mempher.AppendRequest{
		said("My dog is called Biscuit."),
		said("I work as a carpenter."),
	}
	questions := []eval.Question{
		{ID: "dog", Type: "pets", Scope: "one", Query: "what is the dog called", Answer: "Biscuit",
			Haystack: haystack, Relevant: []int{0}, Grading: "be generous"},
		{ID: "cat", Type: "pets", Scope: "one", Query: "what is the cat called", Answer: "Mittens",
			Haystack: haystack, Relevant: []int{0}},
		{ID: "job", Type: "work", Scope: "one", Query: "what work do they do as a carpenter", Answer: "carpenter",
			Haystack: haystack},
		{ID: "down", Type: "work", Query: "what is the plumber called", Answer: "nobody",
			Haystack: []mempher.AppendRequest{said("The plumber came on Tuesday.")}},
	}
	answerer := &echo{fail: "plumber"}

	summaries, err := eval.RunQA(t.Context(), eval.QAConfig{
		Memory:   memory,
		Answerer: answerer,
		Judge:    substring{},
		Variants: []eval.Variant{{Name: "lexical", Channels: []mempher.Channel{mempher.ChannelLexical}}},
		Parallel: 3,
		Attempts: 1,
	}, eval.Dataset{Name: "tiny", Questions: func(yield func(eval.Question, error) bool) {
		for _, q := range questions {
			if !yield(q, nil) {
				return
			}
		}
	}})
	if err != nil {
		t.Fatalf("RunQA: %v", err)
	}

	s := summaries[0]
	if s.Questions != 3 || s.Correct != 2 || s.Errors != 1 {
		t.Fatalf("%d judged, %d correct, %d errors; want 3, 2, 1\n%s", s.Questions, s.Correct, s.Errors, s)
	}
	if got := s.ByType["pets"]; got == nil || got.Questions != 2 || got.Correct != 1 {
		t.Errorf("pets = %+v, want 1 of 2", got)
	}
	if s.Labelled != 2 || s.HitRate != 1 || s.CorrectGivenHit != 0.5 {
		t.Errorf("labelled %d, hit rate %v, accuracy with the evidence %v; want 2, 1, 0.5",
			s.Labelled, s.HitRate, s.CorrectGivenHit)
	}
	if s.Answerer != "echo" || s.Judge != "substring" {
		t.Errorf("summary names %q and %q", s.Answerer, s.Judge)
	}

	// Three questions shared a scope, so its haystack went in once.
	var episodes int
	if err := pool.QueryRow(t.Context(),
		`SELECT count(*) FROM mempher.episodes WHERE scope_id = 'eval:one'`).Scan(&episodes); err != nil {
		t.Fatalf("count: %v", err)
	}
	if episodes != len(haystack) {
		t.Errorf("the shared scope holds %d episodes, want %d", episodes, len(haystack))
	}
	const excerpt = "[Fri 1 Mar 2024 09:00] user: My dog is called Biscuit."
	if !strings.Contains(strings.Join(answerer.prompts, "\n"), excerpt) {
		t.Errorf("no answer prompt holds the dated excerpt %q:\n%s", excerpt, answerer.prompts[0])
	}
}

func TestRunQARejectsAnIncompleteConfig(t *testing.T) {
	_, err := eval.RunQA(t.Context(), eval.QAConfig{}, eval.Dataset{})
	if !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("err = %v, want ErrInvalidConfig", err)
	}
}

func TestLoCoMoLoader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "locomo.json")
	if err := os.WriteFile(path, []byte(`[{
		"sample_id": "conv-1",
		"conversation": {
			"speaker_a": "Ana", "speaker_b": "Ben",
			"session_1_date_time": "1:56 pm on 8 May, 2023",
			"session_1": [
				{"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a dog."},
				{"speaker": "Ben", "dia_id": "D1:2", "text": "", "blip_caption": "a beach at dusk"}
			],
			"session_2_date_time": "12:05 am on 1 June, 2023",
			"session_2": [{"speaker": "Ben", "dia_id": "D2:1", "text": "How is the dog?"}]
		},
		"qa": [
			{"question": "What did Ana adopt?", "answer": "a dog", "evidence": ["D1:1"], "category": 4},
			{"question": "When?", "answer": 2023, "evidence": ["D1:1; D2:1", "D"], "category": 2},
			{"question": "A trap", "adversarial_answer": "x", "evidence": ["D1:2"], "category": 5}
		]
	}]`), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	var got []eval.Question
	for q, err := range eval.LoCoMo(path).Questions {
		if err != nil {
			t.Fatalf("LoCoMo: %v", err)
		}
		got = append(got, q)
	}
	if len(got) != 2 {
		t.Fatalf("got %d questions, want 2: the adversarial one is left out", len(got))
	}
	first, second := got[0], got[1]
	if first.Scope != "conv-1" || second.Scope != "conv-1" || first.ID == second.ID {
		t.Errorf("scopes %q and %q, ids %q and %q", first.Scope, second.Scope, first.ID, second.ID)
	}
	if first.Type != "single-hop" || second.Type != "temporal" || second.Answer != "2023" {
		t.Errorf("types %q and %q, second answer %q", first.Type, second.Type, second.Answer)
	}
	if len(first.Haystack) != 3 {
		t.Fatalf("haystack holds %d turns, want 3", len(first.Haystack))
	}
	if c := first.Haystack[0].Content; c != "Ana: I adopted a dog." {
		t.Errorf("first turn = %q", c)
	}
	if c := first.Haystack[1].Content; c != "Ben: [shares a photo: a beach at dusk]" {
		t.Errorf("a turn that is only a photo = %q", c)
	}
	if want := time.Date(2023, 6, 1, 0, 5, 0, 0, time.UTC); !first.Haystack[2].OccurredAt.Equal(want) {
		t.Errorf("session 2 is dated %s, want %s", first.Haystack[2].OccurredAt, want)
	}
	if len(second.Relevant) != 2 || second.Relevant[0] != 0 || second.Relevant[1] != 2 {
		t.Errorf("evidence = %v, want turns 0 and 2", second.Relevant)
	}
	if first.Grading == "" {
		t.Error("LoCoMo questions carry no grading rule")
	}
}
