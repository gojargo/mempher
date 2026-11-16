package eval_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/eval"
	"github.com/gojargo/mempher/extract"
	"github.com/gojargo/mempher/internal/pgtest"
	"github.com/gojargo/mempher/memphertest"
	"github.com/gojargo/mempher/postgres"
)

func TestBEAMLoader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "beam-100K.jsonl")
	conversation := `{"conversation_id": "7", "chat": [
		[
			{"id": 0, "role": "user", "content": "My first sprint ends on March 29. ->-> 1,1", "time_anchor": "March-15-2024"},
			{"id": 1, "role": "assistant", "content": "Noted.", "time_anchor": "March-15-2024"}
		],
		[
			{"id": 2, "role": "user", "content": "Always highlight code. ->-> 2,1", "time_anchor": "April-05-2024"},
			{"id": 3, "role": "assistant", "content": "", "time_anchor": "April-05-2024"}
		]
	], "probing_questions": {
		"information_extraction": [{"question": "When does my first sprint end?", "answer": "March 29.",
			"source_chat_ids": [0], "rubric": ["LLM response should state: March 29"]}],
		"instruction_following": [{"question": "Show me a login feature.", "expected_compliance": "highlighted code",
			"source_chat_ids": {"instruction": [2], "other": [[0]]}, "rubric": ["code blocks", "highlighting"]}],
		"abstention": [{"question": "What did the reviewer say?", "ideal_response": "There is no information.",
			"source_chat_ids": null, "rubric": ["states there is no information"]}]
	}}`
	if err := os.WriteFile(path, []byte(strings.ReplaceAll(conversation, "\n", " ")+"\n"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	ds := eval.BEAM(path)
	if ds.Name != "BEAM (beam-100K)" || ds.AnswerStyle == "" || ds.Instructions == "" {
		t.Errorf("dataset %q, style %q, instructions %q", ds.Name, ds.AnswerStyle, ds.Instructions)
	}
	var got []eval.Question
	for q, err := range ds.Questions {
		if err != nil {
			t.Fatalf("BEAM: %v", err)
		}
		got = append(got, q)
	}
	if len(got) != 3 {
		t.Fatalf("got %d questions, want 3", len(got))
	}
	// Types come in the benchmark's order, whatever the order in the file.
	types := []string{got[0].Type, got[1].Type, got[2].Type}
	if !slices.Equal(types, []string{"abstention", "information_extraction", "instruction_following"}) {
		t.Errorf("types = %v", types)
	}
	abstain, extraction, instruction := got[0], got[1], got[2]
	if extraction.Scope != "beam-7" || extraction.Query != "When does my first sprint end?" ||
		extraction.Answer != "March 29." || len(extraction.Rubric) != 1 {
		t.Errorf("extraction question = %+v", extraction)
	}
	if abstain.Answer != "There is no information." || len(abstain.Relevant) != 0 {
		t.Errorf("abstention answer %q, relevant %v", abstain.Answer, abstain.Relevant)
	}
	if instruction.Answer != "highlighted code" || !slices.Equal(instruction.Relevant, []int{2, 0}) {
		t.Errorf("instruction answer %q, relevant %v", instruction.Answer, instruction.Relevant)
	}

	haystack := extraction.Haystack
	if len(haystack) != 3 {
		t.Fatalf("haystack holds %d turns, want 3: an empty turn is skipped", len(haystack))
	}
	if c := haystack[0].Content; c != "My first sprint ends on March 29." {
		t.Errorf("the position marker was kept: %q", c)
	}
	if haystack[0].Role != mempher.RoleUser || haystack[1].Role != mempher.RoleAssistant {
		t.Errorf("roles %q and %q", haystack[0].Role, haystack[1].Role)
	}
	if want := time.Date(2024, 4, 5, 9, 0, 0, 0, time.UTC); !haystack[2].OccurredAt.Equal(want) {
		t.Errorf("session 2 is dated %s, want %s", haystack[2].OccurredAt, want)
	}
	if !extraction.Asked.After(haystack[2].OccurredAt) {
		t.Errorf("asked %s, before the last turn", extraction.Asked)
	}
}

// grader stands in for a rubric judge: an item met in full when the response
// holds it, by half when it holds the item's first word, and not at all
// otherwise.
type grader struct{}

func (grader) Model() string { return "grader" }

func (grader) Complete(_ context.Context, p extract.Prompt) ([]byte, error) {
	_, rest, _ := strings.Cut(p.User, "LLM RESPONSE:\n")
	response, rest, _ := strings.Cut(rest, "\n\nRUBRIC CRITERION:\n")
	item, _, _ := strings.Cut(rest, "\n\nSCORING GUIDELINES")
	score := 0.0
	switch first, _, _ := strings.Cut(item, " "); {
	case strings.Contains(response, item):
		score = 1
	case strings.Contains(response, first):
		score = 0.5
	}
	return json.Marshal(map[string]any{"reason": "compared as text", "score": score}) //nolint:wrapcheck // a test double
}

func TestRunQAScoresRubricsItemByItem(t *testing.T) {
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
	haystack := []mempher.AppendRequest{{
		Content: "The sprint ends on March 29 and the demo is in April.", Role: mempher.RoleUser,
		Source: "test", OccurredAt: day,
	}}
	questions := []eval.Question{
		{ID: "full", Type: "dates", Scope: "a", Query: "when does the sprint end",
			Rubric: []string{"March 29"}, Haystack: haystack},
		{ID: "half", Type: "dates", Scope: "a", Query: "when does the sprint end and when is the demo",
			Rubric: []string{"March 29", "April 12"}, Haystack: haystack},
		{ID: "none", Type: "people", Scope: "b", Query: "who reviewed the sprint",
			Rubric: []string{"Dana reviewed it"}, Haystack: []mempher.AppendRequest{{
				Content: "Nothing about reviews here.", Role: mempher.RoleUser, Source: "test", OccurredAt: day,
			}}},
	}
	answerer := &echo{}
	summaries, err := eval.RunQA(t.Context(), eval.QAConfig{
		Memory:    memory,
		Answerer:  answerer,
		Judge:     grader{},
		Variants:  []eval.Variant{{Name: "lexical", Channels: []mempher.Channel{mempher.ChannelLexical}}},
		Attempts:  1,
		Haystacks: 2,
		Outstanding: func(context.Context, mempher.ScopeID) (int, error) {
			return 0, nil
		},
	}, eval.Dataset{Name: "rubric", AnswerStyle: "Answer at length.", Questions: func(yield func(eval.Question, error) bool) {
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
	// full scores 1, half scores (1 + 0.5) / 2, none scores 0.
	if !s.Rubric || s.Questions != 3 || s.Correct != 1 || s.Score != 1.75/3 {
		t.Fatalf("rubric %v, %d judged, %d fully met, score %v; want true, 3, 1, %v\n%s",
			s.Rubric, s.Questions, s.Correct, s.Score, 1.75/3, s)
	}
	if got := s.ByType["dates"]; got == nil || got.Score != 0.875 {
		t.Errorf("dates = %+v, want a score of 0.875", got)
	}
	if !strings.Contains(s.String(), "rubric score") {
		t.Errorf("the summary does not report a rubric score:\n%s", s)
	}
	for _, prompt := range answerer.systems {
		if !strings.Contains(prompt, "Answer at length.") || strings.Contains(prompt, "shortest answer") {
			t.Fatalf("the dataset's answer style was not used:\n%s", prompt)
		}
	}
	if len(answerer.systems) != 3 {
		t.Errorf("%d answer prompts, want 3", len(answerer.systems))
	}
}
