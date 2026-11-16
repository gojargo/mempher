package eval_test

import (
	"bytes"
	"context"
	"encoding/json"
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

// zepGrader stands in for Zep's grader: CORRECT when the generated answer
// holds the gold one.
type zepGrader struct{}

func (zepGrader) Model() string { return "zep-grader" }

func (zepGrader) Complete(_ context.Context, p extract.Prompt) ([]byte, error) {
	// The prompt's own example has a gold answer too; the real one follows.
	_, rest, _ := strings.Cut(p.User, "Now it's time for the real question:")
	_, rest, _ = strings.Cut(rest, "Gold answer: ")
	gold, rest, _ := strings.Cut(rest, "\nGenerated answer: ")
	generated, _, _ := strings.Cut(rest, "\n\nFirst, provide")
	verdict := "WRONG"
	if strings.Contains(generated, gold) {
		verdict = "CORRECT"
	}
	return json.Marshal(map[string]string{"is_correct": verdict, "reasoning": "compared as text"}) //nolint:wrapcheck // a test double
}

func TestRunQAWithZepPromptsAtSeveralSizes(t *testing.T) {
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
	said := func(content string, at int) mempher.AppendRequest {
		return mempher.AppendRequest{Content: content, Role: mempher.RoleUser, Source: "test",
			OccurredAt: day.Add(time.Duration(at) * time.Minute)}
	}
	// The answer is in the episode the question matches least, so it comes
	// back only when recall returns all three.
	haystack := []mempher.AppendRequest{
		said("the dog is called Biscuit", 0),
		said("the dog and the dog park and the dog walker", 1),
		said("the dog loves the dog park", 2),
	}
	questions := []eval.Question{{ID: "dog", Type: "pets", Scope: "s", Query: "the dog park dog walker",
		Answer: "Biscuit", Haystack: haystack}}

	answerer := &echo{}
	var recorded bytes.Buffer
	lexical := []mempher.Channel{mempher.ChannelLexical}
	summaries, err := eval.RunQA(t.Context(), eval.QAConfig{
		Memory:   memory,
		Answerer: answerer,
		Judge:    zepGrader{},
		Protocol: eval.ProtocolZep,
		Record:   &recorded,
		Attempts: 1,
		Variants: []eval.Variant{
			{Name: "one", Channels: lexical, Limit: 1},
			{Name: "three", Channels: lexical, Limit: 3},
		},
	}, eval.Dataset{Name: "zep", Questions: func(yield func(eval.Question, error) bool) {
		for _, q := range questions {
			if !yield(q, nil) {
				return
			}
		}
	}})
	if err != nil {
		t.Fatalf("RunQA: %v", err)
	}

	if one, three := summaries[0], summaries[1]; one.Correct != 0 || three.Correct != 1 ||
		one.MeanEpisodes != 1 || three.MeanEpisodes != 3 {
		t.Errorf("one episode: %d correct from %.0f episodes; three: %d correct from %.0f; want 0 and 1",
			one.Correct, one.MeanEpisodes, three.Correct, three.MeanEpisodes)
	}
	for _, system := range answerer.systems {
		if !strings.Contains(system, "helpful expert assistant") {
			t.Fatalf("Zep's response system prompt was not used:\n%s", system)
		}
	}
	if !strings.Contains(answerer.prompts[0], "# APPROACH (Think step by step):") ||
		!strings.Contains(answerer.prompts[0], "EXCERPTS are messages") {
		t.Errorf("the answer prompt is not Zep's with mempher's context:\n%s", answerer.prompts[0])
	}

	lines := strings.Split(strings.TrimSpace(recorded.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("recorded %d lines, want one per variant", len(lines))
	}
	var line struct {
		Variant, Context string
		Correct          bool
	}
	if err := json.Unmarshal([]byte(lines[1]), &line); err != nil {
		t.Fatalf("decode a recorded line: %v", err)
	}
	if line.Variant != "three" || !line.Correct || !strings.Contains(line.Context, "Biscuit") {
		t.Errorf("recorded %+v", line)
	}
}
