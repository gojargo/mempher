package eval_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/gojargo/mempher"
	embedopenai "github.com/gojargo/mempher/embed/openai"
	"github.com/gojargo/mempher/eval"
	"github.com/gojargo/mempher/extract"
	extractopenai "github.com/gojargo/mempher/extract/openai"
	"github.com/gojargo/mempher/internal/pgtest"
	"github.com/gojargo/mempher/ops"
	"github.com/gojargo/mempher/postgres"
)

// TestQA runs a benchmark end to end against real models and prints what it
// measured: recall, an answer written from what was recalled, and a judge.
//
// Everything is one OpenAI-compatible endpoint, which most providers and every
// local server offer:
//
//	LOCOMO=/path/to/locomo10.json \
//	QA_URL=https://api.example.com/v1 QA_API_KEY=... \
//	QA_ANSWER_MODEL=... QA_JUDGE_MODEL=... QA_EXTRACT_MODEL=... \
//	QA_EMBED_MODEL=... QA_EMBED_DIMS=1024 \
//	go test ./eval/ -run '^TestQA$' -v -timeout 600m
//
// LONGMEMEVAL=/path/to/longmemeval_s runs that benchmark instead. Leaving
// QA_EXTRACT_MODEL unset runs without facts, which is the cheap half: nothing
// then reads the haystack but the embedder.
//
// QA_MAX_ASSERTIONS raises how many facts one extraction may assert, which
// matters for a haystack ingested in batches of sixty-four dense turns.
// QA_QUESTIONS caps the run, QA_PARALLEL answers that many questions of one
// haystack at once, QA_RPS caps requests per second across every model, and
// QA_K and QA_MAX_TOKENS set the recall limit and budget, and QA_VARIANTS
// keeps only some of "episodes", "episodes and facts" and "facts". QA_SIZES
// measures one ingestion at several recall sizes instead.
//
// QA_PROTOCOL=zep writes and grades answers with the prompts of Zep's LoCoMo
// harness, and QA_RECORD=/path/to/answers.jsonl keeps every answer.
//
// BEAM=/path/to/beam-1M.jsonl runs BEAM, converted as [eval.BEAM] describes.
// Its conversations are long enough that QA_HAYSTACKS, how many are ingested
// and answered at once, decides whether a run takes hours or a day. Some of its
// turns exceed an embedding model's input, which QA_EMBED_MAX_BYTES cuts to;
// QA_MAX_PROMPT_BYTES raises what one extraction may send.
func TestQA(t *testing.T) {
	var dataset eval.Dataset
	switch {
	case os.Getenv("LOCOMO") != "":
		dataset = eval.LoCoMo(os.Getenv("LOCOMO"))
	case os.Getenv("BEAM") != "":
		dataset = eval.BEAM(os.Getenv("BEAM"))
	case os.Getenv("LONGMEMEVAL") != "" && os.Getenv("QA_URL") != "":
		dataset = eval.LongMemEval(os.Getenv("LONGMEMEVAL"))
	default:
		t.Skip("set LOCOMO, BEAM or LONGMEMEVAL to a benchmark file, and QA_URL to a model endpoint")
	}
	url := os.Getenv("QA_URL")
	if url == "" || os.Getenv("QA_ANSWER_MODEL") == "" || os.Getenv("QA_EMBED_MODEL") == "" {
		t.Skip("set QA_URL, QA_ANSWER_MODEL and QA_EMBED_MODEL to run the benchmark end to end")
	}
	client := &http.Client{
		Timeout:   5 * time.Minute,
		Transport: newPatient(floatFromEnv("QA_RPS", 4)),
	}
	completer := func(model string) extract.Completer {
		c, err := extractopenai.New(extractopenai.Config{
			BaseURL: url, APIKey: os.Getenv("QA_API_KEY"), Model: model,
			Lenient: os.Getenv("QA_LENIENT") != "", HTTPClient: client,
		})
		if err != nil {
			t.Fatalf("completer %s: %v", model, err)
		}
		return c
	}
	answerer := completer(os.Getenv("QA_ANSWER_MODEL"))
	judge := answerer
	if model := os.Getenv("QA_JUDGE_MODEL"); model != "" {
		judge = completer(model)
	}

	embedder, err := embedopenai.New(embedopenai.Config{
		BaseURL: url, APIKey: os.Getenv("QA_API_KEY"),
		Model:      os.Getenv("QA_EMBED_MODEL"),
		Dimensions: intFromEnv("QA_EMBED_DIMS", 0),
		MaxBatch:   intFromEnv("QA_EMBED_BATCH", 0),
		// A long episode is embedded by its start rather than failing its
		// batch: OpenAI's limit is about 8,000 tokens.
		MaxInputBytes: intFromEnv("QA_EMBED_MAX_BYTES", 0),
		HTTPClient:    client,
	})
	if err != nil {
		t.Fatalf("embedder: %v", err)
	}

	var extractor mempher.Extractor
	if model := os.Getenv("QA_EXTRACT_MODEL"); model != "" {
		extractor, err = extract.New(completer(model), extract.Options{
			MaxAssertions:  intFromEnv("QA_MAX_ASSERTIONS", 0),
			MaxPromptBytes: intFromEnv("QA_MAX_PROMPT_BYTES", 0),
			Instructions:   dataset.Instructions,
		})
		if err != nil {
			t.Fatalf("extract.New: %v", err)
		}
	}

	pool := pgtest.Pool(t)
	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("postgres.New: %v", err)
	}
	memory, err := mempher.New(mempher.Config{Store: store, Embedder: embedder, Extractor: extractor})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Extractor: extractor, ID: "qa",
		// One job per lease, so that haystacks drained side by side share the
		// queue's work rather than one of them leasing all of it.
		Batch: 1,
	})
	if err != nil {
		t.Fatalf("mempher.NewWorker: %v", err)
	}

	episodes := []mempher.Channel{mempher.ChannelSemantic, mempher.ChannelLexical}
	variants := []eval.Variant{{Name: "episodes", Channels: episodes}}
	if extractor != nil {
		variants = append(variants,
			eval.Variant{Name: "episodes and facts", Channels: append(episodes, mempher.ChannelFact)},
			eval.Variant{Name: "facts", Channels: []mempher.Channel{mempher.ChannelFact}})
	}
	// QA_SIZES replaces the variants with episodes and facts recalled at
	// several sizes, as "episodes:facts" pairs, comma-separated: "5:8,10:32"
	// answers every question twice from one ingestion.
	if sizes := os.Getenv("QA_SIZES"); sizes != "" {
		variants = nil
		for _, size := range strings.Split(sizes, ",") {
			k, f, ok := strings.Cut(strings.TrimSpace(size), ":")
			limit, errK := strconv.Atoi(k)
			facts, errF := strconv.Atoi(f)
			if !ok || errK != nil || errF != nil || limit <= 0 || facts < 0 {
				t.Fatalf("QA_SIZES: %q is not episodes:facts", size)
			}
			channels := slices.Clone(episodes)
			if facts > 0 {
				channels = append(channels, mempher.ChannelFact)
			}
			variants = append(variants, eval.Variant{
				Name:     fmt.Sprintf("%d episodes, %d facts", limit, facts),
				Channels: channels, Limit: limit, FactLimit: facts,
			})
		}
	}
	// QA_VARIANTS keeps only the named variants, comma-separated: every
	// variant is a judged answer per question, which is most of a run's cost.
	if names := os.Getenv("QA_VARIANTS"); names != "" {
		keep := strings.Split(names, ",")
		variants = slices.DeleteFunc(variants, func(v eval.Variant) bool {
			return !slices.Contains(keep, v.Name)
		})
		if len(variants) == 0 {
			t.Fatalf("QA_VARIANTS=%q names no variant", names)
		}
	}

	// QA_RECORD writes every answer, with its context and verdict, as JSON
	// lines to a file.
	var record io.Writer
	if path := os.Getenv("QA_RECORD"); path != "" {
		f, err := os.Create(path)
		if err != nil {
			t.Fatalf("QA_RECORD: %v", err)
		}
		defer func() { _ = f.Close() }()
		record = f
	}

	failures := 0
	started := time.Now()
	summaries, err := eval.RunQA(t.Context(), eval.QAConfig{
		Memory:    memory,
		Worker:    worker,
		Answerer:  answerer,
		Judge:     judge,
		Variants:  variants,
		Limit:     intFromEnv("QA_K", eval.DefaultLimit),
		FactLimit: intFromEnv("QA_FACTS", 0),
		MaxTokens: intFromEnv("QA_MAX_TOKENS", 0),
		Parallel:  intFromEnv("QA_PARALLEL", 4),
		Haystacks: intFromEnv("QA_HAYSTACKS", 1),
		Protocol:  eval.QAProtocol(os.Getenv("QA_PROTOCOL")),
		Record:    record,
		Outstanding: func(ctx context.Context, scope mempher.ScopeID) (int, error) {
			jobs, err := store.Jobs(ctx, ops.JobQuery{
				Scope:  scope,
				States: []mempher.JobState{mempher.JobStatePending, mempher.JobStateRunning},
				Limit:  1,
			})
			return len(jobs), err
		},
		Progress: func(done int, last eval.QAResult) {
			if last.Err != nil && failures < 5 {
				failures++
				t.Logf("  %s has no verdict: %v", last.ID, last.Err)
			}
			if done%25 == 0 {
				t.Logf("  %d questions, last %s correct=%v err=%v", done, last.ID, last.Correct, last.Err)
			}
		},
	}, capped(dataset, intFromEnv("QA_QUESTIONS", 0)))
	if err != nil {
		t.Fatalf("RunQA: %v", err)
	}

	// The scopes are kept, so what was built can be counted: a run whose
	// extraction jobs died measured a memory with no facts in it, and would
	// otherwise report that as a result.
	var facts, entities, aliases, unfinished int
	if err := pool.QueryRow(t.Context(), `
		SELECT (SELECT count(*) FROM mempher.facts), (SELECT count(*) FROM mempher.entities),
		       (SELECT count(*) FROM mempher.entity_aliases),
		       (SELECT count(*) FROM mempher.jobs WHERE state <> 'done')`).
		Scan(&facts, &entities, &aliases, &unfinished); err != nil {
		t.Fatalf("count what was built: %v", err)
	}
	t.Logf("embedder %s, extractor %v, wall clock %s", embedder.Model(), extractorID(extractor),
		time.Since(started).Round(time.Second))
	t.Logf("built %d facts about %d entities under %d names; %d jobs did not succeed",
		facts, entities, aliases, unfinished)
	logEntities(t, pool)
	for _, summary := range summaries {
		t.Logf("\n%s", summary)
	}
	if unfinished > 0 {
		rows, err := pool.Query(t.Context(), `
			SELECT kind || ' ' || state || ' x' || count(*) || ': ' || left(coalesce(min(last_error), ''), 300)
			FROM mempher.jobs WHERE state <> 'done' GROUP BY kind, state`)
		if err != nil {
			t.Fatalf("read the unfinished jobs: %v", err)
		}
		for rows.Next() {
			var line string
			if err := rows.Scan(&line); err != nil {
				t.Fatalf("scan: %v", err)
			}
			t.Logf("  %s", line)
		}
		rows.Close()
		t.Errorf("%d jobs did not succeed, so the memory measured is incomplete", unfinished)
	}
	if summaries[0].Questions == 0 {
		t.Fatal("no question was judged")
	}
}

// logEntities prints the entities with the most facts and the names each goes
// by. Two people merged into one is the failure an accuracy figure hides, and
// it is visible here at a glance.
func logEntities(t *testing.T, pool *pgxpool.Pool) {
	t.Helper()
	rows, err := pool.Query(t.Context(), `
		SELECT e.scope_id || '  ' || e.name || '  (' ||
		       (SELECT count(*) FROM mempher.facts f
		        WHERE f.scope_id = e.scope_id AND f.extractor = e.extractor AND f.subject = e.name) ||
		       ' facts)  also: ' ||
		       coalesce((SELECT string_agg(a.alias, ', ' ORDER BY a.alias) FROM mempher.entity_aliases a
		                 WHERE a.entity_id = e.id AND a.alias <> e.name), 'nothing')
		FROM mempher.entities e
		ORDER BY (SELECT count(*) FROM mempher.facts f
		          WHERE f.scope_id = e.scope_id AND f.extractor = e.extractor AND f.subject = e.name) DESC
		LIMIT 20`)
	if err != nil {
		t.Fatalf("read the entities: %v", err)
	}
	defer rows.Close()
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			t.Fatalf("scan: %v", err)
		}
		t.Logf("  %s", line)
	}
}

func extractorID(e mempher.Extractor) string {
	if e == nil {
		return "none"
	}
	return string(e.Model())
}

func floatFromEnv(key string, fallback float64) float64 {
	if raw := os.Getenv(key); raw != "" {
		if value, err := strconv.ParseFloat(raw, 64); err == nil && value > 0 {
			return value
		}
	}
	return fallback
}

// patient is a transport for a benchmark run: it spaces requests out, and
// tries again when a provider says it is busy. A run is thousands of calls, and
// one rate-limit reply should not fail a job or cost a question its verdict.
type patient struct {
	ticks <-chan time.Time
}

func newPatient(perSecond float64) *patient {
	return &patient{ticks: time.Tick(time.Duration(float64(time.Second) / perSecond))}
}

func (p *patient) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		var err error
		if body, err = io.ReadAll(req.Body); err != nil {
			return nil, fmt.Errorf("read the request body: %w", err)
		}
		_ = req.Body.Close()
	}
	const attempts = 8
	for attempt := 1; ; attempt++ {
		select {
		case <-req.Context().Done():
			return nil, fmt.Errorf("waiting to send: %w", req.Context().Err())
		case <-p.ticks:
		}
		again := req.Clone(req.Context())
		again.Body = io.NopCloser(bytes.NewReader(body))
		resp, err := http.DefaultTransport.RoundTrip(again)
		busy := err == nil && (resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500)
		if (err == nil && !busy) || attempt == attempts {
			return resp, err //nolint:wrapcheck // a transport passes the error through
		}
		if resp != nil {
			_ = resp.Body.Close()
		}
		select {
		case <-req.Context().Done():
			return nil, fmt.Errorf("waiting to send: %w", req.Context().Err())
		case <-time.After(time.Duration(attempt) * 3 * time.Second):
		}
	}
}
