package postgres_test

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The SQL API is called directly here, as a client in another language would
// call it. The Go adapter's tests cover the same functions through Store; these
// pin what a SQL caller sees: the SQLSTATE, which is the contract.

// wantSQLState fails unless err is a PostgreSQL error with the given code.
func wantSQLState(t *testing.T, err error, code string) {
	t.Helper()
	var pgErr *pgconn.PgError
	switch {
	case err == nil:
		t.Fatalf("succeeded, want SQLSTATE %s", code)
	case !errors.As(err, &pgErr):
		t.Fatalf("err = %v, want SQLSTATE %s", err, code)
	case pgErr.Code != code:
		t.Fatalf("SQLSTATE %s (%s), want %s", pgErr.Code, pgErr.Message, code)
	}
}

// exec runs one statement and returns its error, discarding any rows.
func exec(t *testing.T, pool *pgxpool.Pool, sql string, args ...any) error {
	t.Helper()
	_, err := pool.Exec(t.Context(), sql, args...)
	return err //nolint:wrapcheck // the test inspects the driver's error itself
}

func TestSQLAppend(t *testing.T) {
	t.Parallel()
	_, pool := migrated(t)

	t.Run("a minimal episode needs only content, role and source", func(t *testing.T) {
		var seq int64
		err := pool.QueryRow(t.Context(),
			`SELECT seq FROM mempher.append('sql:min', '[{"content":"hi","role":"user","source":"psql"}]')`).
			Scan(&seq)
		if err != nil {
			t.Fatalf("append: %v", err)
		}
		if seq != 1 {
			t.Errorf("seq = %d, want 1", seq)
		}
		var actor string
		var binding map[string]string
		var kinds []string
		err = pool.QueryRow(t.Context(), `
			SELECT e.actor, e.binding, (SELECT array_agg(kind ORDER BY kind) FROM mempher.jobs WHERE scope_id = e.scope_id)
			FROM mempher.episodes e WHERE e.scope_id = 'sql:min'`).Scan(&actor, &binding, &kinds)
		switch {
		case err != nil:
			t.Fatalf("read back: %v", err)
		case actor != "" || len(binding) != 0:
			t.Errorf("actor = %q, binding = %v, want the empty defaults", actor, binding)
		case len(kinds) != 1 || kinds[0] != "encode":
			t.Errorf("jobs = %v, want the default encode job", kinds)
		}
	})

	t.Run("a batch takes consecutive seqs and one job per kind", func(t *testing.T) {
		rows, err := pool.Query(t.Context(), `
			SELECT seq FROM mempher.append('sql:batch',
				'[{"content":"a","role":"user","source":"s"},{"content":"b","role":"assistant","source":"s"}]',
				'2026-01-01T00:00:00Z', ARRAY['encode','extract','encode'])`)
		if err != nil {
			t.Fatalf("append: %v", err)
		}
		var seqs []int64
		for rows.Next() {
			var seq int64
			if err := rows.Scan(&seq); err != nil {
				t.Fatalf("scan: %v", err)
			}
			seqs = append(seqs, seq)
		}
		if err := rows.Err(); err != nil {
			t.Fatalf("rows: %v", err)
		}
		if len(seqs) != 2 || seqs[0] != 1 || seqs[1] != 2 {
			t.Errorf("seqs = %v, want [1 2]", seqs)
		}
		var jobs, episodes int
		err = pool.QueryRow(t.Context(), `
			SELECT count(*), max(cardinality(episode_ids)) FROM mempher.jobs WHERE scope_id = 'sql:batch'`).
			Scan(&jobs, &episodes)
		if err != nil {
			t.Fatalf("count jobs: %v", err)
		}
		if jobs != 2 || episodes != 2 {
			t.Errorf("%d jobs of up to %d episodes, want 2 jobs naming both episodes", jobs, episodes)
		}
	})

	invalid := []struct {
		name     string
		scope    string
		episodes string
		code     string
	}{
		{"an empty scope", "", `[{"content":"x","role":"user","source":"s"}]`, "MP001"},
		{"a scope with a newline", "a\nb", `[{"content":"x","role":"user","source":"s"}]`, "MP001"},
		{"empty content", "s", `[{"content":"","role":"user","source":"s"}]`, "MP002"},
		{"content that is not a string", "s", `[{"content":1,"role":"user","source":"s"}]`, "MP002"},
		{"an unknown role", "s", `[{"content":"x","role":"bot","source":"s"}]`, "MP003"},
		{"no source", "s", `[{"content":"x","role":"user"}]`, "MP004"},
		{"an actor that is too long", "s", `[{"content":"x","role":"user","source":"s","actor":"` + strings.Repeat("a", 201) + `"}]`, "MP005"},
		{"a nested binding", "s", `[{"content":"x","role":"user","source":"s","binding":{"a":["b"]}}]`, "MP006"},
		{"a binding that is not an object", "s", `[{"content":"x","role":"user","source":"s","binding":"a"}]`, "MP006"},
		{"a key in the wrong case", "s", `[{"content":"x","role":"user","source":"s","occurredAt":"2026-01-01"}]`, "MP013"},
		{"an unparseable occurred_at", "s", `[{"content":"x","role":"user","source":"s","occurred_at":"yesterday-ish"}]`, "MP013"},
		{"no episodes", "s", `[]`, "MP013"},
		{"an object instead of an array", "s", `{"content":"x","role":"user","source":"s"}`, "MP013"},
	}
	for _, tc := range invalid {
		t.Run("rejects "+tc.name, func(t *testing.T) {
			err := exec(t, pool, `SELECT mempher.append($1, $2::jsonb)`, tc.scope, tc.episodes)
			wantSQLState(t, err, tc.code)
		})
	}

	t.Run("rejects an unknown job kind", func(t *testing.T) {
		err := exec(t, pool, `SELECT mempher.append('s', '[{"content":"x","role":"user","source":"s"}]', now(), ARRAY['summarise'])`)
		wantSQLState(t, err, "MP008")
	})

	t.Run("rejects a batch of 65", func(t *testing.T) {
		err := exec(t, pool, `
			SELECT mempher.append('s', (SELECT jsonb_agg(jsonb_build_object('content','x','role','user','source','s'))
			                            FROM generate_series(1, 65)))`)
		wantSQLState(t, err, "MP013")
	})

	t.Run("a rejected batch writes nothing, not even the scope", func(t *testing.T) {
		_ = exec(t, pool, `SELECT mempher.append('sql:atomic', '[{"content":"ok","role":"user","source":"s"},{"content":"","role":"user","source":"s"}]')`)
		var scopes int
		if err := pool.QueryRow(t.Context(), `SELECT count(*) FROM mempher.scopes WHERE id = 'sql:atomic'`).Scan(&scopes); err != nil {
			t.Fatalf("count: %v", err)
		}
		if scopes != 0 {
			t.Error("the scope exists after a rejected append")
		}
	})
}

func TestSQLQueue(t *testing.T) {
	t.Parallel()
	_, pool := migrated(t)
	ctx := t.Context()

	// One scope with one episode and its encode job, at a fixed instant.
	if err := exec(t, pool, `
		SELECT mempher.append('q', '[{"content":"x","role":"user","source":"s"}]',
		                      '2026-01-01T00:00:00Z', ARRAY['encode'], 3)`); err != nil {
		t.Fatalf("append: %v", err)
	}
	lease := func(t *testing.T, worker, at string) (string, int) {
		t.Helper()
		var id string
		var attempts int
		err := pool.QueryRow(ctx,
			`SELECT id::text, attempts FROM mempher.lease_jobs($1, 1, NULL, '1 minute', $2::timestamptz)`,
			worker, at).Scan(&id, &attempts)
		if err != nil {
			t.Fatalf("lease_jobs at %s: %v", at, err)
		}
		return id, attempts
	}

	t.Run("a failure backs off, doubling, and the last attempt is dead", func(t *testing.T) {
		at := "2026-01-01T00:00:00Z"
		wantDelay := []string{"00:00:10", "00:00:20"}
		for attempt := 1; attempt <= 3; attempt++ {
			id, got := lease(t, "w1", at)
			if got != attempt {
				t.Fatalf("attempts = %d at lease %d", got, attempt)
			}
			var state, delay string
			err := pool.QueryRow(ctx, `
				SELECT state, (run_after - $2::timestamptz)::text
				FROM mempher.fail_job($1::uuid, 'w1', 'boom', $2::timestamptz)`, id, at).
				Scan(&state, &delay)
			if err != nil {
				t.Fatalf("fail_job: %v", err)
			}
			if attempt < 3 {
				if state != "pending" || delay != wantDelay[attempt-1] {
					t.Fatalf("attempt %d: state %s, retry after %s, want pending after %s",
						attempt, state, delay, wantDelay[attempt-1])
				}
				err = pool.QueryRow(ctx, `SELECT (run_after)::text FROM mempher.jobs WHERE id = $1::uuid`, id).Scan(&at)
				if err != nil {
					t.Fatalf("read run_after: %v", err)
				}
				continue
			}
			if state != "dead" {
				t.Fatalf("after the last attempt, state = %s, want dead", state)
			}
		}
	})

	t.Run("a dead job is revived once, and not twice", func(t *testing.T) {
		var id string
		if err := pool.QueryRow(ctx, `SELECT id::text FROM mempher.jobs WHERE scope_id = 'q'`).Scan(&id); err != nil {
			t.Fatalf("find job: %v", err)
		}
		var attempts int
		if err := pool.QueryRow(ctx, `SELECT attempts FROM mempher.retry_job($1::uuid, '2026-01-02')`, id).Scan(&attempts); err != nil {
			t.Fatalf("retry_job: %v", err)
		}
		if attempts != 0 {
			t.Errorf("attempts = %d after retry, want 0", attempts)
		}
		wantSQLState(t, exec(t, pool, `SELECT mempher.retry_job($1::uuid, '2026-01-02')`, id), "MP024")
		wantSQLState(t, exec(t, pool, `SELECT mempher.retry_job(uuidv7(), '2026-01-02')`), "MP020")
	})

	t.Run("only the worker holding a lease may finish it", func(t *testing.T) {
		id, _ := lease(t, "w1", "2026-01-02T00:00:00Z")
		wantSQLState(t, exec(t, pool, `SELECT mempher.succeed_job($1::uuid, 'w2', '2026-01-02')`, id), "MP023")
		wantSQLState(t, exec(t, pool, `SELECT mempher.fail_job($1::uuid, 'w2', 'x', '2026-01-02')`, id), "MP023")
		wantSQLState(t, exec(t, pool, `SELECT mempher.succeed_job(uuidv7(), 'w1', '2026-01-02')`), "MP020")

		var released int
		if err := pool.QueryRow(ctx, `SELECT mempher.reclaim_jobs('2026-01-02T00:05:00Z')`).Scan(&released); err != nil {
			t.Fatalf("reclaim_jobs: %v", err)
		}
		if released != 1 {
			t.Errorf("reclaimed %d, want the expired lease", released)
		}
		wantSQLState(t, exec(t, pool, `SELECT mempher.succeed_job($1::uuid, 'w1', '2026-01-02T00:06:00Z')`, id), "MP023")
	})

	t.Run("a long error is kept, truncated", func(t *testing.T) {
		id, _ := lease(t, "w3", "2026-01-03T00:00:00Z")
		var kept int
		err := pool.QueryRow(ctx,
			`SELECT length(last_error) FROM mempher.fail_job($1::uuid, 'w3', repeat('e', 5000), '2026-01-03')`, id).
			Scan(&kept)
		if err != nil {
			t.Fatalf("fail_job: %v", err)
		}
		if want := 2000 + len("... (truncated)"); kept != want {
			t.Errorf("kept %d characters, want %d", kept, want)
		}
	})

	t.Run("purge refuses work still owed", func(t *testing.T) {
		wantSQLState(t, exec(t, pool, `SELECT mempher.purge_jobs(now(), ARRAY['pending'])`), "MP009")
		wantSQLState(t, exec(t, pool, `SELECT mempher.purge_jobs(NULL)`), "MP013")
	})

	t.Run("enqueue needs a scope that exists", func(t *testing.T) {
		wantSQLState(t, exec(t, pool, `SELECT mempher.enqueue('encode', 'nobody', ARRAY[uuidv7()])`), "MP020")
		wantSQLState(t, exec(t, pool, `SELECT mempher.enqueue('summarise', 'q', ARRAY[uuidv7()])`), "MP008")
		wantSQLState(t, exec(t, pool, `SELECT mempher.enqueue('encode', 'q', '{}')`), "MP013")
	})

	t.Run("enqueueing outstanding work returns the job already holding it", func(t *testing.T) {
		var first, second string
		if err := pool.QueryRow(ctx, `SELECT id::text FROM mempher.enqueue('extract', 'q', ARRAY['01900000-0000-7000-8000-000000000001'::uuid])`).Scan(&first); err != nil {
			t.Fatalf("enqueue: %v", err)
		}
		if err := pool.QueryRow(ctx, `SELECT id::text FROM mempher.enqueue('extract', 'q', ARRAY['01900000-0000-7000-8000-000000000001'::uuid])`).Scan(&second); err != nil {
			t.Fatalf("enqueue again: %v", err)
		}
		if first != second {
			t.Errorf("second enqueue made job %s, want the outstanding %s", second, first)
		}
	})
}

func TestSQLProjections(t *testing.T) {
	t.Parallel()
	_, pool := migrated(t)
	ctx := t.Context()

	var episode string
	if err := pool.QueryRow(ctx, `
		SELECT id::text FROM mempher.append('p', '[{"content":"I moved to Paris.","role":"user","source":"s"}]',
		                                    '2026-01-01T00:00:00Z')`).Scan(&episode); err != nil {
		t.Fatalf("append: %v", err)
	}
	apply := func(assert, retract string) error {
		return exec(t, pool,
			`SELECT count(*) FROM mempher.apply_extraction('p', 'x/v1', ARRAY[$1::uuid], $2::jsonb, $3::jsonb, '2026-01-02')`,
			episode, assert, retract)
	}
	paris := `[{"subject":"user","predicate":"lives_in","object":"Paris","statement":"The user lives in Paris.","valid_from":"2026-01-01T00:00:00Z"}]`

	t.Run("applying an extraction twice changes nothing the second time", func(t *testing.T) {
		if err := apply(paris, `[]`); err != nil {
			t.Fatalf("first apply: %v", err)
		}
		if err := apply(`[{"subject":"user","predicate":"lives_in","object":"Rome","statement":"x","valid_from":"2026-01-01T00:00:00Z"}]`, `[]`); err != nil {
			t.Fatalf("second apply: %v", err)
		}
		var facts int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM mempher.facts WHERE scope_id = 'p'`).Scan(&facts); err != nil {
			t.Fatalf("count: %v", err)
		}
		if facts != 1 {
			t.Errorf("%d facts, want the one the first application wrote", facts)
		}
	})

	t.Run("search_facts finds what is true at an instant, and not before", func(t *testing.T) {
		var now, before int
		err := pool.QueryRow(ctx, `
			SELECT (SELECT count(*) FROM mempher.search_facts('p', 'x/v1', '2026-06-01', '2026-06-01', 'where do they live')),
			       (SELECT count(*) FROM mempher.search_facts('p', 'x/v1', '2025-06-01', '2026-06-01'))`).
			Scan(&now, &before)
		if err != nil {
			t.Fatalf("search_facts: %v", err)
		}
		if now != 1 || before != 0 {
			t.Errorf("true in mid-2026: %d, in mid-2025: %d; want 1 and 0", now, before)
		}
	})

	t.Run("rejects", func(t *testing.T) {
		cases := []struct {
			name, assert, retract, code string
		}{
			{"a predicate with a space", `[{"subject":"u","predicate":"lives in","object":"o","statement":"s","valid_from":"2026-01-01"}]`, `[]`, "MP010"},
			{"an empty statement", `[{"subject":"u","predicate":"p","object":"o","statement":"  ","valid_from":"2026-01-01"}]`, `[]`, "MP010"},
			{"a confidence above one", `[{"subject":"u","predicate":"p","object":"o","statement":"s","valid_from":"2026-01-01","confidence":2}]`, `[]`, "MP010"},
			{"a window that ends as it starts", `[{"subject":"u","predicate":"p","object":"o","statement":"s","valid_from":"2026-01-01","valid_to":"2026-01-01"}]`, `[]`, "MP011"},
			{"no valid_from", `[{"subject":"u","predicate":"p","object":"o","statement":"s"}]`, `[]`, "MP011"},
			{"an unknown key", `[{"subject":"u","predicate":"p","object":"o","statement":"s","valid_from":"2026-01-01","since":"x"}]`, `[]`, "MP012"},
			{"two overlapping windows for one claim", `[
				{"subject":"u","predicate":"p","object":"o","statement":"s","valid_from":"2026-01-01"},
				{"subject":"u","predicate":"p","object":"o","statement":"s","valid_from":"2026-02-01"}]`, `[]`, "MP022"},
			{"a retraction naming no fact", `[]`, `[{"at":"2026-01-01"}]`, "MP012"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				wantSQLState(t, apply(tc.assert, tc.retract), tc.code)
			})
		}
		wantSQLState(t, exec(t, pool,
			`SELECT mempher.apply_extraction('p', 'x/v1', ARRAY[uuidv7()], '[]', '[]')`), "MP020")
	})

	t.Run("a retraction before the fact began is refused", func(t *testing.T) {
		if err := exec(t, pool, `SELECT mempher.append('p2', '[{"content":"c","role":"user","source":"s"}]', '2026-01-01')`); err != nil {
			t.Fatalf("append: %v", err)
		}
		var fact string
		err := pool.QueryRow(ctx, `
			SELECT f.id::text FROM mempher.apply_extraction('p2', 'x/v1',
				(SELECT array_agg(id) FROM mempher.episodes WHERE scope_id = 'p2'),
				$1::jsonb, '[]', '2026-01-02') f`, paris).Scan(&fact)
		if err != nil {
			t.Fatalf("apply: %v", err)
		}
		if err := exec(t, pool, `SELECT mempher.append('p2', '[{"content":"d","role":"user","source":"s"}]', '2026-01-03')`); err != nil {
			t.Fatalf("append: %v", err)
		}
		wantSQLState(t, exec(t, pool, `
			SELECT mempher.apply_extraction('p2', 'x/v1',
				(SELECT array_agg(id) FROM mempher.episodes WHERE scope_id = 'p2' AND seq = 2),
				'[]', jsonb_build_array(jsonb_build_object('fact', $1::text, 'at', '2025-01-01')), '2026-01-03')`, fact),
			"MP011")
	})

	t.Run("put_encodings checks the width and the episode", func(t *testing.T) {
		wantSQLState(t, exec(t, pool,
			`SELECT mempher.put_encodings('m', ARRAY[$1::uuid], ARRAY['[1,2]'::vector])`, episode), "MP021")
		wantSQLState(t, exec(t, pool,
			`SELECT mempher.put_encodings('m', ARRAY[$1::uuid], '{}'::vector[])`, episode), "MP013")
	})
}

// axes renders a testDimensions-wide vector literal with the given leading
// components and zeros after them.
func axes(components ...float64) string {
	parts := make([]string, testDimensions)
	for i := range parts {
		parts[i] = "0"
		if i < len(components) {
			parts[i] = strconv.FormatFloat(components[i], 'g', -1, 64)
		}
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func TestSQLRecall(t *testing.T) {
	t.Parallel()
	_, pool := migrated(t)
	ctx := t.Context()

	// Three episodes whose ranks are known in both channels. Against the
	// query vector (1,0): "apple" is nearest, "banana apple" next, "cherry"
	// last. Against the word "banana", only the second matches.
	var ids []string
	rows, err := pool.Query(ctx, `
		SELECT id::text FROM mempher.append('r', '[
			{"content":"apple","role":"user","source":"s"},
			{"content":"banana apple","role":"user","source":"s"},
			{"content":"cherry","role":"user","source":"s"}]', '2026-01-01')`)
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatalf("scan: %v", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	if err := exec(t, pool, `SELECT mempher.put_encodings('m', $1::uuid[], ARRAY[$2, $3, $4]::vector[], '2026-01-01')`,
		ids, axes(1, 0), axes(1, 1), axes(0, 1)); err != nil {
		t.Fatalf("put_encodings: %v", err)
	}

	type hit struct {
		Channel string  `json:"channel"`
		Rank    int     `json:"rank"`
		Score   float64 `json:"score"`
	}
	type doc struct {
		Episodes []struct {
			ID      string  `json:"id"`
			Content string  `json:"content"`
			Score   float64 `json:"score"`
			Hits    []hit   `json:"hits"`
		} `json:"episodes"`
		Channels []struct {
			Channel    string `json:"channel"`
			Candidates int    `json:"candidates"`
		} `json:"channels"`
		Truncated bool `json:"truncated"`
	}
	recall := func(t *testing.T, sql string, args ...any) doc {
		t.Helper()
		var raw []byte
		if err := pool.QueryRow(ctx, sql, args...).Scan(&raw); err != nil {
			t.Fatalf("recall: %v", err)
		}
		var d doc
		if err := json.Unmarshal(raw, &d); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return d
	}
	order := func(d doc) []string {
		out := make([]string, len(d.Episodes))
		for i, e := range d.Episodes {
			out[i] = e.Content
		}
		return out
	}

	t.Run("agreement across channels outranks a bare first place", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', 'banana', $1::vector, 'm', p_as_of := '2026-02-01')`, axes(1, 0))
		if got := order(d); !slices.Equal(got, []string{"banana apple", "apple", "cherry"}) {
			t.Fatalf("order = %v", got)
		}
		// Semantic rank 2 plus a quarter-weighted lexical rank 1, at K = 60.
		want := 1.0/62 + 0.25/61
		if got := d.Episodes[0].Score; math.Abs(got-want) > 1e-12 {
			t.Errorf("fused score = %v, want %v", got, want)
		}
		hits := d.Episodes[0].Hits
		if len(hits) != 2 || hits[0].Channel != "lexical" || hits[0].Rank != 1 ||
			hits[1].Channel != "semantic" || hits[1].Rank != 2 {
			t.Errorf("hits = %+v, want lexical rank 1 then semantic rank 2", hits)
		}
		if len(d.Channels) != 2 || d.Channels[0].Candidates != 3 || d.Channels[1].Candidates != 1 {
			t.Errorf("channels = %+v, want semantic with 3 candidates and lexical with 1", d.Channels)
		}
	})

	t.Run("a zero weight scores nothing but is still reported", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', 'banana', $1::vector, 'm', p_as_of := '2026-02-01',
			p_weights := '{"lexical": 0}')`, axes(1, 0))
		if got := order(d); !slices.Equal(got, []string{"apple", "banana apple", "cherry"}) {
			t.Fatalf("order = %v, want the semantic order alone", got)
		}
		if len(d.Episodes[1].Hits) != 2 {
			t.Errorf("hits = %+v, want the lexical hit still reported", d.Episodes[1].Hits)
		}
	})

	t.Run("a smaller K sharpens the advantage of first place", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', 'banana', $1::vector, 'm', p_as_of := '2026-02-01',
			p_fusion_k := 1)`, axes(1, 0))
		if got := order(d); got[0] != "apple" {
			t.Errorf("order = %v, want the semantic first place to win at K = 1", got)
		}
	})

	t.Run("ties break towards the newer episode", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', NULL, $1::vector, 'm', p_as_of := '2026-02-01')`, axes(0, 0, 1))
		// Every episode is orthogonal to (0,0,1), so all score the same.
		if got := order(d); !slices.Equal(got, []string{"cherry", "banana apple", "apple"}) {
			t.Errorf("order = %v, want newest first", got)
		}
	})

	t.Run("the limit cuts, and says so", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', 'apple banana cherry', p_limit := 2, p_as_of := '2026-02-01')`)
		if len(d.Episodes) != 2 || !d.Truncated {
			t.Errorf("%d episodes, truncated %v; want 2 and true", len(d.Episodes), d.Truncated)
		}
	})

	t.Run("as_of hides what was not yet ingested", func(t *testing.T) {
		d := recall(t, `SELECT mempher.recall('r', 'apple', p_as_of := '2025-12-31')`)
		if len(d.Episodes) != 0 {
			t.Errorf("recalled %v before anything was ingested", order(d))
		}
	})

	t.Run("rejects", func(t *testing.T) {
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', NULL)`), "MP002")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_channels := ARRAY['graph'])`), "MP007")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_channels := ARRAY['semantic'])`), "MP013")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_channels := ARRAY['fact'])`), "MP013")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', '[1,2]'::vector, 'm')`), "MP021")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_weights := '{"lexical": -1}')`), "MP013")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_roles := ARRAY['bot'])`), "MP003")
		wantSQLState(t, exec(t, pool, `SELECT mempher.recall('r', 'x', p_occurred_from := '2026-02-01', p_occurred_to := '2026-01-01')`), "MP013")
	})
}

func TestSQLForget(t *testing.T) {
	t.Parallel()
	_, pool := migrated(t)
	ctx := t.Context()

	for _, scope := range []string{"f1", "f2"} {
		if err := exec(t, pool, `SELECT mempher.append($1, '[
			{"content":"a","role":"user","source":"s"},{"content":"b","role":"user","source":"s"}]')`, scope); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	t.Run("the erasure declaration does not outlive the call", func(t *testing.T) {
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatalf("begin: %v", err)
		}
		defer func() { _ = tx.Rollback(ctx) }()
		if _, err := tx.Exec(ctx, `SELECT mempher.forget('f1', ARRAY[(SELECT id FROM mempher.episodes WHERE scope_id = 'f1' AND seq = 1)])`); err != nil {
			t.Fatalf("forget: %v", err)
		}
		_, err = tx.Exec(ctx, `DELETE FROM mempher.episodes WHERE scope_id = 'f2'`)
		var pgErr *pgconn.PgError
		if !errors.As(err, &pgErr) || pgErr.Code != "P0001" {
			t.Fatalf("a bare DELETE after forget in the same transaction: err = %v, want the append-only trigger", err)
		}
	})

	t.Run("pending_episodes lists what L0 still owes", func(t *testing.T) {
		var pending int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM mempher.pending_episodes('encode', 'm')`).Scan(&pending); err != nil {
			t.Fatalf("pending_episodes: %v", err)
		}
		if pending != 4 {
			t.Errorf("%d pending, want all 4 episodes", pending)
		}
		wantSQLState(t, exec(t, pool, `SELECT * FROM mempher.pending_episodes('encode', 'm', NULL, 3)`), "MP013")
	})

	t.Run("a whole scope goes, with its row, and says so", func(t *testing.T) {
		var episodes, jobs int
		var removed bool
		err := pool.QueryRow(ctx, `SELECT episodes, jobs, scope_removed FROM mempher.forget('f2')`).
			Scan(&episodes, &jobs, &removed)
		if err != nil {
			t.Fatalf("forget: %v", err)
		}
		if episodes != 2 || jobs != 1 || !removed {
			t.Errorf("erased %d episodes and %d jobs, scope removed %v; want 2, 1, true", episodes, jobs, removed)
		}
		var again bool
		if err := pool.QueryRow(ctx, `SELECT scope_removed FROM mempher.forget('f2')`).Scan(&again); err != nil {
			t.Fatalf("forget again: %v", err)
		}
		if again {
			t.Error("forgetting an absent scope reported removing it")
		}
	})
}
