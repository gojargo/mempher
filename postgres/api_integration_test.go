package postgres_test

import (
	"errors"
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
