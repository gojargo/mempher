// Translating the SQL API's error contract back into the package's sentinels.

package postgres

import (
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/gojargo/mempher"
)

// apiErrors maps each SQLSTATE in class MP, which the mempher.* functions raise
// for a caller's mistake, to the sentinel a Go caller tests for. The table is
// the Go half of the contract listed at the top of 0005_api_append.sql, and
// TestAPIErrorCodesAreMapped keeps the two halves in step.
var apiErrors = map[string]error{
	"MP001": mempher.ErrInvalidScope,
	"MP002": mempher.ErrInvalidContent,
	"MP003": mempher.ErrInvalidRole,
	"MP004": mempher.ErrInvalidSource,
	"MP005": mempher.ErrInvalidActor,
	"MP006": mempher.ErrInvalidBinding,
	"MP007": mempher.ErrInvalidChannel,
	"MP008": mempher.ErrInvalidJobKind,
	"MP009": mempher.ErrInvalidJobState,
	"MP010": mempher.ErrInvalidFact,
	"MP011": mempher.ErrInvalidValidity,
	"MP012": mempher.ErrInvalidExtraction,
	"MP013": mempher.ErrInvalidConfig,
	"MP020": mempher.ErrNotFound,
	"MP021": mempher.ErrDimensionMismatch,
	"MP022": mempher.ErrFactConflict,
	"MP023": mempher.ErrJobNotLeased,
	"MP024": mempher.ErrJobNotDead,
	"MP025": mempher.ErrJobOutstanding,
}

// apiError wraps an error from a mempher.* function for op. A caller error in
// class MP becomes its sentinel, keeping the database's message, which already
// names the offending field; a missing schema becomes [ErrSchemaNotReady];
// anything else is passed through wrapped, for errors.As to find.
func apiError(op string, err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		if sentinel, ok := apiErrors[pgErr.Code]; ok {
			return fmt.Errorf("mempher/postgres: %s: %s: %w", op, pgErr.Message, sentinel)
		}
	}
	if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema, pgerrcodeUndefinedFunction) {
		return fmt.Errorf("mempher/postgres: %s: run Migrate first: %w", op, ErrSchemaNotReady)
	}
	return fmt.Errorf("mempher/postgres: %s: %w", op, err)
}
