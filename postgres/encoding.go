// The vector projection: writing encodings, and finding episodes that lack one.

package postgres

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/ops"
)

// DefaultPendingEncodingsLimit is how many ids [Store.PendingEncodings] returns
// when a query does not say.
const DefaultPendingEncodingsLimit = 100

// PutEncodings writes a batch of encodings in one statement, replacing any
// that exist for the same episode and model. The batch shares one Model and one
// EncodedAt, because mempher.put_encodings takes one of each.
func (s *Store) PutEncodings(ctx context.Context, encodings []mempher.Encoding) error {
	if len(encodings) == 0 {
		return nil
	}
	first := encodings[0]
	episodes := make([]uuid.UUID, len(encodings))
	vectors := make([]string, len(encodings))
	surprise := make([]float32, len(encodings))
	for i, enc := range encodings {
		if err := s.validateEncoding(enc); err != nil {
			return err
		}
		if enc.Model != first.Model || !enc.EncodedAt.Equal(first.EncodedAt) {
			return fmt.Errorf(
				"mempher/postgres: put encodings: encoding %d differs in Model or EncodedAt from the first: %w",
				i, mempher.ErrInvalidConfig)
		}
		episodes[i] = uuid.UUID(enc.Episode)
		vectors[i] = formatVector(enc.Vector)
		surprise[i] = enc.Surprise
	}
	err := s.exec(ctx,
		fmt.Sprintf(`SELECT mempher.put_encodings($1, $2, $3::%s[], $4, $5)`, s.vectorType),
		string(first.Model), episodes, vectors, first.EncodedAt, surprise)
	if err != nil {
		return apiError(fmt.Sprintf("put %d encodings", len(encodings)), err)
	}
	return nil
}

// PutEncoding writes one encoding. It is [Store.PutEncodings] for a batch of
// one.
func (s *Store) PutEncoding(ctx context.Context, enc mempher.Encoding) error {
	return s.PutEncodings(ctx, []mempher.Encoding{enc})
}

// validateEncoding rejects what pgvector or a CHECK would reject anyway, with an
// error that names the problem.
func (s *Store) validateEncoding(enc mempher.Encoding) error {
	switch {
	case enc.Episode.IsZero():
		return fmt.Errorf("mempher/postgres: put encoding: episode id is unset: %w",
			mempher.ErrInvalidConfig)
	case enc.Model == "":
		return fmt.Errorf("mempher/postgres: put encoding: model is empty: %w",
			mempher.ErrInvalidConfig)
	case enc.EncodedAt.IsZero():
		return fmt.Errorf(
			"mempher/postgres: put encoding: EncodedAt must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	case enc.Surprise < 0 || enc.Surprise > 1:
		return fmt.Errorf("mempher/postgres: put encoding: surprise is %v, must be in [0,1]: %w",
			enc.Surprise, mempher.ErrInvalidConfig)
	}

	if err := s.validateVector(enc.Vector, enc.Model); err != nil {
		return fmt.Errorf("mempher/postgres: put encoding: %w", err)
	}
	return nil
}

// validateVector checks a vector about to be stored, of an episode or of a
// fact: it has the width the schema was migrated for, and every element is
// finite.
func (s *Store) validateVector(v mempher.Vector, model mempher.ModelID) error {
	if got := v.Dimensions(); got != s.vectorDimensions {
		return fmt.Errorf(
			"vector has %d dimensions but the schema was migrated for %d: %w",
			got, s.vectorDimensions, mempher.ErrDimensionMismatch)
	}
	// pgvector rejects these, but as a parse error on a 20 KB literal, which
	// says nothing about which model produced the bad value.
	for i, f := range v {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return fmt.Errorf("vector element %d from model %q is %v: %w",
				i, model, f, mempher.ErrInvalidConfig)
		}
	}
	return nil
}

// PendingEncodings returns ids of episodes with no encoding for a model, in Seq
// order.
func (s *Store) PendingEncodings(
	ctx context.Context,
	q ops.PendingEncodings,
) ([]mempher.EpisodeID, error) {
	if q.Limit < 0 {
		return nil, fmt.Errorf("mempher/postgres: pending encodings: Limit is %d: %w",
			q.Limit, mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultPendingEncodingsLimit
	}
	var scope *string
	if q.Scope != "" {
		name := string(q.Scope)
		scope = &name
	}
	rows, err := s.pool.Query(ctx,
		`SELECT id FROM mempher.pending_episodes('encode', $1, $2, $3, $4)`,
		string(q.Model), scope, q.AfterSeq, int32(limit))
	if err != nil {
		return nil, apiError("pending encodings", err)
	}
	defer rows.Close()
	out := make([]mempher.EpisodeID, 0, min(limit, 128))
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending encodings: scan: %w", err)
		}
		out = append(out, mempher.EpisodeID(id))
	}
	if err := rows.Err(); err != nil {
		return nil, apiError("pending encodings", err)
	}
	return out, nil
}

// formatVector renders a vector in pgvector's text input form.
//
// Text rather than the binary protocol, because binary would need the vector type
// registered on every pooled connection and this library does not own the pool.
// The cost is a few extra bytes per write and nothing in correctness.
func formatVector(v mempher.Vector) string {
	var b strings.Builder
	b.Grow(len(v)*10 + 2)
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		// 32-bit shortest round-trip: the value read back equals the value
		// written, without printing digits float32 does not hold.
		b.WriteString(strconv.FormatFloat(float64(f), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}
