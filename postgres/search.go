// The read path: one call to mempher.recall, which runs the channels, fuses
// them, reads L1 and spends the budget where the data is.

package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/gojargo/mempher"
)

const recallSQL = `SELECT mempher.recall(
	p_scope := $1, p_query := $2, p_embedding := $3::%s, p_model := $4, p_extractor := $5,
	p_channels := $6, p_limit := $7, p_candidates := $8, p_fact_limit := $9,
	p_max_tokens := $10, p_as_of := $11, p_occurred_from := $12, p_occurred_to := $13,
	p_binding := $14::jsonb, p_roles := $15, p_subjects := $16,
	p_min_fact_confidence := $17, p_weights := $18::jsonb, p_fusion_k := $19,
	p_bytes_per_token := $20)`

// recollectionJSON is the document mempher.recall returns.
type recollectionJSON struct {
	Facts []struct {
		ID         uuid.UUID   `json:"id"`
		Scope      string      `json:"scope"`
		Extractor  string      `json:"extractor"`
		Subject    string      `json:"subject"`
		Predicate  string      `json:"predicate"`
		Object     string      `json:"object"`
		Statement  string      `json:"statement"`
		ValidFrom  time.Time   `json:"valid_from"`
		ValidTo    *time.Time  `json:"valid_to"`
		Confidence float32     `json:"confidence"`
		Episodes   []uuid.UUID `json:"episodes"`
		AssertedAt time.Time   `json:"asserted_at"`
		UpdatedAt  time.Time   `json:"updated_at"`
		Tokens     int         `json:"tokens"`
	} `json:"facts"`
	Episodes []struct {
		ID         uuid.UUID         `json:"id"`
		Scope      string            `json:"scope"`
		Seq        int64             `json:"seq"`
		Content    string            `json:"content"`
		Role       string            `json:"role"`
		Actor      string            `json:"actor"`
		Source     string            `json:"source"`
		OccurredAt time.Time         `json:"occurred_at"`
		IngestedAt time.Time         `json:"ingested_at"`
		Binding    map[string]string `json:"binding"`
		Score      float64           `json:"score"`
		Hits       []struct {
			Channel string  `json:"channel"`
			Rank    int     `json:"rank"`
			Score   float64 `json:"score"`
		} `json:"hits"`
		Tokens int `json:"tokens"`
	} `json:"episodes"`
	Channels []struct {
		Channel    string `json:"channel"`
		Candidates int    `json:"candidates"`
		DurationUS int64  `json:"duration_us"`
	} `json:"channels"`
	Tokens    int  `json:"tokens"`
	Truncated bool `json:"truncated"`
}

// Recall runs q in one call to mempher.recall.
func (s *Store) Recall(ctx context.Context, q mempher.RecallQuery) (mempher.Recollection, error) {
	if q.AsOf.IsZero() {
		return mempher.Recollection{}, fmt.Errorf(
			"mempher/postgres: recall: AsOf must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	}
	if q.Vector != nil {
		if got := q.Vector.Dimensions(); got != s.vectorDimensions {
			return mempher.Recollection{}, fmt.Errorf(
				"mempher/postgres: recall: query vector has %d dimensions but the schema was migrated for %d: %w",
				got, s.vectorDimensions, mempher.ErrDimensionMismatch)
		}
	}
	doc, err := s.recall(ctx, q)
	if err != nil {
		return mempher.Recollection{}, apiError(fmt.Sprintf("recall in scope %q", q.Scope), err)
	}
	out, err := doc.recollection(q)
	if err != nil {
		return mempher.Recollection{}, fmt.Errorf("mempher/postgres: recall in scope %q: %w", q.Scope, err)
	}
	return out, nil
}

// recall makes the call and decodes the document, leaving errors unwrapped for
// the caller to name.
func (s *Store) recall(ctx context.Context, q mempher.RecallQuery) (recollectionJSON, error) {
	var vector, model, extractor *string
	if q.Vector != nil {
		formatted, name := formatVector(q.Vector), string(q.Model)
		vector, model = &formatted, &name
	}
	if q.Extractor != "" {
		name := string(q.Extractor)
		extractor = &name
	}
	var text *string
	if strings.TrimSpace(q.Text) != "" {
		text = &q.Text
	}
	channels := make([]string, len(q.Channels))
	for i, channel := range q.Channels {
		channels[i] = string(channel)
	}
	var maxTokens *int32
	if q.MaxTokens > 0 {
		budget := int32(q.MaxTokens)
		maxTokens = &budget
	}
	perToken := q.BytesPerToken
	if perToken <= 0 {
		perToken = mempher.DefaultBytesPerToken
	}
	var binding []byte
	if len(q.Binding) > 0 {
		encoded, err := json.Marshal(q.Binding)
		if err != nil {
			return recollectionJSON{}, fmt.Errorf("encode binding filter: %w", err)
		}
		binding = encoded
	}
	var roles []string
	if len(q.Roles) > 0 {
		roles = make([]string, len(q.Roles))
		for i, role := range q.Roles {
			roles[i] = string(role)
		}
	}
	var subjects []string
	if len(q.Subjects) > 0 {
		subjects = make([]string, len(q.Subjects))
		for i, subject := range q.Subjects {
			subjects[i] = string(subject)
		}
	}
	weights, err := json.Marshal(q.Fusion.Weights)
	if err != nil {
		return recollectionJSON{}, fmt.Errorf("encode fusion weights: %w", err)
	}
	k := q.Fusion.K
	if k == 0 {
		k = mempher.DefaultFusionK
	}
	limit := q.Limit
	if limit <= 0 {
		limit = mempher.DefaultRecallLimit
	}
	candidates := max(q.CandidatesPerChannel, limit)
	if q.CandidatesPerChannel <= 0 {
		candidates = max(mempher.DefaultCandidatesPerChannel, limit)
	}
	factLimit := q.FactLimit
	if factLimit <= 0 {
		factLimit = mempher.DefaultFactLimit
	}

	var raw []byte
	err = s.pool.QueryRow(ctx, fmt.Sprintf(recallSQL, s.vectorType),
		string(q.Scope), text, vector, model, extractor,
		channels, int32(limit), int32(candidates), int32(factLimit),
		maxTokens, q.AsOf, optionalTime(q.OccurredFrom), optionalTime(q.OccurredTo),
		binding, roles, subjects,
		q.MinFactConfidence, weights, k, int32(perToken),
	).Scan(&raw)
	if err != nil {
		return recollectionJSON{}, err //nolint:wrapcheck // Recall wraps it through apiError
	}
	var doc recollectionJSON
	if err := json.Unmarshal(raw, &doc); err != nil {
		return recollectionJSON{}, fmt.Errorf("decode mempher.recall's result: %w", err)
	}
	return doc, nil
}

// recollection converts the document into the package's types.
func (doc recollectionJSON) recollection(q mempher.RecallQuery) (mempher.Recollection, error) {
	out := mempher.Recollection{
		Scope:     q.Scope,
		AsOf:      q.AsOf,
		Tokens:    doc.Tokens,
		Truncated: doc.Truncated,
		Facts:     make([]mempher.RecalledFact, 0, len(doc.Facts)),
		Episodes:  make([]mempher.RecalledEpisode, 0, len(doc.Episodes)),
		Channels:  make([]mempher.ChannelReport, 0, len(doc.Channels)),
	}
	for _, f := range doc.Facts {
		fact := mempher.Fact{
			ID:         mempher.FactID(f.ID),
			Scope:      mempher.ScopeID(f.Scope),
			Extractor:  mempher.ExtractorID(f.Extractor),
			Subject:    mempher.Subject(f.Subject),
			Predicate:  mempher.Predicate(f.Predicate),
			Object:     f.Object,
			Statement:  f.Statement,
			Valid:      mempher.Validity{From: f.ValidFrom.UTC()},
			Confidence: f.Confidence,
			Episodes:   make([]mempher.EpisodeID, len(f.Episodes)),
			AssertedAt: f.AssertedAt.UTC(),
			UpdatedAt:  f.UpdatedAt.UTC(),
		}
		if f.ValidTo != nil {
			fact.Valid.To = f.ValidTo.UTC()
		}
		for i, id := range f.Episodes {
			fact.Episodes[i] = mempher.EpisodeID(id)
		}
		out.Facts = append(out.Facts, mempher.RecalledFact{Fact: fact, Tokens: f.Tokens})
	}
	for _, e := range doc.Episodes {
		episode := mempher.Episode{
			ID:         mempher.EpisodeID(e.ID),
			Scope:      mempher.ScopeID(e.Scope),
			Seq:        e.Seq,
			Content:    e.Content,
			Role:       mempher.Role(e.Role),
			Actor:      mempher.Actor(e.Actor),
			Source:     mempher.Source(e.Source),
			OccurredAt: e.OccurredAt.UTC(),
			IngestedAt: e.IngestedAt.UTC(),
		}
		if !episode.Role.Valid() {
			return mempher.Recollection{}, fmt.Errorf("episode %s has role %q: %w",
				episode.ID, e.Role, mempher.ErrInvalidRole)
		}
		if len(e.Binding) > 0 {
			episode.Binding = e.Binding
		}
		hits := make([]mempher.ChannelHit, len(e.Hits))
		for i, h := range e.Hits {
			hits[i] = mempher.ChannelHit{Channel: mempher.Channel(h.Channel), Rank: h.Rank, Score: h.Score}
		}
		out.Episodes = append(out.Episodes, mempher.RecalledEpisode{
			Episode: episode, Score: e.Score, Hits: hits, Tokens: e.Tokens,
		})
	}
	for _, c := range doc.Channels {
		out.Channels = append(out.Channels, mempher.ChannelReport{
			Channel:    mempher.Channel(c.Channel),
			Candidates: c.Candidates,
			Duration:   time.Duration(c.DurationUS) * time.Microsecond,
		})
	}
	return out, nil
}

// optionalTime is NULL for the zero time, which means "unbounded" to the API.
func optionalTime(t time.Time) *time.Time {
	if t.IsZero() {
		return nil
	}
	return &t
}

// SearchSemantic returns the nearest episodes by cosine distance over one model's
// encodings, best first. It is mempher.recall with only the semantic channel,
// so it exercises exactly the query Recall runs.
func (s *Store) SearchSemantic(
	ctx context.Context,
	q mempher.SemanticQuery,
) ([]mempher.Candidate, error) {
	if err := s.validateSemantic(q); err != nil {
		return nil, err
	}
	return s.searchOne(ctx, mempher.ChannelSemantic, mempher.RecallQuery{
		Vector: q.Vector, Model: q.Model,
	}, q.ChannelQuery)
}

// SearchLexical returns the best full-text matches, best first. It is
// mempher.recall with only the lexical channel.
func (s *Store) SearchLexical(
	ctx context.Context,
	q mempher.LexicalQuery,
) ([]mempher.Candidate, error) {
	if err := validateLexical(q); err != nil {
		return nil, err
	}
	return s.searchOne(ctx, mempher.ChannelLexical, mempher.RecallQuery{Text: q.Text}, q.ChannelQuery)
}

// searchOne runs one channel alone, weighted 1 so its fused order is its own
// rank order, and returns its hits as candidates.
func (s *Store) searchOne(
	ctx context.Context,
	channel mempher.Channel,
	q mempher.RecallQuery,
	filters mempher.ChannelQuery,
) ([]mempher.Candidate, error) {
	limit := candidateLimit(filters.Limit)
	q.Scope = filters.Scope
	q.Channels = []mempher.Channel{channel}
	q.AsOf = filters.AsOf
	q.OccurredFrom, q.OccurredTo = filters.OccurredFrom, filters.OccurredTo
	q.Binding, q.Roles = filters.Binding, filters.Roles
	q.Limit, q.CandidatesPerChannel = limit, limit
	q.Fusion = mempher.FusionOptions{
		K:       mempher.DefaultFusionK,
		Weights: map[mempher.Channel]float64{channel: 1},
	}
	doc, err := s.recall(ctx, q)
	if err != nil {
		return nil, apiError(fmt.Sprintf("search %s in scope %q", channel, filters.Scope), err)
	}
	out, err := doc.recollection(q)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: search %s in scope %q: %w", channel, filters.Scope, err)
	}
	candidates := make([]mempher.Candidate, len(out.Episodes))
	for i, recalled := range out.Episodes {
		hit := recalled.Hits[0]
		candidates[i] = mempher.Candidate{
			Episode: recalled.Episode, Channel: channel, Rank: hit.Rank, Score: hit.Score,
		}
	}
	return candidates, nil
}

func candidateLimit(limit int) int {
	if limit <= 0 {
		return mempher.DefaultCandidatesPerChannel
	}
	return limit
}

// queryBuilder numbers bind parameters as the SQL is assembled, so a filter can
// be added without renumbering everything after it.
type queryBuilder struct {
	args []any
}

// arg records a value and returns its placeholder.
func (b *queryBuilder) arg(v any) string {
	b.args = append(b.args, v)
	return "$" + strconv.Itoa(len(b.args))
}

// validateSemantic rejects a query that cannot be answered.
func (s *Store) validateSemantic(q mempher.SemanticQuery) error {
	if err := validateChannelQuery(q.ChannelQuery); err != nil {
		return fmt.Errorf("mempher/postgres: search semantic: %w", err)
	}
	if q.Model == "" {
		return fmt.Errorf("mempher/postgres: search semantic: model is empty: %w",
			mempher.ErrInvalidConfig)
	}
	if got := q.Vector.Dimensions(); got != s.vectorDimensions {
		return fmt.Errorf(
			"mempher/postgres: search semantic: query vector has %d dimensions but the schema was migrated for %d: %w",
			got, s.vectorDimensions, mempher.ErrDimensionMismatch)
	}
	return nil
}

// validateLexical rejects a query that cannot be answered.
func validateLexical(q mempher.LexicalQuery) error {
	if err := validateChannelQuery(q.ChannelQuery); err != nil {
		return fmt.Errorf("mempher/postgres: search lexical: %w", err)
	}
	if strings.TrimSpace(q.Text) == "" {
		return fmt.Errorf("mempher/postgres: search lexical: text is empty: %w",
			mempher.ErrInvalidContent)
	}
	return nil
}

// validateChannelQuery checks what both channels need.
func validateChannelQuery(q mempher.ChannelQuery) error {
	if err := q.Scope.Validate(); err != nil {
		return fmt.Errorf("scope: %w", err)
	}
	// A zero AsOf would silently match nothing, since every episode was
	// ingested after the zero time. Recall resolves it from the clock; a Store
	// insists on having been told.
	if q.AsOf.IsZero() {
		return fmt.Errorf("AsOf must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	}
	if q.Limit < 0 {
		return fmt.Errorf("limit is %d: %w", q.Limit, mempher.ErrInvalidConfig)
	}
	if !q.OccurredFrom.IsZero() && !q.OccurredTo.IsZero() &&
		q.OccurredTo.Before(q.OccurredFrom) {
		return fmt.Errorf("event-time window ends before it starts: %w", mempher.ErrInvalidConfig)
	}
	for _, role := range q.Roles {
		if !role.Valid() {
			return fmt.Errorf("role filter %q: %w", role, mempher.ErrInvalidRole)
		}
	}
	if err := q.Binding.Validate(); err != nil {
		return fmt.Errorf("binding filter: %w", err)
	}
	return nil
}
