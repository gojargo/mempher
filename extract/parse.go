// Reading the model's answer, and refusing the parts of it a FactStore would.

package extract

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gojargo/mempher"
)

// extraction is a reply to the extraction call, in the shape the schema asks
// for.
//
// The slices are pointers so that an extraction which found nothing can be told
// from one that did not answer in this shape at all. Both decode into the same
// zero value otherwise: a model that replies with a bare assertion object rather
// than the envelope around it produces valid JSON with no matching fields, and
// "nothing to record" is the single most common correct answer here -- so
// without this, a model that has misunderstood the schema is indistinguishable
// from one doing its job, on every call, silently.
type extraction struct {
	Assertions  *[]jsonAssertion  `json:"assertions"`
	Retractions *[]jsonRetraction `json:"retractions"`
	// Not part of answered: a reply holding only aliases has still missed
	// the two fields every reply must carry.
	Aliases []jsonAlias `json:"aliases"`
}

// answered reports that the reply was in the shape asked for, whether or not it
// held anything.
func (e extraction) answered() bool { return e.Assertions != nil || e.Retractions != nil }

// topics is a reply to the first pass of [Options.Reconcile].
type topics struct {
	Subjects   *[]string `json:"subjects"`
	Predicates *[]string `json:"predicates"`
}

func (t topics) answered() bool { return t.Subjects != nil || t.Predicates != nil }

type jsonAssertion struct {
	Subject   string `json:"subject"`
	Predicate string `json:"predicate"`
	Object    string `json:"object"`
	Statement string `json:"statement"`
	ValidFrom string `json:"valid_from"`
	ValidTo   string `json:"valid_to"`
	// A pointer, because a missing confidence must not arrive as zero.
	// [mempher.Assertion.Confidence] takes zero at face value -- it means no
	// confidence at all, not "unset" -- so an omission would otherwise record
	// every fact as one the extractor believed worthless.
	Confidence *float64 `json:"confidence"`
}

type jsonRetraction struct {
	FactID string `json:"fact_id"`
	At     string `json:"at"`
}

type jsonAlias struct {
	Alias string `json:"alias"`
	Of    string `json:"of"`
	// A pointer for the reason an assertion's is: an alias with no
	// confidence is one the model did not vouch for, and is dropped.
	Confidence *float64 `json:"confidence"`
}

// decode reads a provider's reply. Unknown fields are tolerated: a provider that
// wraps the answer in a key of its own is something to read around, not a job to
// dead-letter.
func decode[T any](raw []byte) (T, error) {
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		return out, fmt.Errorf("mempher/extract: decode %d bytes of response: %w: %w",
			len(raw), err, ErrMalformedResponse)
	}
	return out, nil
}

// validate turns a decoded reply into what a [mempher.FactStore] will accept.
//
// Every rejection here drops one item and logs why, rather than failing the
// extraction. A model returns a bad triple beside good ones often enough that
// the alternative -- losing the whole turn to one hallucinated predicate, and
// re-losing it on every retry until the job dies -- costs more than it saves.
func (e *Extractor) validate(
	ctx context.Context,
	body extraction,
	req mempher.ExtractRequest,
) mempher.ExtractResult {
	var assertions []jsonAssertion
	if body.Assertions != nil {
		assertions = *body.Assertions
	}
	var retractions []jsonRetraction
	if body.Retractions != nil {
		retractions = *body.Retractions
	}
	retract := e.retractions(ctx, retractions, req)
	return mempher.ExtractResult{
		Assert:  e.unsettled(ctx, e.assertions(ctx, assertions, req.Now), retract, req),
		Retract: retract,
		Alias:   e.aliases(ctx, body.Aliases),
	}
}

// unsettled drops the assertions that restate a claim over a window that
// overlaps without matching: one already known, or one made earlier in the same
// answer.
//
// A model shown "lives in Paris since 2019" and an episode that mentions Paris
// again tends to assert it once more, dated from the mention. A FactStore
// refuses that as a conflict, rightly: it will not guess which window is true.
// But refusing it there fails the whole extraction, every retry of it, and the
// good claims beside it. The prompt already says a restated fact is recorded
// and needs nothing, so the restatement is what goes, and the window the scope
// already believes stands.
//
// It can only see the facts the worker supplied. A conflict with a fact outside
// them still reaches the store and is still an error.
func (e *Extractor) unsettled(
	ctx context.Context,
	in []mempher.Assertion,
	retract []mempher.Retraction,
	req mempher.ExtractRequest,
) []mempher.Assertion {
	if len(in) == 0 {
		return in
	}
	// A fact this same answer closes is compared as it will stand once
	// closed: "moved away in June, and back the next year" asserts a window
	// that overlaps the old one only until the retraction is applied.
	closed := make(map[mempher.FactID]time.Time, len(retract))
	for _, r := range retract {
		closed[r.Fact] = r.At
	}
	// Any name an entity goes by is that entity, and the known facts carry
	// its own name.
	names := map[string]string{}
	for _, entity := range req.Entities {
		for _, alias := range entity.Aliases {
			names[nameKey(string(alias))] = nameKey(string(entity.Name))
		}
	}
	subject := func(s mempher.Subject) string {
		key := nameKey(string(s))
		if name, ok := names[key]; ok {
			return name
		}
		return key
	}
	type claim struct{ subject, predicate, object string }
	known := make(map[claim][]mempher.Validity, len(req.Known))
	for _, f := range req.Known {
		window := f.Valid
		if at, ok := closed[f.ID]; ok && (window.IsOpen() || at.Before(window.To)) {
			window.To = at
		}
		c := claim{subject(f.Subject), string(f.Predicate), f.Object}
		known[c] = append(known[c], window)
	}

	clashes := func(windows []mempher.Validity, with mempher.Validity) bool {
		for _, window := range windows {
			same := window.From.Equal(with.From) && window.To.Equal(with.To)
			if !same && window.Overlaps(with) {
				return true
			}
		}
		return false
	}

	// The same holds within one answer: a model that gives one claim two
	// overlapping windows has contradicted itself, the store refuses the
	// whole extraction for it, and the first window it gave is the one kept.
	out := in[:0:0]
	kept := make(map[claim][]mempher.Validity, len(in))
	for _, a := range in {
		c := claim{subject(a.Subject), string(a.Predicate), a.Object}
		switch {
		case clashes(known[c], a.Valid):
			e.log.InfoContext(ctx, "mempher/extract: dropped an assertion that restates a known fact",
				"subject", a.Subject, "predicate", a.Predicate, "object", a.Object)
		case clashes(kept[c], a.Valid):
			e.log.WarnContext(ctx, "mempher/extract: dropped an assertion that contradicts an earlier one",
				"subject", a.Subject, "predicate", a.Predicate, "object", a.Object)
		default:
			kept[c] = append(kept[c], a.Valid)
			out = append(out, a)
		}
	}
	return out
}

// nameKey is the form in which two spellings of a name are one name, as the
// store compares them: case and runs of whitespace aside.
func nameKey(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

// aliases keeps the declarations a FactStore would accept and the model was
// sure of. One that names a thing after itself says nothing and is dropped
// without a log line.
func (e *Extractor) aliases(ctx context.Context, in []jsonAlias) []mempher.Alias {
	if len(in) == 0 {
		return nil
	}
	out := make([]mempher.Alias, 0, len(in))
	for _, raw := range in {
		if len(out) >= mempher.MaxAliasesPerExtraction {
			e.log.WarnContext(ctx, "mempher/extract: aliases over the cap were dropped",
				"cap", mempher.MaxAliasesPerExtraction, "returned", len(in))
			break
		}
		a := mempher.Alias{
			Name: mempher.Subject(strings.TrimSpace(raw.Alias)),
			Of:   mempher.Subject(strings.TrimSpace(raw.Of)),
		}
		if strings.EqualFold(string(a.Name), string(a.Of)) {
			continue
		}
		if raw.Confidence == nil || *raw.Confidence < e.opts.MinAliasConfidence {
			e.log.InfoContext(ctx, "mempher/extract: dropped an alias the model was not sure of",
				"alias", raw.Alias, "of", raw.Of, "threshold", e.opts.MinAliasConfidence)
			continue
		}
		if err := a.Validate(); err != nil {
			e.log.WarnContext(ctx, "mempher/extract: dropped an alias",
				"reason", err, "alias", raw.Alias, "of", raw.Of)
			continue
		}
		out = append(out, a)
	}
	return out
}

func (e *Extractor) assertions(
	ctx context.Context,
	in []jsonAssertion,
	now time.Time,
) []mempher.Assertion {
	if len(in) == 0 {
		return nil
	}
	out := make([]mempher.Assertion, 0, len(in))
	for _, raw := range in {
		if len(out) >= e.opts.MaxAssertions {
			e.log.WarnContext(ctx, "mempher/extract: assertions over the cap were dropped",
				"cap", e.opts.MaxAssertions, "returned", len(in))
			break
		}
		a, err := e.assertion(raw, now)
		if err != nil {
			e.log.WarnContext(ctx, "mempher/extract: dropped an assertion",
				"reason", err,
				"subject", raw.Subject, "predicate", raw.Predicate, "object", raw.Object)
			continue
		}
		out = append(out, a)
	}
	return out
}

// assertion builds one claim and holds it to everything a FactStore would,
// plus the one rule only this package knows: the configured vocabulary.
func (e *Extractor) assertion(raw jsonAssertion, now time.Time) (mempher.Assertion, error) {
	if raw.Confidence == nil {
		return mempher.Assertion{}, fmt.Errorf(
			"no confidence was given: %w", mempher.ErrInvalidFact)
	}
	from, err := instant(raw.ValidFrom, now)
	if err != nil {
		return mempher.Assertion{}, fmt.Errorf("valid_from: %w", err)
	}
	var to time.Time
	if strings.TrimSpace(raw.ValidTo) != "" {
		if to, err = instant(raw.ValidTo, now); err != nil {
			return mempher.Assertion{}, fmt.Errorf("valid_to: %w", err)
		}
	}

	a := mempher.Assertion{
		Subject:    mempher.Subject(strings.TrimSpace(raw.Subject)),
		Predicate:  mempher.Predicate(strings.TrimSpace(raw.Predicate)),
		Object:     strings.TrimSpace(raw.Object),
		Statement:  strings.TrimSpace(raw.Statement),
		Valid:      mempher.Validity{From: from, To: to},
		Confidence: float32(*raw.Confidence),
	}
	if e.predicates != nil {
		if _, ok := e.predicates[a.Predicate]; !ok {
			return mempher.Assertion{}, fmt.Errorf(
				"predicate %q is outside the configured vocabulary: %w",
				a.Predicate, mempher.ErrInvalidFact)
		}
	}
	if err := a.Validate(); err != nil {
		return mempher.Assertion{}, err
	}
	return a, nil
}

func (e *Extractor) retractions(
	ctx context.Context,
	in []jsonRetraction,
	req mempher.ExtractRequest,
) []mempher.Retraction {
	if len(in) == 0 {
		return nil
	}
	// Checked against every fact the worker supplied, not against the narrowed
	// set an Options.Reconcile pass chose: a fact id is either real or invented,
	// and which facts the model was shown does not change which.
	known := make(map[mempher.FactID]mempher.Fact, len(req.Known))
	for _, f := range req.Known {
		known[f.ID] = f
	}

	out := make([]mempher.Retraction, 0, len(in))
	for _, raw := range in {
		if len(out) >= mempher.MaxRetractionsPerExtraction {
			e.log.WarnContext(ctx, "mempher/extract: retractions over the cap were dropped",
				"cap", mempher.MaxRetractionsPerExtraction, "returned", len(in))
			break
		}
		r, err := retraction(raw, known, req.Now)
		if err != nil {
			e.log.WarnContext(ctx, "mempher/extract: dropped a retraction",
				"reason", err, "fact", raw.FactID, "at", raw.At)
			continue
		}
		out = append(out, r)
	}
	return out
}

func retraction(
	raw jsonRetraction,
	known map[mempher.FactID]mempher.Fact,
	now time.Time,
) (mempher.Retraction, error) {
	id, err := mempher.ParseFactID(strings.TrimSpace(raw.FactID))
	if err != nil {
		return mempher.Retraction{}, fmt.Errorf(
			"fact_id %q is not an id: %w", raw.FactID, mempher.ErrInvalidExtraction)
	}
	fact, ok := known[id]
	if !ok {
		// A fact the model was never shown is a fact it invented, and closing a
		// window by a guessed id would close the wrong claim.
		return mempher.Retraction{}, fmt.Errorf(
			"fact %s is not among the known facts: %w", id, mempher.ErrInvalidExtraction)
	}
	at, err := instant(raw.At, now)
	if err != nil {
		return mempher.Retraction{}, fmt.Errorf("at: %w", err)
	}
	if at.Before(fact.Valid.From) {
		// Clamping to Valid.From would invent a window the world never had, and
		// the store would accept it without complaint.
		return mempher.Retraction{}, fmt.Errorf(
			"fact %s would close at %s, before it began at %s: %w",
			id, at.Format(time.RFC3339), fact.Valid.From.Format(time.RFC3339),
			mempher.ErrInvalidValidity)
	}

	r := mempher.Retraction{Fact: id, At: at}
	if err := r.Validate(); err != nil {
		return mempher.Retraction{}, err
	}
	return r, nil
}

// instant parses a date or a timestamp, as the schema asks for either.
//
// A bare date resolves at midnight in now's location. Natural language yields
// dates rather than timestamps, and demanding RFC 3339 would only make the model
// invent a time of day; taking the zone from the injected instant is what keeps
// that invention out of the data and out of a replay.
func instant(s string, now time.Time) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, fmt.Errorf("no instant was given: %w", mempher.ErrInvalidValidity)
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	if t, err := time.ParseInLocation(time.DateOnly, s, now.Location()); err == nil {
		return t, nil
	}
	return time.Time{}, fmt.Errorf(
		"%q is neither a date nor an RFC 3339 timestamp: %w", s, mempher.ErrInvalidValidity)
}
