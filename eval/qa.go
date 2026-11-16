// End-to-end question answering: recall, an answer written from what was
// recalled, and a judge.

package eval

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/extract"
)

// QAConfig assembles an end-to-end run.
//
// [Run] asks whether the evidence came back. This asks the question a user of
// an agent asks: was the answer right. It costs two model calls a question, and
// its number moves with the models that write and grade the answer, so it is
// reported with their names and is comparable only to a run that used the same
// ones.
type QAConfig struct {
	// Memory is what gets measured. Required.
	Memory *mempher.Memory
	// Store erases each haystack once its questions are answered. Optional.
	Store mempher.Store
	// Worker drains the queue after each haystack is ingested, which is what
	// builds the vectors and the facts. Optional, as in [Config].
	Worker *mempher.Worker
	// Answerer writes the answer from what recall returned. Required.
	//
	// It is an [extract.Completer] because that is the one structured model
	// call this module already has adapters for.
	Answerer extract.Completer
	// Judge decides whether an answer matches the expected one. Required.
	Judge extract.Completer
	// Variants are the recall configurations to answer from, each summarised
	// separately. Empty means one: whatever [QAConfig.Memory] does by default.
	Variants []Variant
	// Limit is how many episodes recall returns. Zero means [DefaultLimit].
	Limit int
	// FactLimit is how many facts recall returns. Zero means the library's
	// default.
	FactLimit int
	// MaxTokens is the token budget of the recalled context. Zero means none.
	MaxTokens int
	// BoundByQuestionTime applies [Question.Asked] as an event-time upper
	// bound, as in [Config].
	BoundByQuestionTime bool
	// Parallel is how many questions of one haystack are answered at once.
	// Zero means one.
	Parallel int
	// Attempts is how many times a failing model call is tried before the
	// question is recorded as an error. Zero means three.
	Attempts int
	// Haystacks is how many haystacks are ingested and answered at once. Zero
	// means one. Above one, [QAConfig.Outstanding] is required, and every
	// haystack's drain also works the others' jobs, which is how one queue is
	// worked by many workers in a deployment.
	Haystacks int
	// Outstanding reports how many jobs of a scope are still pending or
	// running. A haystack is answered only once it reports none, so that no
	// question is asked of a memory still being built. Required when
	// Haystacks is above one; otherwise a drain that finds the queue empty is
	// taken to have finished.
	Outstanding func(ctx context.Context, scope mempher.ScopeID) (int, error)
	// Progress, when set, is called after each question of the first variant.
	Progress func(done int, last QAResult)

	// style is the dataset's [Dataset.AnswerStyle].
	style string
}

// QAResult is what one question came to under one variant.
type QAResult struct {
	// ID and Type identify the question.
	ID   string
	Type string
	// Expected is the benchmark's answer, and Response the one given.
	Expected string
	Response string
	// Correct is the judge's verdict. Under a rubric it means every item was
	// fully met.
	Correct bool
	// Score is 1 for a correct answer and 0 for a wrong one, or under a
	// rubric the mean of its items' scores.
	Score float64
	// Err is why there is no verdict: a model call that kept failing. Such a
	// question is counted in [QASummary.Errors] and in no accuracy.
	Err error
	// Labelled reports that the benchmark marks the episodes holding the
	// answer, and Hit that recall returned at least one of them.
	Labelled bool
	Hit      bool
	// Facts and Episodes are how much recall returned, and Tokens its counted
	// cost.
	Facts    int
	Episodes int
	Tokens   int
	// Recall is how long retrieval took.
	Recall time.Duration
}

// QATypeSummary is one question type's share of a [QASummary].
type QATypeSummary struct {
	Questions int
	Correct   int
	Accuracy  float64
	// Score is the mean [QAResult.Score].
	Score float64

	score float64
}

// QASummary is an end-to-end run under one variant.
type QASummary struct {
	// Dataset, Variant, Answerer and Judge say what was measured and by what.
	Dataset  string
	Variant  string
	Answerer string
	Judge    string
	// Questions is how many were judged, Correct how many the judge accepted,
	// and Accuracy their ratio.
	Questions int
	Correct   int
	Accuracy  float64
	// Score is the mean [QAResult.Score]. It equals Accuracy unless questions
	// are graded by rubric, where it gives partial credit.
	Score float64
	// Rubric reports that some questions were graded by rubric.
	Rubric bool
	// Errors is how many questions have no verdict.
	Errors int
	// Labelled is how many judged questions mark their evidence, and HitRate
	// the share of those for which recall returned some of it.
	Labelled int
	HitRate  float64
	// CorrectGivenHit and CorrectGivenMiss split accuracy by whether the
	// evidence was recalled. The first is the generator's share of the
	// result; the second is what was answered without the marked evidence,
	// from a fact or from another episode.
	CorrectGivenHit  float64
	CorrectGivenMiss float64
	// MeanFacts, MeanEpisodes and MeanTokens describe the context an answer
	// was written from.
	MeanFacts    float64
	MeanEpisodes float64
	MeanTokens   float64
	// ByType breaks the result down by the benchmark's categories.
	ByType map[string]*QATypeSummary

	hits, hitCorrect, missCorrect int
	facts, episodes, tokens       int
	score                         float64
}

// RunQA answers every question in ds from what recall returns and has each
// answer judged.
//
// Consecutive questions that share a [Question.Scope] are asked of one
// haystack, ingested once.
func RunQA(ctx context.Context, cfg QAConfig, ds Dataset) ([]QASummary, error) {
	switch {
	case cfg.Memory == nil:
		return nil, fmt.Errorf("eval: run qa: Memory is required: %w", mempher.ErrInvalidConfig)
	case cfg.Answerer == nil || cfg.Judge == nil:
		return nil, fmt.Errorf("eval: run qa: Answerer and Judge are required: %w", mempher.ErrInvalidConfig)
	case cfg.Parallel < 0 || cfg.Attempts < 0 || cfg.Haystacks < 0:
		return nil, fmt.Errorf("eval: run qa: Parallel, Attempts and Haystacks must not be negative: %w",
			mempher.ErrInvalidConfig)
	case cfg.Haystacks > 1 && cfg.Outstanding == nil:
		return nil, fmt.Errorf("eval: run qa: Outstanding is required to answer haystacks in parallel: %w",
			mempher.ErrInvalidConfig)
	}
	if cfg.Haystacks == 0 {
		cfg.Haystacks = 1
	}
	cfg.style = ds.AnswerStyle
	if cfg.Limit == 0 {
		cfg.Limit = DefaultLimit
	}
	if cfg.Parallel == 0 {
		cfg.Parallel = 1
	}
	if cfg.Attempts == 0 {
		cfg.Attempts = 3
	}
	if len(cfg.Variants) == 0 {
		cfg.Variants = []Variant{{Name: "default"}}
	}
	for i := range cfg.Variants {
		if cfg.Variants[i].Memory == nil {
			cfg.Variants[i].Memory = cfg.Memory
		}
	}

	summaries := make([]QASummary, len(cfg.Variants))
	for i, variant := range cfg.Variants {
		summaries[i] = QASummary{
			Dataset: ds.Name, Variant: variant.Name,
			Answerer: cfg.Answerer.Model(), Judge: cfg.Judge.Model(),
			ByType: map[string]*QATypeSummary{},
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		mu       sync.Mutex
		done     int
		firstErr error
		rubric   bool
		wg       sync.WaitGroup
		slots    = make(chan struct{}, cfg.Haystacks)
	)
	fail := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if firstErr == nil {
			firstErr = err
			cancel()
		}
	}
	flush := func(group []Question) {
		if len(group) == 0 {
			return
		}
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			return
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			results, err := runQAGroup(ctx, cfg, group)
			if err != nil {
				fail(err)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			for qi := range group {
				for vi := range cfg.Variants {
					summaries[vi].add(results[qi][vi])
				}
				done++
				if cfg.Progress != nil {
					cfg.Progress(done, results[qi][0])
				}
			}
		}()
	}

	var group []Question
	for question, err := range ds.Questions {
		if err != nil {
			fail(err)
			break
		}
		if ctx.Err() != nil {
			break
		}
		if len(question.Rubric) > 0 {
			rubric = true
		}
		if len(group) > 0 && (question.Scope == "" || question.Scope != group[0].Scope) {
			flush(group)
			group = nil
		}
		group = append(group, question)
	}
	flush(group)
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	for i := range summaries {
		summaries[i].Rubric = rubric
		summaries[i].finish()
	}
	return summaries, nil
}

// runQAGroup ingests one haystack and answers the questions asked of it. The
// result is indexed by question, then by variant.
func runQAGroup(ctx context.Context, cfg QAConfig, group []Question) ([][]QAResult, error) {
	name := group[0].Scope
	if name == "" {
		name = group[0].ID
	}
	scope := mempher.ScopeID("eval:" + name)

	if err := ingest(ctx, cfg.Memory, scope, group[0].Haystack); err != nil {
		return nil, fmt.Errorf("eval: %s: ingest: %w", name, err)
	}
	if cfg.Worker != nil {
		if err := drainScope(ctx, cfg, scope); err != nil {
			return nil, fmt.Errorf("eval: %s: drain: %w", name, err)
		}
	}

	results := make([][]QAResult, len(group))
	var (
		wg    sync.WaitGroup
		slots = make(chan struct{}, cfg.Parallel)
	)
	for qi, question := range group {
		results[qi] = make([]QAResult, len(cfg.Variants))
		wg.Add(1)
		slots <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			for vi, variant := range cfg.Variants {
				results[qi][vi] = answerOne(ctx, cfg, scope, question, variant)
			}
		}()
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", name, err)
	}

	if cfg.Store != nil {
		if _, err := cfg.Store.Forget(ctx, mempher.ForgetRequest{Scope: scope}); err != nil {
			return nil, fmt.Errorf("eval: %s: erase the scope: %w", name, err)
		}
	}
	return results, nil
}

// answerOne recalls for one question, has an answer written from the result,
// and has it judged. A failure is recorded on the result rather than returned:
// one question a provider would not answer should not cost a run of thousands.
func answerOne(
	ctx context.Context,
	cfg QAConfig,
	scope mempher.ScopeID,
	q Question,
	variant Variant,
) QAResult {
	out := QAResult{ID: q.ID, Type: q.Type, Expected: q.Answer, Labelled: len(q.Relevant) > 0}

	request := mempher.RecallRequest{
		Scope:     scope,
		Query:     q.Query,
		Channels:  variant.Channels,
		Limit:     cfg.Limit,
		FactLimit: cfg.FactLimit,
		MaxTokens: cfg.MaxTokens,
	}
	if cfg.BoundByQuestionTime {
		request.OccurredTo = q.Asked
	}
	started := time.Now()
	recollection, err := variant.Memory.Recall(ctx, request)
	if err != nil {
		out.Err = fmt.Errorf("recall: %w", err)
		return out
	}
	out.Recall = time.Since(started)
	out.Facts, out.Episodes, out.Tokens = len(recollection.Facts), len(recollection.Episodes), recollection.Tokens
	out.Hit = score(q, recollection, cfg.Limit).Hit

	var answer struct {
		Answer *string `json:"answer"`
	}
	err = complete(ctx, cfg.Answerer, cfg.Attempts, extract.Prompt{
		System: answerSystemFor(cfg.style),
		User:   answerPrompt(q, recollection),
		Schema: answerSchema,
	}, &answer, func() bool { return answer.Answer != nil })
	if err != nil {
		out.Err = fmt.Errorf("answer: %w", err)
		return out
	}
	out.Response = strings.TrimSpace(*answer.Answer)

	if len(q.Rubric) > 0 {
		total := 0.0
		for _, item := range q.Rubric {
			var graded struct {
				Score *float64 `json:"score"`
			}
			err = complete(ctx, cfg.Judge, cfg.Attempts, extract.Prompt{
				System: rubricSystem,
				User:   rubricPrompt(q.Query, item, out.Response),
				Schema: rubricSchema,
			}, &graded, func() bool { return graded.Score != nil && validRubricScore(*graded.Score) })
			if err != nil {
				out.Err = fmt.Errorf("judge: %w", err)
				return out
			}
			total += *graded.Score
		}
		out.Score = total / float64(len(q.Rubric))
		out.Correct = out.Score == 1
		return out
	}

	var verdict struct {
		Correct *bool `json:"correct"`
	}
	err = complete(ctx, cfg.Judge, cfg.Attempts, extract.Prompt{
		System: judgeSystem,
		User:   judgePrompt(q, out.Response),
		Schema: judgeSchema,
	}, &verdict, func() bool { return verdict.Correct != nil })
	if err != nil {
		out.Err = fmt.Errorf("judge: %w", err)
		return out
	}
	out.Correct = *verdict.Correct
	if out.Correct {
		out.Score = 1
	}
	return out
}

// complete makes one structured call, retrying a failure or a reply of the
// wrong shape with a growing pause.
func complete(
	ctx context.Context,
	model extract.Completer,
	attempts int,
	prompt extract.Prompt,
	into any,
	answered func() bool,
) error {
	var last error
	for attempt := range attempts {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return fmt.Errorf("waiting to retry: %w", ctx.Err())
			case <-time.After(time.Duration(attempt*attempt) * 2 * time.Second):
			}
		}
		raw, err := model.Complete(ctx, prompt)
		if err != nil {
			last = err
			continue
		}
		if err := json.Unmarshal(raw, into); err != nil {
			last = fmt.Errorf("decode %d bytes of reply: %w", len(raw), err)
			continue
		}
		if !answered() {
			last = errors.New("the reply is not in the shape asked for")
			continue
		}
		return nil
	}
	return fmt.Errorf("after %d attempts: %w", attempts, last)
}

const answerSystem = `You answer a question from an assistant's memory of past conversations.

The memory is given in two parts. FACTS are claims the assistant has extracted,
each with the period over which it holds. EXCERPTS are messages as they were
written, each with the date it was sent.

Use only the memory. When an excerpt says "yesterday" or "last week", work the
date out from the date of that excerpt, not from today's. When a fact and an
older excerpt disagree, the more recent information is the current one.

%s

Return JSON conforming to the schema you were given, and nothing else.`

// defaultAnswerStyle is the form an answer takes unless [Dataset.AnswerStyle]
// says otherwise.
const defaultAnswerStyle = `Give the shortest answer that is complete: a name, a date, a number, a phrase,
or a sentence or two when the question asks for advice. If the memory does not
hold the answer, say that you do not know.`

// answerSystemFor fills the answer prompt with a dataset's style.
func answerSystemFor(style string) string {
	if style == "" {
		style = defaultAnswerStyle
	}
	return fmt.Sprintf(answerSystem, style)
}

const judgeSystem = `You grade an answer to a question about past conversations.

You are given the question, the expected answer, and a response. Decide whether
the response is correct.

The response is correct if it contains the expected answer or is equivalent to
it. Different wording, more detail, or another date format do not make it wrong.
A response that holds only part of what the expected answer requires is wrong,
and so is one that says it does not know when an answer was expected.

Some questions come with an extra grading rule. Where it is given, it takes
precedence over the paragraph above.

Return JSON conforming to the schema you were given, and nothing else.`

var (
	answerSchema = json.RawMessage(`{"type":"object","additionalProperties":false,` +
		`"required":["answer"],"properties":{"answer":{"type":"string"}}}`)
	judgeSchema = json.RawMessage(`{"type":"object","additionalProperties":false,` +
		`"required":["reason","correct"],"properties":{` +
		`"reason":{"type":"string","description":"One sentence saying why."},` +
		`"correct":{"type":"boolean"}}}`)
)

// answerPrompt lays out what recall returned for a reader: the facts, then the
// episodes in the order they were said, whatever order they were ranked in. A
// conversation read out of order answers "what happened first" wrongly.
func answerPrompt(q Question, rec mempher.Recollection) string {
	var b strings.Builder
	if !q.Asked.IsZero() {
		fmt.Fprintf(&b, "Today is %s.\n\n", q.Asked.Format("Monday 2 January 2006"))
	}

	b.WriteString("FACTS\n\n")
	if len(rec.Facts) == 0 {
		b.WriteString("None.\n")
	}
	for _, f := range rec.Facts {
		window := "since " + f.Fact.Valid.From.Format(time.DateOnly)
		if !f.Fact.Valid.IsOpen() {
			window = "from " + f.Fact.Valid.From.Format(time.DateOnly) +
				" to " + f.Fact.Valid.To.Format(time.DateOnly)
		}
		fmt.Fprintf(&b, "- %s (%s)\n", f.Fact.Statement, window)
	}

	b.WriteString("\nEXCERPTS\n\n")
	if len(rec.Episodes) == 0 {
		b.WriteString("None.\n")
	}
	episodes := slices.Clone(rec.Episodes)
	slices.SortStableFunc(episodes, func(x, y mempher.RecalledEpisode) int {
		return cmp.Or(
			x.Episode.OccurredAt.Compare(y.Episode.OccurredAt),
			cmp.Compare(x.Episode.Seq, y.Episode.Seq))
	})
	for _, e := range episodes {
		fmt.Fprintf(&b, "[%s] %s: %s\n",
			e.Episode.OccurredAt.Format("Mon 2 Jan 2006 15:04"), e.Episode.Role, e.Episode.Content)
	}

	fmt.Fprintf(&b, "\nQUESTION\n\n%s\n", q.Query)
	return b.String()
}

func judgePrompt(q Question, response string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "QUESTION\n\n%s\n\nEXPECTED ANSWER\n\n%s\n\nRESPONSE\n\n%s\n", q.Query, q.Answer, response)
	if q.Grading != "" {
		fmt.Fprintf(&b, "\nGRADING RULE FOR THIS QUESTION\n\n%s\n", q.Grading)
	}
	return b.String()
}

func (s *QASummary) add(r QAResult) {
	if r.Err != nil {
		s.Errors++
		return
	}
	s.Questions++
	s.facts += r.Facts
	s.episodes += r.Episodes
	s.tokens += r.Tokens
	byType := s.ByType[r.Type]
	if byType == nil {
		byType = &QATypeSummary{}
		s.ByType[r.Type] = byType
	}
	byType.Questions++
	s.score += r.Score
	byType.score += r.Score
	if r.Correct {
		s.Correct++
		byType.Correct++
	}
	if !r.Labelled {
		return
	}
	s.Labelled++
	switch {
	case r.Hit:
		s.hits++
		if r.Correct {
			s.hitCorrect++
		}
	case r.Correct:
		s.missCorrect++
	}
}

func ratio(n, of int) float64 {
	if of == 0 {
		return 0
	}
	return float64(n) / float64(of)
}

func (s *QASummary) finish() {
	s.Accuracy = ratio(s.Correct, s.Questions)
	if s.Questions > 0 {
		s.Score = s.score / float64(s.Questions)
	}
	s.HitRate = ratio(s.hits, s.Labelled)
	s.CorrectGivenHit = ratio(s.hitCorrect, s.hits)
	s.CorrectGivenMiss = ratio(s.missCorrect, s.Labelled-s.hits)
	s.MeanFacts = ratio(s.facts, s.Questions)
	s.MeanEpisodes = ratio(s.episodes, s.Questions)
	s.MeanTokens = ratio(s.tokens, s.Questions)
	for _, byType := range s.ByType {
		byType.Accuracy = ratio(byType.Correct, byType.Questions)
		if byType.Questions > 0 {
			byType.Score = byType.score / float64(byType.Questions)
		}
	}
}

// String renders the summary as a table.
func (s QASummary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s, %s\n", s.Dataset, s.Variant)
	fmt.Fprintf(&b, "answers by %s, judged by %s\n", s.Answerer, s.Judge)
	if s.Rubric {
		fmt.Fprintf(&b, "rubric score %.3f over %d questions; fully met %.3f (%d); %d without a verdict\n",
			s.Score, s.Questions, s.Accuracy, s.Correct, s.Errors)
	} else {
		fmt.Fprintf(&b, "accuracy %.3f (%d of %d), %d without a verdict\n",
			s.Accuracy, s.Correct, s.Questions, s.Errors)
	}
	fmt.Fprintf(&b, "evidence recalled for %.3f of %d labelled; accuracy %.3f with it, %.3f without\n",
		s.HitRate, s.Labelled, s.CorrectGivenHit, s.CorrectGivenMiss)
	fmt.Fprintf(&b, "context: %.1f facts, %.1f episodes, %.0f tokens on average\n",
		s.MeanFacts, s.MeanEpisodes, s.MeanTokens)
	types := make([]string, 0, len(s.ByType))
	for name := range s.ByType {
		types = append(types, name)
	}
	slices.Sort(types)
	for _, name := range types {
		byType := s.ByType[name]
		if s.Rubric {
			fmt.Fprintf(&b, "  %-28s %.3f  (%d questions)\n", name, byType.Score, byType.Questions)
		} else {
			fmt.Fprintf(&b, "  %-28s %.3f  (%d of %d)\n", name, byType.Accuracy, byType.Correct, byType.Questions)
		}
	}
	return b.String()
}

// drainPause is how long a haystack waits, when the queue offers nothing, for
// jobs of its scope that another haystack's drain is running or that are
// backing off after a failure.
const drainPause = time.Second

// drainScope works the queue until scope has no job left to wait for. With one
// haystack at a time an empty queue means that; with several, a job of this
// scope may be running under another haystack's drain, so the scope itself is
// asked.
func drainScope(ctx context.Context, cfg QAConfig, scope mempher.ScopeID) error {
	if cfg.Outstanding == nil {
		return drain(ctx, cfg.Worker)
	}
	for {
		if err := drain(ctx, cfg.Worker); err != nil {
			return err
		}
		left, err := cfg.Outstanding(ctx, scope)
		if err != nil {
			return err
		}
		if left == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("eval: wait for %s: %w", scope, ctx.Err())
		case <-time.After(drainPause):
		}
	}
}

// rubricSystem and rubricPrompt grade one rubric item. They are adapted from
// the BEAM judge in mem0's memory-benchmarks (Apache-2.0), itself a port of
// the unified_llm_judge_base_prompt of the BEAM paper's code, so that a score
// here is graded the way published BEAM figures are.
const rubricSystem = `You are an expert evaluator assessing whether an AI assistant's response satisfies
specific rubric criteria. You must be objective, fair, and consistent.

Return JSON conforming to the schema you were given, and nothing else.`

func rubricPrompt(question, item, response string) string {
	return fmt.Sprintf(`Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
%s

LLM RESPONSE:
%s

RUBRIC CRITERION:
%s

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

For POSITIVE requirements (response should contain, mention, or demonstrate something):
- 1.0 (Complete Compliance): The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- 0.5 (Partial Compliance): The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- 0.0 (No Compliance): The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

For NEGATIVE constraints (response should NOT contain or should avoid something):
- 1.0 (Complete Compliance): The response is responsive to the question AND the prohibited element is absent.
- 0.5 (Partial Compliance): The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- 0.0 (No Compliance): The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

Compound statement handling: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. Semantic tolerance: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. Numeric and date equivalence: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. Case / punctuation / whitespace tolerance: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. Hedging tolerance: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. Style neutrality: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. Responsiveness: If the LLM response is completely off-topic or refuses to answer, score 0.0 for all criteria.
7. Independence: Evaluate this criterion in isolation; do not consider other rubric items.
8. Specificity matters: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
1. Understand the Requirement: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. Parse Compound Statements: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. Check Compliance: Compare the LLM response against each element, applying the tolerance rules above.
4. Assign Score: Use the appropriate scoring table and compound-statement rule to determine the score.
5. Provide Reasoning: Write a concise explanation referencing which elements were or were not satisfied.
`, question, response, item)
}

var rubricSchema = json.RawMessage(`{"type":"object","additionalProperties":false,` +
	`"required":["reason","score"],"properties":{` +
	`"reason":{"type":"string","description":"One concise sentence explaining the score."},` +
	`"score":{"type":"number","enum":[0,0.5,1]}}}`)

// validRubricScore holds a judge to the three scores the rubric allows.
func validRubricScore(score float64) bool {
	return score == 0 || score == 0.5 || score == 1
}
