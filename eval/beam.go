// BEAM: long conversations between a user and an assistant, from a hundred
// thousand to a million tokens each, asked twenty questions apiece.

package eval

import (
	"bufio"
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/mempher"
)

// beamDate is how BEAM dates a turn: "March-15-2024".
const beamDate = "January-02-2006"

// beamTypes are BEAM's ten question types, in the order the benchmark lists
// them. Each conversation asks two questions of each.
var beamTypes = []string{
	"abstention",
	"contradiction_resolution",
	"event_ordering",
	"information_extraction",
	"instruction_following",
	"knowledge_update",
	"multi_session_reasoning",
	"preference_following",
	"summarization",
	"temporal_reasoning",
}

// beamMarker is the position label BEAM appends to some user turns, such as
// " ->-> 1,1". It is generation bookkeeping, not something the user said.
var beamMarker = regexp.MustCompile(`\s*->->\s*\d+\s*,\s*\d+\s*$`)

// beamAnswerStyle replaces the default request for the shortest answer. BEAM
// asks for summaries, advice and code as well as facts, and grades whether an
// answer follows the instructions and preferences the user gave earlier, so a
// one-line answer would fail it for its form.
const beamAnswerStyle = "Answer as the assistant in these conversations would, as fully as the " +
	"question needs: a fact, a summary, an explanation, a recommendation or code. Follow every " +
	"instruction and preference the user stated earlier that applies to the question. Combine " +
	"every relevant part of the memory rather than stopping at the first one. When two parts " +
	"disagree, say so if the question is about that, and otherwise use the more recent one. If " +
	"the memory does not hold the answer, say that you do not have enough information to answer."

// BEAM reads one size of the BEAM benchmark, converted to JSON lines.
//
// BEAM is published as Parquet, with its questions stored as a Python literal,
// and this module reads neither. One command converts a split:
//
//	uv run --with pyarrow python -c 'import ast, json, sys, pyarrow.parquet as pq
//	for r in pq.read_table(sys.argv[1]).to_pylist():
//	    r["probing_questions"] = ast.literal_eval(r["probing_questions"])
//	    print(json.dumps(r))' 1M.parquet > beam-1M.jsonl
//
// Every conversation is one haystack, shared by its twenty questions. Turns are
// recorded as the user's and the assistant's, dated by the session they belong
// to. Each question is graded by its rubric, item by item, rather than against
// one expected answer: see [Question.Rubric].
//
// The 10M size is published in another layout and is not read here. Event
// ordering is graded by its rubric like every other type, not by the Kendall
// tau the paper adds for it.
func BEAM(path string) Dataset {
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return Dataset{
		Name:        "BEAM (" + name + ")",
		Questions:   beamQuestions(path),
		AnswerStyle: beamAnswerStyle,
		Instructions: "These are conversations between a user and an AI assistant about a project " +
			"the user is working on. Record what the user says about themselves and their project: " +
			"plans, decisions, dates and deadlines, figures, and every instruction or preference " +
			"they give about how the assistant should answer. When the user corrects or updates " +
			"something, record the new value.",
	}
}

type beamConversation struct {
	ID        string                       `json:"conversation_id"`
	Chat      [][]beamTurn                 `json:"chat"`
	Questions map[string][]json.RawMessage `json:"probing_questions"`
}

type beamTurn struct {
	ID      int    `json:"id"`
	Role    string `json:"role"`
	Content string `json:"content"`
	Anchor  string `json:"time_anchor"`
}

// beamQuestion holds the fields every type shares. The expected answer is
// under a different key for each type.
type beamQuestion struct {
	Question string          `json:"question"`
	Rubric   []string        `json:"rubric"`
	Sources  json.RawMessage `json:"source_chat_ids"`
}

// beamAnswerKeys are where the types keep their reference answer, in the order
// they are tried.
var beamAnswerKeys = []string{"answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance"}

func beamQuestions(path string) iter.Seq2[Question, error] {
	return func(yield func(Question, error) bool) {
		f, err := os.Open(path)
		if err != nil {
			yield(Question{}, fmt.Errorf("eval: open the benchmark: %w", err))
			return
		}
		defer func() { _ = f.Close() }()

		lines := bufio.NewScanner(f)
		// One conversation is one line, and a 1M conversation is several
		// megabytes of JSON.
		lines.Buffer(make([]byte, 0, 1<<20), 64<<20)
		for lines.Scan() {
			if len(strings.TrimSpace(lines.Text())) == 0 {
				continue
			}
			var c beamConversation
			if err := json.Unmarshal(lines.Bytes(), &c); err != nil {
				yield(Question{}, fmt.Errorf("eval: decode a conversation: %w", err))
				return
			}
			questions, err := c.questions()
			if err != nil {
				yield(Question{}, err)
				return
			}
			for _, q := range questions {
				if !yield(q, nil) {
					return
				}
			}
		}
		if err := lines.Err(); err != nil {
			yield(Question{}, fmt.Errorf("eval: read the benchmark: %w", err))
		}
	}
}

func (c beamConversation) questions() ([]Question, error) {
	haystack, index, last, err := c.haystack()
	if err != nil {
		return nil, err
	}
	scope := "beam-" + c.ID
	var out []Question
	for _, kind := range beamTypes {
		for i, raw := range c.Questions[kind] {
			var q beamQuestion
			if err := json.Unmarshal(raw, &q); err != nil {
				return nil, fmt.Errorf("eval: beam %s: %s question %d: %w", c.ID, kind, i, err)
			}
			if strings.TrimSpace(q.Question) == "" || len(q.Rubric) == 0 {
				return nil, fmt.Errorf("eval: beam %s: %s question %d has no question or no rubric",
					c.ID, kind, i)
			}
			question := Question{
				ID:       fmt.Sprintf("%s-%s-%d", scope, kind, i),
				Type:     kind,
				Scope:    scope,
				Query:    q.Question,
				Answer:   beamAnswer(raw),
				Rubric:   q.Rubric,
				Asked:    last,
				Haystack: haystack,
			}
			for _, id := range beamSources(q.Sources) {
				if at, ok := index[id]; ok && !slices.Contains(question.Relevant, at) {
					question.Relevant = append(question.Relevant, at)
				}
			}
			out = append(out, question)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("eval: beam %s: the conversation has no questions", c.ID)
	}
	return out, nil
}

// haystack flattens the sessions into episodes in the order they were said,
// and returns where each turn id landed and a moment after the last session.
func (c beamConversation) haystack() ([]mempher.AppendRequest, map[int]int, time.Time, error) {
	var (
		out   []mempher.AppendRequest
		index = map[int]int{}
		last  time.Time
	)
	for n, session := range c.Chat {
		var day time.Time
		for i, turn := range session {
			if turn.Anchor != "" {
				at, err := time.Parse(beamDate, turn.Anchor)
				if err != nil {
					return nil, nil, time.Time{}, fmt.Errorf("eval: beam %s: session %d: date %q: %w",
						c.ID, n, turn.Anchor, err)
				}
				day = at
			}
			if day.IsZero() {
				return nil, nil, time.Time{}, fmt.Errorf("eval: beam %s: session %d has no date", c.ID, n)
			}
			content := strings.TrimSpace(beamMarker.ReplaceAllString(turn.Content, ""))
			if content == "" {
				continue
			}
			role := mempher.RoleUser
			if turn.Role == "assistant" {
				role = mempher.RoleAssistant
			}
			index[turn.ID] = len(out)
			// Sessions start at nine, turns a second apart, so that a reader
			// sorting by time keeps them in order.
			occurred := day.Add(9*time.Hour + time.Duration(i)*time.Second)
			out = append(out, mempher.AppendRequest{
				Content:    content,
				Role:       role,
				Source:     "beam",
				OccurredAt: occurred,
				Binding:    mempher.Binding{"session": fmt.Sprint(n + 1)},
			})
			last = occurred
		}
	}
	if len(out) == 0 {
		return nil, nil, time.Time{}, fmt.Errorf("eval: beam %s: the conversation holds no turns", c.ID)
	}
	return out, index, last.Add(24 * time.Hour), nil
}

// beamAnswer finds a question's reference answer, which the types keep under
// different keys. It is shown to no judge, since rubrics grade BEAM; it is kept
// so a result can be read next to what was expected.
func beamAnswer(raw json.RawMessage) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return ""
	}
	for _, key := range beamAnswerKeys {
		var s string
		if json.Unmarshal(fields[key], &s) == nil && s != "" {
			return s
		}
	}
	return ""
}

// beamSources reads the turn ids a question cites. They come as a list, a list
// of lists, a map of lists, or not at all, so every number found is taken.
func beamSources(raw json.RawMessage) []int {
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return nil
	}
	var ids []int
	var walk func(any)
	walk = func(v any) {
		switch v := v.(type) {
		case float64:
			ids = append(ids, int(v))
		case []any:
			for _, item := range v {
				walk(item)
			}
		case map[string]any:
			keys := make([]string, 0, len(v))
			for key := range v {
				keys = append(keys, key)
			}
			slices.Sort(keys)
			for _, key := range keys {
				walk(v[key])
			}
		}
	}
	walk(value)
	return ids
}
