// LoCoMo: ten long two-person conversations, each asked about two hundred
// questions.

package eval

import (
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gojargo/mempher"
)

// locomoDate is how a session is dated: "1:56 pm on 8 May, 2023".
const locomoDate = "3:04 pm on 2 January, 2006"

// locomoGrading is the lenient rule LoCoMo answers are customarily judged by.
// The gold answers are terse and the conversations date things relatively, so a
// strict match would mark a right answer wrong for its format.
const locomoGrading = "The expected answer is short. Be generous: the response is correct if it " +
	"refers to the same thing, even in more words or with more detail. For a question " +
	"about time, the response is correct if it refers to the same date or period as the " +
	"expected answer, whatever the format, and whether it is given as a date or relative " +
	"to one."

// locomoCategories names the benchmark's question categories.
var locomoCategories = map[int]string{
	1: "multi-hop",
	2: "temporal",
	3: "open-domain",
	4: "single-hop",
}

// locomoAdversarial is the category whose questions have no answer in the
// conversation.
const locomoAdversarial = 5

// locomoEvidence finds dialogue ids in an evidence entry. Most entries are one
// id; a few hold several separated by spaces or semicolons, and a few are
// malformed and yield nothing.
var locomoEvidence = regexp.MustCompile(`D(\d+):(\d+)`)

// LoCoMo reads the benchmark's locomo10.json.
//
// Every conversation is one haystack, shared by all of its questions through
// [Question.Scope], so [RunQA] ingests ten conversations rather than fifteen
// hundred. Neither speaker is the agent or the person it serves: the
// conversation is something overheard. So every turn is recorded with
// [mempher.RoleObservation], the speaker as its actor, and the speaker's name
// at the front of the content, which is what lets a reader tell them apart.
// Recording both as [mempher.RoleUser] makes an extractor take two people for
// one user and merge them.
//
// The adversarial category is left out, as it is in the figures usually quoted
// for this benchmark: its questions carry no expected answer, only a wrong one
// to avoid.
func LoCoMo(path string) Dataset {
	return Dataset{
		Name:      "LoCoMo",
		Questions: locomoQuestions(path),
		Instructions: "These are conversations between two people, each line starting with the " +
			"speaker's name. Neither of them is \"the user\": never use \"the user\" as a " +
			"subject. Use each person's name as the subject of the facts about them.",
	}
}

type locomoSample struct {
	ID           string                     `json:"sample_id"`
	Conversation map[string]json.RawMessage `json:"conversation"`
	QA           []locomoQA                 `json:"qa"`
}

type locomoTurn struct {
	Speaker string `json:"speaker"`
	ID      string `json:"dia_id"`
	Text    string `json:"text"`
	Caption string `json:"blip_caption"`
}

type locomoQA struct {
	Question string   `json:"question"`
	Answer   loose    `json:"answer"`
	Evidence []string `json:"evidence"`
	Category int      `json:"category"`
}

func locomoQuestions(path string) iter.Seq2[Question, error] {
	return func(yield func(Question, error) bool) {
		raw, err := os.ReadFile(path)
		if err != nil {
			yield(Question{}, fmt.Errorf("eval: open the benchmark: %w", err))
			return
		}
		var samples []locomoSample
		if err := json.Unmarshal(raw, &samples); err != nil {
			yield(Question{}, fmt.Errorf("eval: decode the benchmark: %w", err))
			return
		}
		for _, sample := range samples {
			haystack, index, last, err := sample.haystack()
			if err != nil {
				yield(Question{}, err)
				return
			}
			for i, qa := range sample.QA {
				if qa.Category == locomoAdversarial {
					continue
				}
				kind, ok := locomoCategories[qa.Category]
				if !ok {
					yield(Question{}, fmt.Errorf("eval: %s: question %d has unknown category %d",
						sample.ID, i, qa.Category))
					return
				}
				q := Question{
					ID:       fmt.Sprintf("%s-q%d", sample.ID, i),
					Type:     kind,
					Scope:    sample.ID,
					Query:    qa.Question,
					Answer:   string(qa.Answer),
					Grading:  locomoGrading,
					Asked:    last,
					Haystack: haystack,
				}
				seen := map[int]struct{}{}
				for _, entry := range qa.Evidence {
					for _, id := range locomoEvidence.FindAllString(entry, -1) {
						at, known := index[id]
						if _, dup := seen[at]; known && !dup {
							seen[at] = struct{}{}
							q.Relevant = append(q.Relevant, at)
						}
					}
				}
				if !yield(q, nil) {
					return
				}
			}
		}
	}
}

// haystack flattens a conversation's sessions into episodes in the order they
// were spoken, and returns where each dialogue id landed and when the last
// session took place.
func (s locomoSample) haystack() ([]mempher.AppendRequest, map[string]int, time.Time, error) {
	var (
		out   []mempher.AppendRequest
		index = map[string]int{}
		last  time.Time
	)
	for n := 1; ; n++ {
		key := "session_" + strconv.Itoa(n)
		rawTurns, ok := s.Conversation[key]
		if !ok {
			break
		}
		var stamp string
		if err := json.Unmarshal(s.Conversation[key+"_date_time"], &stamp); err != nil {
			return nil, nil, time.Time{}, fmt.Errorf("eval: %s: %s has no date: %w", s.ID, key, err)
		}
		occurred, err := time.Parse(locomoDate, stamp)
		if err != nil {
			return nil, nil, time.Time{}, fmt.Errorf("eval: %s: %s date %q: %w", s.ID, key, stamp, err)
		}
		var turns []locomoTurn
		if err := json.Unmarshal(rawTurns, &turns); err != nil {
			return nil, nil, time.Time{}, fmt.Errorf("eval: %s: decode %s: %w", s.ID, key, err)
		}
		for i, turn := range turns {
			content := strings.TrimSpace(turn.Text)
			if caption := strings.TrimSpace(turn.Caption); caption != "" {
				// The turn shared a photograph, and the caption is all of it a
				// text benchmark has.
				content = strings.TrimSpace(content + " [shares a photo: " + caption + "]")
			}
			if content == "" {
				continue
			}
			index[turn.ID] = len(out)
			out = append(out, mempher.AppendRequest{
				Content: turn.Speaker + ": " + content,
				Role:    mempher.RoleObservation,
				Actor:   mempher.Actor(turn.Speaker),
				Source:  "locomo",
				// A second apart, so that turns of one session keep their
				// order when a reader sorts by time.
				OccurredAt: occurred.Add(time.Duration(i) * time.Second),
				Binding:    mempher.Binding{"session": key},
			})
		}
		last = occurred
	}
	if len(out) == 0 {
		return nil, nil, time.Time{}, fmt.Errorf("eval: %s: the conversation holds no turns", s.ID)
	}
	return out, index, last.Add(24 * time.Hour), nil
}
