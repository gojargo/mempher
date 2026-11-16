// The answer and grader prompts of Zep's LoCoMo harness, for runs that are set
// beside Zep's published results.

package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gojargo/mempher"
	"github.com/gojargo/mempher/extract"
)

// The prompts below are copied from getzep/zep, benchmarks/locomo/prompts.py
// (Apache-2.0), with their wording unchanged. Only the context differs: Zep's
// is facts and entities from its graph, and this is what mempher recalled,
// introduced by zepContextHeader.

const zepResponseSystem = `
You are a helpful expert assistant answering questions based on the provided context.
`

const zepResponsePrompt = `
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

Clarification:
When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

Example:

Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023
Explanation:
Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Context:

%s

Question: %s
Answer:
`

const zepGraderSystem = `
You are an expert grader that determines if answers to questions match a gold standard answer.
`

const zepGraderPrompt = `
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: %s
Gold answer: %s
Generated answer: %s

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
`

// zepContextHeader says what the two parts of mempher's context are, which
// Zep's context template does for its own.
const zepContextHeader = "FACTS are claims drawn from the conversation, each with the period over " +
	"which it holds. EXCERPTS are messages as they were written, each with the date and time it " +
	"was sent.\n\n"

var (
	zepAnswerSchema = json.RawMessage(`{"type":"object","additionalProperties":false,` +
		`"required":["answer"],"properties":{"answer":{"type":"string"}}}`)
	// zepGradeSchema is Zep's Grade model: is_correct, then reasoning.
	zepGradeSchema = json.RawMessage(`{"type":"object","additionalProperties":false,` +
		`"required":["is_correct","reasoning"],"properties":{` +
		`"is_correct":{"type":"string","description":"CORRECT or WRONG"},` +
		`"reasoning":{"type":"string","description":"Explain why the answer is correct or incorrect."}}}`)
)

// zepAnswer writes and grades one answer with Zep's prompts. out arrives with
// recall's part filled in.
func zepAnswer(
	ctx context.Context,
	cfg QAConfig,
	q Question,
	_ Variant,
	_ mempher.Recollection,
	out QAResult,
) QAResult {
	var answer struct {
		Answer *string `json:"answer"`
	}
	err := complete(ctx, cfg.Answerer, cfg.Attempts, extract.Prompt{
		System: zepResponseSystem,
		User:   fmt.Sprintf(zepResponsePrompt, zepContextHeader+out.Context, q.Query),
		Schema: zepAnswerSchema,
	}, &answer, func() bool { return answer.Answer != nil })
	if err != nil {
		out.Err = fmt.Errorf("answer: %w", err)
		return out
	}
	out.Response = strings.TrimSpace(*answer.Answer)

	var grade struct {
		IsCorrect *string `json:"is_correct"`
	}
	err = complete(ctx, cfg.Judge, cfg.Attempts, extract.Prompt{
		System: zepGraderSystem,
		User:   fmt.Sprintf(zepGraderPrompt, q.Query, q.Answer, out.Response),
		Schema: zepGradeSchema,
	}, &grade, func() bool { return grade.IsCorrect != nil })
	if err != nil {
		out.Err = fmt.Errorf("judge: %w", err)
		return out
	}
	// Zep's own reading of the verdict: anything but "correct" is wrong.
	out.Correct = strings.EqualFold(strings.TrimSpace(*grade.IsCorrect), "correct")
	if out.Correct {
		out.Score = 1
	}
	return out
}
