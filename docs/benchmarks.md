# Benchmarks

## Speed

PostgreSQL 18 in a container on a developer laptop, a scope of 1,000 episodes,
and an embedder that costs microseconds so the numbers are mempher's own.

| | per operation | per episode |
| --- | --- | --- |
| `Append` | 0.45 ms | 0.45 ms |
| `AppendBatch`, 8 | 1.2 ms | 0.16 ms |
| `AppendBatch`, 32 | 2.2 ms | 0.07 ms |
| `AppendBatch`, 64 | 3.3 ms | 0.05 ms |
| `Recall`, all channels | 4.1 ms | |
| `Recall`, semantic only | 2.5 ms | |
| `Recall`, lexical only | 1.1 ms | |
| `Recall`, with facts | 4.3 ms | |

A single durable append is mostly one WAL fsync. A batch shares one round trip
and one commit, and is also one encode job, so one embedding call.

Your embedder adds its latency to `Recall` and nothing to an append, because
the write path makes no model call. Reproduce with `make bench`.

## Retrieval quality

[LongMemEval](https://github.com/xiaowu0162/LongMemEval): 500 questions, each
with its own haystack of about 50 sessions and 490 turns, 246,930 episodes in
all, and a median of 2 turns per question that carry the answer.

| k=10, 479 questions scored | hit@10 | recall@10 | nDCG@10 | MRR |
| --- | --- | --- | --- | --- |
| lexical alone | 0.384 | 0.313 | 0.166 | 0.123 |
| semantic alone | 0.850 | 0.727 | 0.513 | 0.495 |
| both, fused (the default) | 0.871 | 0.743 | 0.482 | 0.425 |

Fused, by question type:

| | hit@10 | | hit@10 |
| --- | --- | --- | --- |
| single-session-assistant | 1.000 | multi-session | 0.872 |
| knowledge-update | 0.972 | temporal-reasoning | 0.773 |
| single-session-user | 0.922 | single-session-preference | 0.700 |

The embedder is `all-MiniLM-L6-v2`, 384 dimensions, running locally. Recall
latency across both channels was 23 ms at p50, most of it the query embedding.

The fusion defaults (`DefaultLexicalWeight`, `DefaultFusionK`) come from a
grid search on this benchmark. The weight matters; the constant barely does.

### What this measures

Retrieval, not question answering. The benchmark marks the turns that hold an
answer, and an episode is a turn, so the labels line up exactly and no LLM
judge is needed. These numbers are not comparable to the end-to-end accuracy
figures usually quoted for LongMemEval, which mostly measure the model that
reads the context.

The fact channel and entity resolution are measured end to end below.

Reproduce with `make eval LONGMEMEVAL=/path/to/longmemeval_s`. The harness is
[`mempher/eval`](https://pkg.go.dev/github.com/gojargo/mempher/eval).

## End-to-end answers

[LoCoMo](https://github.com/snap-research/locomo): 10 long conversations
between two people, 5,882 turns in all, and 1,540 questions (the adversarial
category is left out, as is usual). For each question mempher recalls, a model
writes an answer from what came back, and a second model judges the answer
against the expected one. The judge is told to accept other wording and other
date formats, and to require the same answer.

### OpenAI models

| Recall returns | accuracy | single-hop | temporal | multi-hop | open-domain | context |
| --- | --- | --- | --- | --- | --- | --- |
| episodes | 0.708 | 0.810 | 0.670 | 0.532 | 0.469 | 450 tokens |
| episodes and facts (the default) | 0.734 | 0.832 | 0.692 | 0.578 | 0.469 | 887 tokens |
| facts | 0.331 | 0.366 | 0.259 | 0.319 | 0.292 | 437 tokens |

| | |
| --- | --- |
| Extractor | `gpt-4o-mini`, up to 64 assertions per extraction |
| Embedder | `text-embedding-3-small`, 1536 dimensions |
| Answers | `gpt-4o-mini` |
| Judge | `gpt-4o` |
| Recall | 10 episodes, 32 facts, no token budget |

- The extractor kept 949 facts, and every extraction job succeeded.
- The labelled evidence came back for 79% of questions. With it, 81% of
  answers were right; without it, 47% with facts and 31% without.
- The run took 38 minutes. [Comparison](comparison.md) puts it next to the
  figures published for other systems.

### Mistral models

| Recall returns | accuracy | single-hop | temporal | multi-hop | open-domain | context |
| --- | --- | --- | --- | --- | --- | --- |
| episodes | 0.644 | 0.753 | 0.642 | 0.390 | 0.438 | 400 tokens |
| episodes and facts (the default) | 0.686 | 0.803 | 0.664 | 0.426 | 0.510 | 808 tokens |
| facts | 0.253 | 0.257 | 0.240 | 0.241 | 0.302 | 408 tokens |

Questions per category: 841 single-hop, 321 temporal, 282 multi-hop, 96
open-domain.

The setup:

| | |
| --- | --- |
| Extractor | `mistral-small-latest`, up to 64 assertions per extraction |
| Embedder | `mistral-embed`, 1024 dimensions |
| Answers | `mistral-small-latest` |
| Judge | `mistral-medium-latest` |
| Recall | 10 episodes, 32 facts, no token budget |

What the run shows:

- Facts add 4 points on top of episodes, for 400 more tokens of context.
- Facts alone are not enough. The extractor kept 861 facts from 5,882 turns,
  and 32 of them are returned per question.
- Retrieval is the larger share of the misses. The labelled evidence came back
  for 77% of questions. With it, 78% of answers were right; without it, 39%.
- The context is small. Ten short turns is 400 tokens. A larger limit would
  raise the evidence rate; it was not measured.
- Run to run, the same configuration moves by about half a point: an earlier
  run of the same setup scored 0.638, 0.691 and 0.263.

What the run found, and what was done about it:

- In an earlier run the extractor declared the two speakers of one
  conversation to be the same entity, and their facts were merged. An alias
  from an extraction now needs a confidence of 0.9 and never merges two
  entities that both have facts. In the run above every conversation kept its
  two speakers apart, and the aliases declared were nicknames (Cal, Deb, Mel,
  Jo).
- In that earlier run 7 of 95 extraction jobs died because the model restated
  a claim the scope already held over a different window. The stored window
  now stands and the job succeeds.
- In the run above 4 of 95 extraction jobs still died, because the model gave
  one claim two overlapping windows in a single answer. Those turns, about 4%,
  have no facts in these numbers. The extractor now keeps the first window and
  drops the other; that change is not in the numbers.

These numbers move with the four models above and with the judge's wording.
They are not comparable to figures published for other systems, which use
other models and other judge prompts.

Reproduce with `make qa LOCOMO=/path/to/locomo10.json` and the `QA_*`
variables described in
[`eval/qa_run_test.go`](../eval/qa_run_test.go). The run takes about an hour
at 5 requests per second.
