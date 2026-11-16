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

The fact channel and entity resolution are not measured here yet. End-to-end
question answering numbers are the next item on the roadmap.

Reproduce with `make eval LONGMEMEVAL=/path/to/longmemeval_s`. The harness is
[`mempher/eval`](https://pkg.go.dev/github.com/gojargo/mempher/eval).
