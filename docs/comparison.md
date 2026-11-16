# mempher and other memory systems

## What each one is

| | mempher | mem0 | Zep (Graphiti) | pgvector by hand |
| --- | --- | --- | --- | --- |
| Client language | Go | Python, TypeScript | Python, or SDKs for Zep Cloud | any |
| What you run | PostgreSQL 18 with pgvector | a vector store, optionally a graph store | Graphiti and a graph database (Neo4j, FalkorDB and others), or Zep Cloud | PostgreSQL with pgvector |
| Hosted offering | none | yes | yes | not applicable |
| Search | vectors and PostgreSQL full text, fused | vectors, optionally a graph | vectors, BM25 and graph traversal | what you write |
| Facts | extracted by a model, with validity windows | extracted by a model, which adds, updates or deletes memories | extracted by a model into a graph whose edges carry validity times | none |
| When a fact changes | the old window is closed, both answers remain | the memory is updated or deleted, with a history record | the old edge is invalidated, both remain | not applicable |
| Source of truth | the messages, append-only; everything else can be rebuilt | the extracted memories | the episodes and the graph | your table |
| Erasing a user | one transaction, with every derived row | through the API | through the API | your own deletes |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 (Graphiti) | not applicable |

## LoCoMo, graded like Zep

Zep publishes the harness behind its LoCoMo results
([getzep/zep, benchmarks/locomo](https://github.com/getzep/zep/tree/main/benchmarks/locomo)),
with ten runs of each configuration and every answer. mempher was run with the
same answer prompt and grader prompt, word for word, the same models
(`gpt-4o-mini` at temperature 0 for answers and grading) and the same 1,540
questions. Context is counted the same way, with tiktoken over the whole
context string.

| Context per question | Zep, mean of 10 runs | mempher, 1 run |
| --- | --- | --- |
| 347 tokens | 0.696 | |
| about 500 tokens | 0.737 (504 tokens) | 0.764 (497 tokens) |
| 756 tokens | 0.771 | |
| about 1,300 tokens | 0.801 (1,377 tokens) | 0.793 (1,278 tokens) |
| about 1,800 to 2,000 tokens | 0.803 (1,996 tokens) | 0.806 (1,796 tokens) |

mempher recalled 5 messages and 8 facts, 10 and 32, and 20 and 32. Zep's runs
recall 5 to 30 facts and 2 to 30 entities from Zep Cloud, with a cross-encoder
reranker. mempher used `gpt-4o-mini` to extract facts and
`text-embedding-3-small` to embed, on PostgreSQL alone.

What differs:

- The context is each system's own: Zep's facts and entities, mempher's facts
  and messages. Zep's prompt says a timestamp is when an event happened;
  mempher's messages are dated when they were sent, which the context states.
- mempher's answers are returned as JSON with one field, Zep's as text.
- Zep's figures are the mean of ten runs, whose spread is about half a point.
  mempher's are one run.
- Zep's runs date from December 2025. Its site reports higher figures from
  other models and settings, which were not reproduced.

Every answer, with its context and verdict, is kept by `QA_RECORD`; the
setting is in [benchmarks](benchmarks.md).

## Scores on LoCoMo, mem0 paper setup

The published LoCoMo figures most often compared are those of the mem0
paper ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413), table 2). In
that paper GPT-4o-mini makes every model call except the judge's, which is
described as a separate, more capable model, and the adversarial category is
left out. mempher was run the same way: GPT-4o-mini for extraction and
answers, GPT-4o as the judge, `text-embedding-3-small` for embeddings, 1,540
questions.

| System | LoCoMo, LLM judge | Source |
| --- | --- | --- |
| mempher | 0.734 | this repository, [benchmarks](benchmarks.md) |
| full context | 0.729 | mem0 paper |
| mem0 with graph | 0.684 | mem0 paper |
| mem0 | 0.669 | mem0 paper |
| Zep | 0.660 | mem0 paper; Zep reports 0.751 for its own run |
| LangMem | 0.581 | mem0 paper |
| OpenAI memory | 0.529 | mem0 paper |

mempher passed the answering model 887 tokens per question: 10 messages and
32 facts.

The rows other than mempher's were not reproduced here. The judge prompts
differ: the mem0 paper's judge is told to accept an answer that touches the
same topic as the expected one, and mempher's judge is told to accept other
wording and date formats but to require the same answer.

Since then, mem0 and Zep have published higher figures for newer versions,
above 0.9, with more context per question and their own judges. Those runs
are not comparable to the table above and were not reproduced.
