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

## Scores on LoCoMo

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
