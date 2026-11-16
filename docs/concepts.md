# Concepts

## One log, and projections of it

mempher stores an agent's memory as an append-only log of episodes. Everything
else is a projection of that log: the vectors, the facts, the entities. A
projection can be deleted and rebuilt by replaying the log.

That one decision is where the useful properties come from:

- Change the embedding model: the old vectors stay, new ones are a backfill.
- Fix a bad extraction prompt: drop the facts and replay the log through the
  fixed extractor.
- Honour an erasure request: one transaction removes the episodes and
  everything derived from them.

Nothing is lost that the log did not lose.

## The layers

| Layer | Table | What it holds | Written by |
| --- | --- | --- | --- |
| L0 | `mempher.episodes` | what was said, exactly as recorded | `mempher.append` |
| projection | `mempher.episode_encodings` | one vector per episode and embedding model | a worker |
| L1 | `mempher.facts` | what is true, as triples with a validity window | a worker |
| L1 | `mempher.entities`, `mempher.entity_aliases` | what the facts are about, and every name each goes by | a worker |
| projection | `mempher.fact_encodings` | one vector per fact and embedding model | a worker |
| queue | `mempher.jobs` | the work the episodes still imply | `mempher.append` |

Every table has a `scope_id`. A scope is the unit of isolation, usually one end
user or one agent, and no function reads across scopes.

## The invariant

Episodes are immutable and are the only source of truth. A trigger on
`mempher.episodes` rejects `UPDATE` and `TRUNCATE`, and rejects `DELETE` unless
it comes from `mempher.forget`.

- An episode is never updated. A row that exists is exactly what was recorded.
- A projection is only written by a worker draining the job queue.

## Three loops

| Loop | When | Model calls | What happens |
| --- | --- | --- | --- |
| Write | on the request path, one round trip | none | insert the episodes, queue the work they imply |
| Consolidate | in the background, batched, replayable | all of them | embed episodes, extract facts, embed facts |
| Read | on the request path, one round trip after the query is embedded | one, to embed the query | run the channels, fuse, read the facts, spend the token budget |

An episode is durable and searchable by words as soon as `append` returns,
because its `tsvector` is a generated column. It becomes searchable by meaning
once a worker has embedded it, and has yielded its facts once a worker has
extracted it. Recall reports what each channel contributed, so a thin result is
not mistaken for an empty memory.

## Facts are bounded in time

A fact is a triple (subject, predicate, object) with a sentence beside it and a
validity window.

- The triple is the identity. The same claim extracted twice is one row.
- The sentence is what a model is shown and what search matches.
- The window is event time: when the claim held in the world. `asserted_at` is
  system time: when mempher learned it.

When the world changes, the old fact is not overwritten. Its window is closed
at the moment it stopped being true, and a new fact opens. "Where do they
live?" and "where did they live in 2020?" both keep their answer.

PostgreSQL enforces this with a temporal primary key (`WITHOUT OVERLAPS`): a
scope cannot hold the same claim over two overlapping windows.

## Episodes and facts are not ranked against each other

Recall returns both. Episodes are fused by relevance to the question, using
reciprocal rank fusion over the semantic and lexical channels. Facts are what
the scope currently believes; the question only decides which ones fit when
there are more than the limit. Facts are spent from the token budget first.

## Forgetting

Closing a fact's window is not deleting it. Erasure is a separate operation,
`mempher.forget`, and it is the only one that destroys anything:

```sql
SELECT * FROM mempher.forget('user:8123');            -- the whole scope
SELECT * FROM mempher.forget('user:8123', $1::uuid[]); -- some episodes
```

One transaction removes the episodes, their vectors, their extraction markers,
their queued jobs, every fact read from them, and the entities and names they
introduced. The result is a row of counts, one per table.

A fact is deleted rather than closed, because a closed window still says what
it said. If the remaining episodes still support the claim, the next extraction
derives it again.

## Compared with a mutable memory store

| | A store that updates in place | mempher |
| --- | --- | --- |
| A fact changes | the row is updated | the old window is closed, both answers remain |
| The extraction prompt was wrong | the bad output is the state | drop the projection, replay the log |
| The embedding model changes | re-ingest or migrate vectors | a backfill |
| Someone asks to be erased | delete rows, hope nothing derived remains | one transaction, counted |
| "What was true last year" | usually not answerable | the validity window |
| What you run | a vector database, often a graph database | PostgreSQL |

The costs: PostgreSQL 18 is required, you make your own model calls, and a
projection is only as current as your worker.
