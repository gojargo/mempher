# Operating mempher

## Run a worker

Consolidation happens off the write path. Something has to drain the queue:

```go
w, err := mempher.NewWorker(mempher.WorkerConfig{
	Store: store, Embedder: myEmbedder, Extractor: myExtractor,
})
go w.Run(ctx) // or w.DrainOnce(ctx) for a one-shot run, and in tests
```

Several workers can run against one database. Jobs are leased with
`FOR UPDATE SKIP LOCKED`, a lease that expires is reclaimed, and every job is
idempotent, so a retry is safe.

## Backfill, and the queue

[`mempher/ops`](https://pkg.go.dev/github.com/gojargo/mempher/ops) holds what
is needed to run mempher and not to use it:

```go
o, err := ops.New(ops.Config{Store: store, Embedder: myEmbedder})

// The work the log implies and the queue has lost: a job that died, a scope
// older than your extractor, a model id that changed.
res, err := o.Backfill(ctx, ops.BackfillRequest{})

dead, err := o.Jobs(ctx, ops.JobQuery{States: []mempher.JobState{mempher.JobStateDead}})
depth, err := o.Stats(ctx, "")    // per kind and state, with the oldest job's age
_, err = o.Retry(ctx, dead[0].ID) // once the cause is dealt with
_, err = o.Purge(ctx, ops.PurgeRequest{Before: lastMonth})
```

`Backfill` asks the log what work it implies and enqueues what is missing. It
reads the projection tables, not the queue's history. It is safe to run live,
safe to run twice, and resumable.

The same operations exist in SQL: `pending_episodes`, `enqueue`, `job_stats`,
`retry_job`, `purge_jobs`.

## Rebuilding a projection

| To rebuild | Do |
| --- | --- |
| episode vectors | `TRUNCATE mempher.episode_encodings`, then backfill |
| facts | `TRUNCATE mempher.facts, mempher.fact_extractions`, then backfill |
| facts and entity names | also `TRUNCATE mempher.entities CASCADE` |
| fact vectors | `TRUNCATE mempher.fact_encodings`, then `mempher.enqueue_fact_encodings` |

## Testing

```sh
make test      # fast suite, no Docker
make test-all  # everything, with -race, against a PostgreSQL 18 container
make bench     # the numbers in benchmarks.md
make check     # what CI runs
```

Tests inject a `Clock` and a fake `Embedder` and never mock the database:
retrieval depends on pgvector's distance ordering and PostgreSQL's text search
ranking.

Without Docker, set `PGTEST_DSN` to a `postgres://` URL on a PostgreSQL 18
server with pgvector, whose role can create databases. Each test then gets its
own database on that server instead of a container.
