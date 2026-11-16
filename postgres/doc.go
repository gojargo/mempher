// Package postgres implements mempher's storage on PostgreSQL 18 with pgvector.
//
// It is a separate package so the root package carries no database driver: the
// ports there stay honest rather than decorative.
//
// [Store] satisfies both [mempher.Store] and [mempher.Queue]. One pool serves
// both, and appending an episode writes to both in a single statement: an
// episode committed without its encode job would be permanently invisible to
// the semantic channel.
//
// Everything lives in one schema, [Schema], so mempher installs alongside an
// existing application schema. The pgvector and btree_gin extensions are used
// wherever the database already keeps them, discovered from the catalog rather
// than assumed, because a ::vector cast fails outright when the extension's
// schema is off the search_path.
//
// Since migration 0005 the schema carries a SQL API, and [Store] is a thin
// client of it: each method is one call to a function such as mempher.append,
// mempher.lease_jobs, mempher.apply_extraction or mempher.recall, which
// validates its own input and raises a documented SQLSTATE in class MP for a
// caller's mistake. [Store] maps those codes back to the mempher package's
// sentinels, so errors.Is works as it always did. The functions are listed,
// with their contract, under "The SQL API" in 0001_init.sql and in docs/schema.md.
//
// Migrations are embedded, forward-only and rendered as templates. See [Migrate].
package postgres
