// The persistence port.

package mempher

import "context"

// Store is the persistence port: L0, its vector projection, and the retrieval
// primitives over both. The postgres subpackage is the one implementation.
//
// Three rules hold across every method. A Store never reads a clock: every
// timestamp arrives as an argument, which is what makes appends replayable. A
// Store never crosses a scope, and an episode addressed from the wrong scope is
// [ErrNotFound], not an authorization error. And a Store never updates an
// episode; there is deliberately no method that could.
//
// Removing one is the single exception, and it has exactly one method, so that
// the whole of erasure is one thing to find, review and audit. See [Store.Forget].
//
// Retrieval is one call rather than one per channel, so that the channels,
// their fusion and the budget all run where the data is, in one round trip. The
// postgres adapter implements it as the mempher.recall function, which a client
// in any language can call.
//
// What a Store needs for operating a deployment rather than serving it -- the
// scope catalogue, and the anti-join saying what work is outstanding -- is
// declared in the ops subpackage instead, and satisfied by the same type.
type Store interface {
	// Append inserts one or more episodes and the jobs they imply,
	// atomically, creating the scope if this is its first episode. The
	// episodes are sequenced in the order given.
	Append(ctx context.Context, cmd AppendCommand) (AppendResult, error)

	// Episode returns one episode by id within a scope, or [ErrNotFound].
	Episode(ctx context.Context, scope ScopeID, id EpisodeID) (Episode, error)

	// Episodes returns the named episodes of a scope in Seq order, in one
	// read, or [ErrNotFound] if any of them is not in the scope. It is how a
	// worker reads the episodes a job names.
	Episodes(ctx context.Context, scope ScopeID, ids []EpisodeID) ([]Episode, error)

	// Replay returns up to limit episodes of a scope in Seq order, starting
	// after afterSeq. It is how a projection is rebuilt from L0.
	Replay(ctx context.Context, scope ScopeID, afterSeq int64, limit int) ([]Episode, error)

	// Recall runs a resolved query's channels, fuses the episode channels by
	// reciprocal rank, reads the facts the query's extractor holds, and
	// spends the token budget, facts first. It reports each channel it ran,
	// with how many candidates it found and how long it took.
	Recall(ctx context.Context, q RecallQuery) (Recollection, error)

	// PutEncodings writes a batch of encodings in one statement, replacing
	// any existing one for the same episode and model, which is what makes
	// encode jobs safe to retry. The batch shares one Model and one EncodedAt.
	// It fills each encoding's scope and ingest time from the episode row, and
	// reports [ErrNotFound] if an episode does not exist.
	PutEncodings(ctx context.Context, encodings []Encoding) error

	// Forget erases episodes and every row this store holds that derives
	// from them, and reports what went.
	//
	// It must be atomic, in the way [Store.Append] must: a half-erased scope
	// is not a smaller erasure, it is a failed one. An implementation
	// therefore removes the facts, extraction markers, encodings and jobs in
	// the same transaction as the episodes, whichever of its tables they
	// live in -- and a deployment that keeps L1 in a genuinely separate
	// backend cannot get that from this method and must erase there too.
	//
	// Erasing what is not there is not an error. A scope that never existed,
	// or an episode id it never held, removes nothing and says so.
	Forget(ctx context.Context, req ForgetRequest) (ForgetResult, error)
}
