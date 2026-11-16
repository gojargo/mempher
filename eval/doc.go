// Package eval measures what [mempher.Memory.Recall] actually puts in front of
// a model.
//
// Latency is already answered by the benchmarks in the root package. This is
// the other question, and the harder one: when the answer is somewhere in a
// scope, does recall return it?
//
// # What is measured
//
// Two things, by two entry points.
//
// [Run] measures retrieval. A labelled question names the episodes that carry
// its answer, so its metrics are computable without a model judging anything,
// which is what makes them reproducible, cheap, and free of an opinion about
// which generator you would have used. It is the number this library is
// directly responsible for: whether the evidence was in the context at all.
//
// [RunQA] measures the answer. It recalls, has a model write an answer from
// what came back, and has a second model judge it against the expected one.
// That is the number a user of an agent cares about and the one other memory
// systems publish, and it is also a number that moves with the two models
// involved, so a [QASummary] names them and is comparable only to a run that
// used the same ones. It reports accuracy with and without the labelled
// evidence having been recalled, which is what separates a retrieval failure
// from a reading failure.
//
// # Running one
//
// A [Dataset] is labelled questions. [Run] ingests each question's haystack into
// a scope of its own, recalls against it, and scores what came back.
//
// Nothing here needs a model. With no [Worker] configured, no episode is ever
// encoded and the semantic channel has nothing to return, so a run over the
// lexical channel alone needs no API key and no budget -- and is a floor rather
// than a result. Configure a real [mempher.Embedder] and a worker to measure the
// whole thing.
package eval
