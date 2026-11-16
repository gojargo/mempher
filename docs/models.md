# Where the models plug in

PostgreSQL does everything except call models. Two interfaces cover that, and
both ship with working implementations.

## Embedder

Turns text into a vector.
[`mempher/embed/openai`](https://pkg.go.dev/github.com/gojargo/mempher/embed/openai)
works against any OpenAI-compatible embeddings endpoint, local servers
included.

```go
embedder, err := openai.New(openai.Config{
	Model:      "text-embedding-3-small",
	Dimensions: 1536,
})

// Asymmetric models (bge, e5) want an instruction on the query side only.
embedder, err := openai.New(openai.Config{
	Model: "bge-small-en-v1.5", Dimensions: 384,
	BaseURL:     "http://localhost:8080/v1",
	QueryPrefix: "Represent this sentence for searching relevant passages: ",
})
```

The width is declared, because the schema is migrated for it, and every reply
is checked against it.

An embedding model rejects a text over its input limit, about 8,000 tokens
for OpenAI's. If your episodes can be longer, such as pasted code or
documents, set `MaxInputBytes` below that limit: a long episode is then
embedded by its start, and stored whole.

Vectors are keyed by model id. Changing the model does not touch existing
rows: the new model's vectors are a backfill.

## Extractor

Turns episodes into facts.
[`mempher/extract`](https://pkg.go.dev/github.com/gojargo/mempher/extract)
owns the prompt, the response schema and the validation of what the model
returns.

```go
// Both adapters call the provider's REST API directly, with no SDK.
completer, err := anthropic.New(anthropic.Config{Model: "claude-opus-5"})

// openai.New has the same shape. BaseURL points it at anything
// OpenAI-compatible: Azure, Groq, Together, OpenRouter, Ollama, vLLM.
completer, err := openai.New(openai.Config{Model: "gpt-5"})

ex, err := extract.New(completer, extract.Options{
	Predicates:   []mempher.Predicate{"lives_in", "allergic_to", "works_at"},
	Instructions: "A support agent: care about entitlements and past incidents.",
})

fmt.Println(ex.Model()) // mempher-extract/v3+claude-opus-5+a3f9c1e2
```

Any other provider is one method, `Complete(ctx, extract.Prompt) ([]byte, error)`,
plus `Model() string`.

### Options that matter

| Option | Effect |
| --- | --- |
| `Predicates` | a closed vocabulary of relations, sent to the provider as an enum. Without it a model writes `lives_in` one day and `resides_in` the next, and nothing supersedes anything. |
| `Instructions` | what this deployment considers worth remembering, appended to the shipped prompt |
| `MaxAssertions` | how many facts one extraction may assert |
| `MinAliasConfidence` | how sure the model must be for an alias it declares to be kept; 0.9 by default |
| `Reconcile` | a first, cheap call that picks which known facts the extraction is shown |

### What an extraction returns

- assertions: new facts,
- retractions: known facts the episodes contradict, closed at the moment they
  stopped being true,
- aliases: names the episodes show to be one entity. See
  [entities](entities.md).

### The extractor id

The id combines the shipped prompt revision, the model, and a digest of the
options. Facts are keyed by it. Editing the instructions therefore makes a
different extractor, and its facts are a backfill instead of a mixture with
the old ones.

### A limit to know about

An extraction can only retract facts it was shown, and how many it is shown is
`WorkerConfig.KnownFactLimit`. Extraction is a pure function of its inputs so
that it can be replayed; an extractor that queried the store could not be.
