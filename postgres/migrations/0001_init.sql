-- 0001_init.sql: the whole of mempher's schema, and its SQL API.
--
-- The first half is the data: L0 (episodes, immutable and append-only), its
-- vector projection, L1 (facts) with its extraction markers, its own vector
-- projection and the entities its facts are about, and the job queue. The second half is the API: every rule the
-- library enforces is a function in the mempher schema, and every client -- the
-- Go package, a service in another language, psql -- is a caller of the same
-- functions.
--
-- Rendered as a Go text/template before it runs. Two deployment decisions are
-- substituted, both recorded in mempher.schema_config so drift is detectable
-- at startup rather than at query time:
--
--   {{.VectorDimensions}}   width of the embedding, from Embedder.Dimensions()
--   {{.TextSearchConfig}}   PostgreSQL text search configuration for FTS
--
-- The migration runner has already created the mempher schema and
-- mempher.schema_migrations, and holds an advisory lock. Migrations are
-- forward-only: this file never changes once it has been released.

-- uuidv7(), uuid_extract_timestamp() and the temporal constraints the next
-- stage needs are all PostgreSQL 18. Fail here, with a legible message, rather
-- than on a missing function three statements later.
DO $$
BEGIN
    IF current_setting('server_version_num')::int < 180000 THEN
        RAISE EXCEPTION
            'mempher requires PostgreSQL 18 or newer, found %',
            current_setting('server_version');
    END IF;
END $$;

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: the vector type and HNSW
CREATE EXTENSION IF NOT EXISTS btree_gin;   -- lets one GIN index cover scope_id + tsvector
-- Temporal primary keys (PERIOD ... WITHOUT OVERLAPS) need GiST operator classes
-- for the scalar parts of the key, which btree_gist supplies.
CREATE EXTENSION IF NOT EXISTS btree_gist;


-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

-- One row, enforced by the primary key. Startup compares these against the
-- configured Embedder: a model swapped for one of a different width is caught
-- as ErrEmbedderMismatch instead of failing on every insert.
CREATE TABLE mempher.schema_config (
    only_row           boolean     PRIMARY KEY DEFAULT true CHECK (only_row),
    vector_dimensions  int         NOT NULL CHECK (vector_dimensions BETWEEN 1 AND 2000),
    text_search_config text        NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now()
);

INSERT INTO mempher.schema_config (vector_dimensions, text_search_config)
VALUES ({{.VectorDimensions}}, '{{.TextSearchConfig}}');


-- ---------------------------------------------------------------------------
-- Scopes
-- ---------------------------------------------------------------------------

-- A scope is the unit of isolation, and this table exists to allocate its
-- episode sequence. Append upserts here and takes last_seq + 1 in the same
-- statement, so seq is dense and gap-free without a second round trip or an
-- explicit lock; concurrent appends to one scope serialise on this row, which
-- is the correct semantics for a totally ordered log.
CREATE TABLE mempher.scopes (
    id         text        NOT NULL,
    last_seq   bigint      NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL,

    CONSTRAINT scopes_pkey        PRIMARY KEY (id),
    CONSTRAINT scopes_id_len      CHECK (length(id) BETWEEN 1 AND 200),
    CONSTRAINT scopes_seq_nonneg  CHECK (last_seq >= 0)
);


-- ---------------------------------------------------------------------------
-- L0: episodes. Immutable, append-only, the only source of truth.
-- ---------------------------------------------------------------------------

-- Nothing in this library issues UPDATE or DELETE against this table. Every
-- other table here is a projection that can be dropped and rebuilt from it.
--
-- There is deliberately no vector column. An embedding cannot be produced on
-- the write path without a model call, and it cannot be added afterwards
-- without an UPDATE, so it lives in episode_encodings, keyed by model, where
-- re-embedding is an insert rather than a mutation of L0.
CREATE TABLE mempher.episodes (
    id          uuid        NOT NULL DEFAULT uuidv7(),
    scope_id    text        NOT NULL,
    seq         bigint      NOT NULL,
    content     text        NOT NULL,
    role        text        NOT NULL,
    actor       text        NOT NULL DEFAULT '',
    source      text        NOT NULL,

    -- occurred_at is event time: when the thing happened in the world.
    -- ingested_at is system time: when this library learned of it. Both come
    -- from the caller's injected Clock, never from the server, so a replay
    -- reproduces byte-identical rows. The id's own uuidv7 timestamp is
    -- separate: it is physical arrival time, from the database.
    occurred_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL,

    binding     jsonb       NOT NULL DEFAULT '{}'::jsonb,

    -- Generated, so it exists the instant the row lands: an episode is
    -- lexically searchable with no worker involved. Immutable content means it
    -- never needs recomputing. The configuration is fixed at DDL time because
    -- a generated column requires an immutable expression.
    content_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('{{.TextSearchConfig}}', content)) STORED,

    CONSTRAINT episodes_pkey          PRIMARY KEY (id),
    CONSTRAINT episodes_scope_fkey    FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),
    CONSTRAINT episodes_scope_seq_key UNIQUE (scope_id, seq),
    CONSTRAINT episodes_seq_positive  CHECK (seq > 0),
    CONSTRAINT episodes_content_len   CHECK (length(content) BETWEEN 1 AND 1048576),
    CONSTRAINT episodes_role_valid    CHECK (role IN ('user', 'assistant', 'system', 'tool', 'observation')),
    CONSTRAINT episodes_source_len    CHECK (length(source) BETWEEN 1 AND 100),
    CONSTRAINT episodes_actor_len     CHECK (length(actor) <= 200),
    -- Binding is flat string-to-string, and the schema says so as well as the
    -- Go type. A CHECK cannot contain a subquery, so this is asserted with
    -- jsonpath: strict mode matters, because lax mode unwraps arrays and would
    -- let {"a": ["x"]} pass as strings. Both functions are IMMUTABLE, which is
    -- what a CHECK requires.
    CONSTRAINT episodes_binding_flat  CHECK (
        jsonb_typeof(binding) = 'object'
        AND NOT jsonb_path_exists(binding, 'strict $.* ? (@.type() != "string")')
    ),
    CONSTRAINT episodes_binding_keys  CHECK (
        jsonb_array_length(jsonb_path_query_array(binding, 'strict $.keyvalue()')) <= 32
    )
);

-- Replay order for consolidation, and the as-of cut for the lexical channel.
CREATE INDEX episodes_scope_ingested_idx
    ON mempher.episodes (scope_id, ingested_at DESC, seq DESC);

-- Event-time windows.
CREATE INDEX episodes_scope_occurred_idx
    ON mempher.episodes (scope_id, occurred_at DESC);

-- Composite so scope selectivity is available inside the index rather than as a
-- filter over every lexical match in the database. This is what btree_gin buys.
CREATE INDEX episodes_scope_tsv_idx
    ON mempher.episodes USING gin (scope_id, content_tsv);

-- jsonb_path_ops is half the size of the default operator class and supports
-- @>, which is the only binding operator this library uses.
CREATE INDEX episodes_binding_idx
    ON mempher.episodes USING gin (binding jsonb_path_ops);


-- The invariant, enforced by the database rather than by discipline in Go.
--
-- L0 being the only source of truth is what makes a bad extractor recoverable
-- and forgetting safe, and it holds only if nothing ever rewrites history --
-- including this library, including a hotfix at 3am, including a well-meaning
-- UPDATE that looks like it only patches one field. A statement-level trigger
-- fires even when the statement would have matched no rows, so an attempt fails
-- loudly instead of silently succeeding against an empty set.
--
-- Corrections are appended. UPDATE and TRUNCATE are refused, unconditionally.
-- DELETE has one exception, because a right to erasure has to be honoured: it
-- is permitted inside a transaction that has said, in that transaction, that
-- it is erasing. mempher.forget is the function that says so. The friction is
-- attached to the statement doing the erasing rather than to the table, so the
-- guard never has to come off for the sessions that are appending meanwhile.
--
-- This is a guard against accident, not a permission boundary. Anything that
-- can set the flag could also drop the trigger; what it buys is that no
-- hotfix, no ORM, and no well-meaning cleanup script deletes from L0 without
-- saying so first, and that the saying is visible in the statement.
--
-- current_setting's second argument makes an unset flag NULL rather than an
-- error, so the overwhelmingly common case -- nobody is erasing anything -- is
-- the cheap path and the refusing path.
CREATE FUNCTION mempher.reject_l0_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    -- Statement-level BEFORE triggers ignore their return value, so this
    -- permits the statement by declining to raise.
    IF TG_OP = 'DELETE' AND current_setting('mempher.erasing', true) = 'on' THEN
        RETURN NULL;
    END IF;

    RAISE EXCEPTION
        'mempher.episodes is append-only: % is not permitted', TG_OP
        USING HINT =
            'append a correcting episode, rebuild the projection from L0, or '
            'erase through mempher.Memory.Forget, which declares itself';
END $$;

COMMENT ON FUNCTION mempher.reject_l0_mutation() IS
    'Guards the append-only invariant on mempher.episodes. A DELETE is permitted '
    'only inside a transaction that has SET LOCAL mempher.erasing = ''on''; '
    'UPDATE and TRUNCATE are never permitted.';

CREATE TRIGGER episodes_append_only
    BEFORE UPDATE OR DELETE OR TRUNCATE ON mempher.episodes
    FOR EACH STATEMENT EXECUTE FUNCTION mempher.reject_l0_mutation();


-- ---------------------------------------------------------------------------
-- Projection: vector encodings
-- ---------------------------------------------------------------------------

-- Derived from L0 and disposable: TRUNCATE this table and workers rebuild it.
-- Keyed by model so several vector spaces coexist and changing embedder is a
-- backfill, not a migration.
--
-- scope_id and ingested_at are copied from the episode row rather than passed
-- in, by an INSERT that selects them from mempher.episodes. They are here
-- because an HNSW scan cannot filter on a joined table, and they cannot drift
-- because the row they came from is immutable.
CREATE TABLE mempher.episode_encodings (
    episode_id  uuid        NOT NULL,
    model       text        NOT NULL,
    scope_id    text        NOT NULL,
    ingested_at timestamptz NOT NULL,
    content_vec vector({{.VectorDimensions}}) NOT NULL,

    -- How novel the episode was against what the scope already held. Nothing
    -- computes it yet, so it is zero unless a caller of put_encodings supplies
    -- it. It sits here rather than on
    -- episodes because it needs the vector, which does not exist at write time,
    -- and an episode is never updated.
    surprise    real        NOT NULL DEFAULT 0 CHECK (surprise >= 0 AND surprise <= 1),
    encoded_at  timestamptz NOT NULL,

    CONSTRAINT episode_encodings_pkey       PRIMARY KEY (episode_id, model),
    CONSTRAINT episode_encodings_ep_fkey    FOREIGN KEY (episode_id) REFERENCES mempher.episodes (id),
    CONSTRAINT episode_encodings_scope_fkey FOREIGN KEY (scope_id)   REFERENCES mempher.scopes (id),
    CONSTRAINT episode_encodings_model_len  CHECK (length(model) BETWEEN 1 AND 200)
);

CREATE INDEX episode_encodings_ann_idx
    ON mempher.episode_encodings USING hnsw (content_vec vector_cosine_ops);

-- Backs both the scope/as-of narrowing around an ANN scan and the anti-join
-- that finds episodes still lacking an encoding.
CREATE INDEX episode_encodings_scope_idx
    ON mempher.episode_encodings (model, scope_id, ingested_at DESC);


-- ---------------------------------------------------------------------------
-- Job queue
-- ---------------------------------------------------------------------------

-- Deferred work always names the episodes it derives from, so there is no
-- opaque payload column: L0 stays the only input to any projection, and no
-- jsonb blob is waiting to become an interface{}. PostgreSQL has no array
-- foreign key, so nothing ties episode_ids to mempher.episodes: mempher.forget
-- removes the jobs of the episodes it erases itself, in the same transaction.
CREATE TABLE mempher.jobs (
    id           uuid        NOT NULL DEFAULT uuidv7(),
    kind         text        NOT NULL,
    scope_id     text        NOT NULL,
    episode_ids  uuid[]      NOT NULL,
    state        text        NOT NULL DEFAULT 'pending',
    attempts     int         NOT NULL DEFAULT 0,
    max_attempts int         NOT NULL DEFAULT 5,
    run_after    timestamptz NOT NULL,
    leased_until timestamptz,
    leased_by    text,
    last_error   text,
    created_at   timestamptz NOT NULL,
    updated_at   timestamptz NOT NULL,

    CONSTRAINT jobs_pkey          PRIMARY KEY (id),
    CONSTRAINT jobs_scope_fkey    FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),
    -- The kind set is closed, so adding one is a migration: a binary cannot
    -- enqueue work that the binaries deployed beside it cannot drain. The
    -- kinds are separate, rather than one that consolidates, because each
    -- projection fails, retries and is backfilled on its own: a change of
    -- embedding model reruns every encode and no extraction, and a prompt fix
    -- reruns the reverse.
    CONSTRAINT jobs_kind_valid    CHECK (kind IN ('encode', 'extract', 'encode_facts')),
    CONSTRAINT jobs_state_valid   CHECK (state IN ('pending', 'running', 'done', 'dead')),
    CONSTRAINT jobs_episodes_set  CHECK (cardinality(episode_ids) > 0),
    CONSTRAINT jobs_attempts_ok   CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts),
    -- A lease exists exactly while the job is running, so a crashed worker
    -- cannot leave a job that looks claimable and claimed at once.
    CONSTRAINT jobs_lease_ok      CHECK (
        (state = 'running') = (leased_until IS NOT NULL AND leased_by IS NOT NULL)
    )
);

-- The lease query: ready jobs of a kind, oldest first, taken with
-- FOR UPDATE SKIP LOCKED.
CREATE INDEX jobs_ready_idx
    ON mempher.jobs (kind, run_after, id) WHERE state = 'pending';

-- Reclaiming leases a dead worker still holds.
CREATE INDEX jobs_expiring_idx
    ON mempher.jobs (leased_until) WHERE state = 'running';

-- Enqueueing the same work twice while it is still outstanding collapses onto
-- the existing job. Done and dead rows are excluded so the same episode can be
-- re-encoded later, after a model change.
CREATE UNIQUE INDEX jobs_inflight_key
    ON mempher.jobs (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running');

-- Finding work by what happened to it, across every scope: the dead-job hunt.
-- Ordering by id is ordering by creation time, because the ids are uuidv7, so
-- one column serves both the filter and the cursor that pages it.
--
-- It is a full index rather than one partial on 'dead'. A partial index would be
-- a fraction of the size and would leave a listing of pending or running work
-- with no path at all, which is the same blind spot moved somewhere quieter.
-- The cost of keeping it whole is one more index to maintain on every state
-- change, and a size that tracks the table's history -- which is what
-- mempher.purge_jobs is for.
CREATE INDEX jobs_state_idx
    ON mempher.jobs (state, id);

-- Finding work by whose memory it belongs to.
--
-- "Why has this user's memory not caught up?" is the question a queue is
-- actually asked, and answering it means every job of one scope regardless of
-- state, oldest first. State is deliberately not in this key: a scope holds few
-- enough jobs that scanning them beats maintaining a wider index, and the
-- selective half of the predicate is the scope.
CREATE INDEX jobs_scope_idx
    ON mempher.jobs (scope_id, id);

COMMENT ON INDEX mempher.jobs_state_idx IS
    'Lists jobs by lifecycle state across scopes; the dead-job hunt.';

COMMENT ON INDEX mempher.jobs_scope_idx IS
    'Lists one scope''s jobs in creation order, whatever state they are in.';


-- ---------------------------------------------------------------------------
-- L1: facts. Derived from L0, disposable, and bounded in event time.
-- ---------------------------------------------------------------------------

-- A fact is a triple with a rendered statement beside it, and the two halves do
-- different jobs. The triple is identity: it is what makes the same claim
-- extracted twice one row rather than two, and what a retraction names. The
-- statement is the payload: it is what goes in front of a model, and the only
-- column the text index sees.
--
-- valid is event time -- when the claim held in the world. asserted_at is system
-- time -- when this library learned it. Keeping them apart is the whole point of
-- the table: "lived in Paris until March" and "we found this out on Tuesday" are
-- different questions, and a store that conflates them answers the first with
-- the second.
--
-- What is deliberately NOT here is belief history: the row says what is
-- currently believed about the past, not what was believed at some past instant.
-- Retraction moves valid in place rather than closing one row and opening
-- another. That keeps the temporal key exact -- WITHOUT OVERLAPS cannot be made
-- partial, so superseded rows in this table would collide with live ones -- and
-- it is honest about which layer is reproducible: L0 is, a projection is
-- current. Full bitemporality is an additive migration if it is ever wanted.
CREATE TABLE mempher.facts (
    -- A surrogate key beside the real one. The temporal primary key below moves
    -- when a window is closed, so a retraction needs something that does not:
    -- this is the handle an Extractor is given and hands back.
    id          uuid        NOT NULL DEFAULT uuidv7(),

    scope_id    text        NOT NULL,

    -- Facts are keyed by extractor for the same reason encodings are keyed by
    -- model: two extractors' opinions must not silently merge into one scope,
    -- and changing extractor must be a backfill rather than a migration.
    extractor   text        NOT NULL,

    subject     text        NOT NULL,
    predicate   text        NOT NULL,
    object      text        NOT NULL,
    statement   text        NOT NULL,

    -- Half-open, [from, to): a fact ending at noon and one starting at noon do
    -- not overlap, so a change of state needs no gap and no tie-break. An
    -- unbounded upper end means still true, which is the common case.
    valid       tstzrange   NOT NULL,

    confidence  real        NOT NULL DEFAULT 1 CHECK (confidence >= 0 AND confidence <= 1),

    -- Provenance: the episodes this claim was read from, and the first thing
    -- asked of a fact that looks wrong. PostgreSQL has no array foreign key, so
    -- mempher.forget removes the facts of the episodes it erases itself.
    episode_ids uuid[]      NOT NULL,

    asserted_at timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,

    -- Generated, like episodes.content_tsv, so a fact is searchable the instant
    -- it lands and no second projection hangs off this one. Only the statement
    -- is indexed: the triple is machine identity, and matching a query against
    -- "allergic_to" would rank on a slug the caller never wrote.
    statement_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('{{.TextSearchConfig}}', statement)) STORED,

    -- The identity of a fact is what it claims and when it held. WITHOUT
    -- OVERLAPS makes the database reject a scope that believes the same triple
    -- over two overlapping windows -- the exact corruption an extractor
    -- re-reading an episode would otherwise cause -- instead of leaving recall
    -- to return the claim twice.
    --
    -- object is part of the key so that being allergic to two things is two
    -- facts rather than one overwriting the other. Contradiction between
    -- different objects of one predicate is not a constraint violation and
    -- cannot be: only the extractor knows whether "lives_in Berlin" replaces
    -- "lives_in Paris" or joins it, so it says so with a retraction.
    CONSTRAINT facts_pkey PRIMARY KEY
        (scope_id, extractor, subject, predicate, object, valid WITHOUT OVERLAPS),

    CONSTRAINT facts_id_key         UNIQUE (id),
    CONSTRAINT facts_scope_fkey     FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),

    -- These lengths are not arbitrary. Everything in the primary key above ends
    -- up in one GiST index tuple, which has a hard size limit, so the five key
    -- columns are sized to fit inside it together with the range and its
    -- overhead. Mirrors the limits in the Go package.
    CONSTRAINT facts_extractor_len  CHECK (length(extractor) BETWEEN 1 AND 200),
    CONSTRAINT facts_subject_len    CHECK (length(subject)   BETWEEN 1 AND 200),
    CONSTRAINT facts_predicate_len  CHECK (length(predicate) BETWEEN 1 AND 100),
    CONSTRAINT facts_object_len     CHECK (length(object)    BETWEEN 1 AND 1000),
    CONSTRAINT facts_statement_len  CHECK (length(statement) BETWEEN 1 AND 4000),

    -- A predicate is the key that makes two extractions of one claim the same
    -- claim, so it may not carry whitespace: "lives_in" and "lives in" must not
    -- become two relations.
    CONSTRAINT facts_predicate_slug CHECK (predicate !~ '\s'),

    CONSTRAINT facts_episodes_set   CHECK (cardinality(episode_ids) > 0),

    -- A window must begin. An unbounded lower end would place the claim before
    -- every episode that could have produced it, and empty holds at no instant
    -- at all, which is a mistake rather than a way to say "never true".
    CONSTRAINT facts_valid_bounded  CHECK (NOT lower_inf(valid) AND NOT isempty(valid)),
    -- '[)' is asserted, not assumed, so a caller building the range by hand
    -- cannot introduce an inclusive upper bound that makes two adjacent windows
    -- overlap by exactly one instant.
    CONSTRAINT facts_valid_halfopen CHECK (
        lower_inc(valid) AND (upper_inf(valid) OR NOT upper_inc(valid))
    )
);

-- The read path: everything believed about one scope, by one extractor, valid at
-- an instant. The GiST primary key can serve this, but a btree on the equality
-- prefix answers the common query without touching it.
CREATE INDEX facts_scope_lookup_idx
    ON mempher.facts (scope_id, extractor, subject, predicate);

-- The system-time cut, which bounds facts to those already learned.
CREATE INDEX facts_scope_asserted_idx
    ON mempher.facts (scope_id, extractor, asserted_at DESC);

-- Relevance ordering when a scope holds more facts than the budget fits.
-- Composite for the same reason as episodes_scope_tsv_idx: scope selectivity
-- belongs inside the index, not as a filter over every match in the database.
CREATE INDEX facts_scope_tsv_idx
    ON mempher.facts USING gin (scope_id, statement_tsv);


-- ---------------------------------------------------------------------------
-- Projection bookkeeping: which episodes an extractor has read
-- ---------------------------------------------------------------------------

-- Most episodes yield no durable fact, so "has this been extracted?" cannot be
-- answered from mempher.facts: an episode with no facts and an episode never
-- read look identical there. This table is the difference, and without it a
-- scope full of small talk is re-extracted, at model prices, on every pass.
--
-- scope_id and ingested_at are copied from the episode row by an INSERT that
-- selects them, exactly as in mempher.episode_encodings, and for the same two
-- reasons: they make the pending anti-join a single-table scan, and they cannot
-- drift because the row they came from is immutable.
CREATE TABLE mempher.fact_extractions (
    episode_id   uuid        NOT NULL,
    extractor    text        NOT NULL,
    scope_id     text        NOT NULL,
    ingested_at  timestamptz NOT NULL,
    -- How many facts the extraction that read this episode asserted, including
    -- zero. Cheap to record and the first number to look at when an extractor is
    -- suspected of finding something durable in every "ok, thanks". It counts
    -- the extraction, not the episode, so a job covering several episodes
    -- records the same total against each.
    fact_count   int         NOT NULL DEFAULT 0 CHECK (fact_count >= 0),
    extracted_at timestamptz NOT NULL,

    CONSTRAINT fact_extractions_pkey       PRIMARY KEY (episode_id, extractor),
    CONSTRAINT fact_extractions_ep_fkey    FOREIGN KEY (episode_id) REFERENCES mempher.episodes (id),
    CONSTRAINT fact_extractions_scope_fkey FOREIGN KEY (scope_id)   REFERENCES mempher.scopes (id),
    CONSTRAINT fact_extractions_extr_len   CHECK (length(extractor) BETWEEN 1 AND 200)
);

-- Backs the anti-join that finds episodes an extractor has not read yet.
CREATE INDEX fact_extractions_scope_idx
    ON mempher.fact_extractions (extractor, scope_id, ingested_at DESC);


-- ---------------------------------------------------------------------------
-- Projection: vector encodings of facts
-- ---------------------------------------------------------------------------

-- Keyed by (fact, model) like episode_encodings, so a change of model is new
-- rows rather than an UPDATE. A row here always encodes the statement the fact
-- holds now: put_fact_encodings refuses a vector made from an older statement,
-- and the triggers below remove the vector when the statement changes or the
-- fact goes.
--
-- There is no ANN index. A fact search is confined to one scope and one
-- extractor, and already reads every fact that is true at the instant asked
-- about, because the question ranks facts and never filters them. The distance
-- is computed over that same set, exactly.
--
-- There is no foreign key to mempher.facts either, and that is deliberate: L1
-- is repaired by truncating it and replaying L0, and PostgreSQL refuses to
-- truncate a table a foreign key points at. The triggers give what the key
-- would have: no vector outlives its fact.
CREATE TABLE mempher.fact_encodings (
    fact_id       uuid        NOT NULL,
    model         text        NOT NULL,
    statement_vec vector({{.VectorDimensions}}) NOT NULL,
    encoded_at    timestamptz NOT NULL,

    CONSTRAINT fact_encodings_pkey      PRIMARY KEY (fact_id, model),
    CONSTRAINT fact_encodings_model_len CHECK (length(model) BETWEEN 1 AND 200)
);

CREATE FUNCTION mempher._drop_fact_encodings()
    RETURNS trigger
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        TRUNCATE mempher.fact_encodings;
    ELSE
        DELETE FROM mempher.fact_encodings WHERE fact_id = OLD.id;
    END IF;
    RETURN NULL;
END $$;

-- A re-asserted claim may arrive with a reworded statement. The old vector
-- then describes text the fact no longer holds, and the fact is pending again.
CREATE TRIGGER facts_statement_changed
    AFTER UPDATE OF statement ON mempher.facts
    FOR EACH ROW
    WHEN (OLD.statement IS DISTINCT FROM NEW.statement)
    EXECUTE FUNCTION mempher._drop_fact_encodings();

-- Erasure, by mempher.forget or by hand.
CREATE TRIGGER facts_deleted
    AFTER DELETE ON mempher.facts
    FOR EACH ROW
    EXECUTE FUNCTION mempher._drop_fact_encodings();

-- Dropping the projection to rebuild it.
CREATE TRIGGER facts_truncated
    AFTER TRUNCATE ON mempher.facts
    FOR EACH STATEMENT
    EXECUTE FUNCTION mempher._drop_fact_encodings();


-- ---------------------------------------------------------------------------
-- L1: entities. What facts are about, and every name each goes by.
-- ---------------------------------------------------------------------------

-- A triple is identity only while its subject is spelled one way. "Anna" in one
-- episode and "the user's sister" in the next are two subjects to the temporal
-- key above, so nothing dedupes, nothing supersedes, and a question about one
-- misses what is known under the other. An entity is the thing behind the
-- names, and mempher.entity_aliases is every name it has been given.
--
-- The rule the rest of L1 leans on: mempher.facts.subject is always the name of
-- an entity, never one of its other aliases. apply_extraction rewrites a
-- subject to that name before it writes, and merging two entities rewrites the
-- facts of the one that goes. So the temporal key, the subject filters and the
-- known facts shown to an extractor all see one spelling per thing, and no
-- reader has to resolve a name to get a consistent answer.
--
-- Objects are left as written. "the user's sister, name, Anna" must keep its
-- object, so an object is linked to an entity only when it is read: whenever
-- its text is one of an entity's names, the fact is an edge between its subject
-- and that entity, which is what mempher.related_facts walks.
--
-- Entities are keyed by extractor, as facts are: two extractors' opinions of
-- who is who must not merge either. They are not removed when mempher.facts is
-- truncated, because an alias may have been declared by hand and no replay
-- would bring it back; truncate mempher.entities CASCADE alongside to rebuild
-- the names as well.

-- The form in which two spellings of a name are the same name: case and runs
-- of whitespace do not distinguish them. Nothing cleverer happens here. Deciding
-- that "Anna" and "Anna Martin" are one person is a judgement, and judgements
-- are declared, by an extractor or an operator, not guessed by a function.
CREATE FUNCTION mempher._entity_key(p_name text)
    RETURNS text
    LANGUAGE sql
    IMMUTABLE
    PARALLEL SAFE
    SET search_path FROM CURRENT
RETURN lower(btrim(regexp_replace(p_name, '\s+', ' ', 'g')));

CREATE TABLE mempher.entities (
    -- The handle a client holds. It survives the entity gaining names, and is
    -- gone once the entity has been merged into another.
    id          uuid        NOT NULL DEFAULT uuidv7(),
    scope_id    text        NOT NULL,
    extractor   text        NOT NULL,

    -- The spelling its facts carry as their subject: the first one written.
    name        text        NOT NULL,

    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,

    CONSTRAINT entities_pkey          PRIMARY KEY (id),
    CONSTRAINT entities_scope_fkey    FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),
    CONSTRAINT entities_extractor_len CHECK (length(extractor) BETWEEN 1 AND 200),
    CONSTRAINT entities_name_len      CHECK (length(name) BETWEEN 1 AND 200)
);

-- Lists a scope's entities by name.
CREATE INDEX entities_scope_idx
    ON mempher.entities (scope_id, extractor, name);

-- One row per name, the entity's own included. The primary key is the
-- resolution: within a scope and an extractor a name leads to exactly one
-- entity, so merging two entities is moving rows here, and there is no chain of
-- redirections for a reader to follow.
CREATE TABLE mempher.entity_aliases (
    scope_id    text        NOT NULL,
    extractor   text        NOT NULL,
    key         text        NOT NULL,
    -- The name as it was first written, which is what a reader is shown.
    alias       text        NOT NULL,
    entity_id   uuid        NOT NULL,

    -- The episodes of the extraction that introduced the name, so that erasing
    -- them takes the name with them. Empty for a name an operator declared.
    episode_ids uuid[]      NOT NULL DEFAULT '{}',

    created_at  timestamptz NOT NULL,

    CONSTRAINT entity_aliases_pkey        PRIMARY KEY (scope_id, extractor, key),
    CONSTRAINT entity_aliases_entity_fkey FOREIGN KEY (entity_id)
        REFERENCES mempher.entities (id) ON DELETE CASCADE,
    CONSTRAINT entity_aliases_key         CHECK (key <> '' AND key = mempher._entity_key(alias)),
    CONSTRAINT entity_aliases_alias_len   CHECK (length(alias) BETWEEN 1 AND 200)
);

-- Every name of one entity: listing them, and moving them in a merge.
CREATE INDEX entity_aliases_entity_idx
    ON mempher.entity_aliases (entity_id);


-- ===========================================================================
-- The SQL API
-- ===========================================================================

-- From here on the database is the implementation and every client is a caller:
-- the Go package, a Python service and psql all reach the same functions, and
-- every rule a row must satisfy is checked here, by the function that writes it,
-- before the CHECK constraints above get a chance to say it less clearly. The
-- only work left outside is what a database cannot do: calling the embedding
-- and extraction models.
--
-- Conventions every function follows:
--
--   * A public function is documented with COMMENT ON FUNCTION and is part of
--     the contract. A function whose name starts with an underscore is not, and
--     may change in any migration.
--
--   * Every function is SECURITY INVOKER and pins its search_path to the one
--     the migration ran under -- the extension schema, then mempher -- so an
--     unqualified vector type or btree_gin operator resolves the same way
--     whoever calls it, and an object in a caller's schema cannot shadow one
--     of ours.
--
--   * No function reads a clock it was not handed. Every timestamp a row keeps
--     is a parameter, defaulting to now() for interactive callers, which is
--     what keeps a replay byte-identical.
--
--   * A caller's mistake raises one of the SQLSTATEs below, in class MP. The
--     message is for a human; the code is the contract.
--
--     MP001 invalid scope id            MP011 invalid validity window
--     MP002 invalid episode content     MP012 invalid extraction
--     MP003 invalid role                MP013 invalid argument
--     MP004 invalid source              MP020 not found
--     MP005 invalid actor               MP021 vector dimension mismatch
--     MP006 invalid binding             MP022 fact windows conflict
--     MP007 invalid channel             MP023 job not leased by this worker
--     MP008 invalid job kind            MP024 job is not dead
--     MP009 invalid job state           MP025 the same work is already queued
--     MP010 invalid fact                MP014 invalid entity name

-- ---------------------------------------------------------------------------
-- Internal helpers
-- ---------------------------------------------------------------------------

-- Raises a caller error. A function rather than a RAISE at each site so the
-- class stays in one place and every message reads the same way.
CREATE FUNCTION mempher._fail(p_code text, p_message text, p_hint text DEFAULT NULL)
    RETURNS void
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_hint IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = p_code, MESSAGE = p_message;
    END IF;
    RAISE EXCEPTION USING ERRCODE = p_code, MESSAGE = p_message, HINT = p_hint;
END $$;

-- A scope id is 1 to 200 bytes and holds no control character: it ends up in
-- log lines and operator queries, where an embedded newline is at best
-- confusing. The range covers C0, DEL and C1, which is what Go's
-- unicode.IsControl means by a control character.
CREATE FUNCTION mempher._check_scope(p_scope text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_scope IS NULL OR p_scope = '' THEN
        PERFORM mempher._fail('MP001', 'scope id is empty');
    ELSIF octet_length(p_scope) > 200 THEN
        PERFORM mempher._fail('MP001', format(
            'scope id is %s bytes, limit is 200', octet_length(p_scope)));
    ELSIF p_scope ~ '[\x01-\x1f\x7f-\x9f]' THEN
        PERFORM mempher._fail('MP001', 'scope id holds a control character');
    END IF;
END $$;

-- A binding is a flat object of at most 32 string entries, each key 1 to 64
-- bytes and each value at most 512. NULL is not a binding; the caller means
-- '{}' and is told so rather than guessed at.
CREATE FUNCTION mempher._check_binding(p_binding jsonb, p_where text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad text;
BEGIN
    IF p_binding IS NULL OR jsonb_typeof(p_binding) <> 'object' THEN
        PERFORM mempher._fail('MP006', format('%s: binding is not a JSON object', p_where));
    END IF;
    IF (SELECT count(*) FROM jsonb_object_keys(p_binding)) > 32 THEN
        PERFORM mempher._fail('MP006', format(
            '%s: binding has %s keys, limit is 32',
            p_where, (SELECT count(*) FROM jsonb_object_keys(p_binding))));
    END IF;
    SELECT key INTO v_bad
    FROM jsonb_each(p_binding)
    WHERE jsonb_typeof(value) <> 'string'
       OR octet_length(key) NOT BETWEEN 1 AND 64
       OR octet_length(value #>> '{}') > 512
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP006', format(
            '%s: binding entry %L must be a string of at most 512 bytes under a key of 1 to 64 bytes',
            p_where, v_bad));
    END IF;
END $$;

-- One element of mempher.append's p_episodes. p_where names it in a message,
-- because "episode 37 of 64" is what a caller needs to find it.
CREATE FUNCTION mempher._check_new_episode(p_episode jsonb, p_where text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_unknown text;
BEGIN
    IF jsonb_typeof(p_episode) IS DISTINCT FROM 'object' THEN
        PERFORM mempher._fail('MP013', format('%s is not a JSON object', p_where));
    END IF;
    -- An unknown key is a typo, and a typo such as "occurredAt" would otherwise
    -- quietly date the episode to now.
    SELECT k INTO v_unknown
    FROM jsonb_object_keys(p_episode) AS k
    WHERE k NOT IN ('content', 'role', 'actor', 'source', 'occurred_at', 'binding')
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format('%s has unknown key %L', p_where, v_unknown),
            'an episode takes content, role, actor, source, occurred_at and binding');
    END IF;

    IF jsonb_typeof(p_episode->'content') IS DISTINCT FROM 'string'
       OR p_episode->>'content' = '' THEN
        PERFORM mempher._fail('MP002', format('%s: content is empty', p_where));
    ELSIF octet_length(p_episode->>'content') > 1048576 THEN
        PERFORM mempher._fail('MP002', format(
            '%s: content is %s bytes, limit is 1048576',
            p_where, octet_length(p_episode->>'content')));
    END IF;

    IF jsonb_typeof(p_episode->'role') IS DISTINCT FROM 'string'
       OR p_episode->>'role' NOT IN ('user', 'assistant', 'system', 'tool', 'observation') THEN
        PERFORM mempher._fail('MP003', format('%s: role %s is not one of user, assistant, system, tool, observation',
            p_where, coalesce(p_episode->'role', 'null'::jsonb)));
    END IF;

    IF p_episode ? 'actor' AND (jsonb_typeof(p_episode->'actor') <> 'string'
       OR octet_length(p_episode->>'actor') > 200) THEN
        PERFORM mempher._fail('MP005', format('%s: actor must be a string of at most 200 bytes', p_where));
    END IF;

    IF jsonb_typeof(p_episode->'source') IS DISTINCT FROM 'string'
       OR octet_length(p_episode->>'source') NOT BETWEEN 1 AND 100 THEN
        PERFORM mempher._fail('MP004', format('%s: source must be a string of 1 to 100 bytes', p_where));
    END IF;

    IF p_episode ? 'occurred_at' AND (jsonb_typeof(p_episode->'occurred_at') <> 'string'
       OR NOT pg_input_is_valid(p_episode->>'occurred_at', 'timestamptz')) THEN
        PERFORM mempher._fail('MP013', format(
            '%s: occurred_at %s is not a timestamp', p_where, p_episode->'occurred_at'));
    END IF;

    IF p_episode ? 'binding' THEN
        PERFORM mempher._check_binding(p_episode->'binding', p_where);
    END IF;
END $$;

-- The job kinds a write may imply, deduplicated in the order given. Unknown
-- kinds are rejected here rather than by the CHECK on mempher.jobs, which would
-- report a constraint name instead of the kind.
CREATE FUNCTION mempher._check_job_kinds(p_kinds text[])
    RETURNS text[]
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad text;
BEGIN
    IF p_kinds IS NULL THEN
        RETURN '{}';
    END IF;
    SELECT k INTO v_bad FROM unnest(p_kinds) AS k
    WHERE k IS NULL OR k NOT IN ('encode', 'extract', 'encode_facts') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP008', format(
            'job kind %L is not one of encode, extract, encode_facts', v_bad));
    END IF;
    RETURN ARRAY(
        SELECT k FROM unnest(p_kinds) WITH ORDINALITY AS u(k, ord)
        GROUP BY k ORDER BY min(ord));
END $$;


-- ---------------------------------------------------------------------------
-- mempher.append
-- ---------------------------------------------------------------------------

CREATE FUNCTION mempher.append(
    p_scope        text,
    p_episodes     jsonb,
    p_at           timestamptz DEFAULT now(),
    p_jobs         text[]      DEFAULT ARRAY['encode'],
    p_max_attempts int         DEFAULT 5
)
    RETURNS TABLE (id uuid, seq bigint)
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
DECLARE
    v_n     int;
    v_kinds text[];
    v_base  bigint;
    v_ids   uuid[];
    v_seqs  bigint[];
    v_elem  jsonb;
    v_ord   bigint;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    IF p_at IS NULL THEN
        PERFORM mempher._fail('MP013', 'at is null');
    END IF;
    IF jsonb_typeof(p_episodes) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP013', 'episodes is not a JSON array');
    END IF;
    v_n := jsonb_array_length(p_episodes);
    IF v_n NOT BETWEEN 1 AND 64 THEN
        PERFORM mempher._fail('MP013', format('append holds %s episodes, must be 1 to 64', v_n),
            'a batch is one round trip and one encode job; split a longer one');
    END IF;
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;
    v_kinds := mempher._check_job_kinds(p_jobs);

    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_episodes) WITH ORDINALITY AS a(e, o)
    LOOP
        PERFORM mempher._check_new_episode(v_elem, format('episode %s of %s', v_ord, v_n));
    END LOOP;

    -- Allocating the sequence and creating the scope are one upsert, so seq is
    -- dense and gap-free without a second round trip or an explicit lock.
    -- Concurrent appends to one scope serialise on this row, which is the
    -- right semantics for a totally ordered log.
    INSERT INTO mempher.scopes AS s (id, last_seq, created_at)
    VALUES (p_scope, v_n, p_at)
    ON CONFLICT (id) DO UPDATE SET last_seq = s.last_seq + v_n
    RETURNING s.last_seq - v_n INTO v_base;

    WITH inserted AS (
        INSERT INTO mempher.episodes
            (scope_id, seq, content, role, actor, source, occurred_at, ingested_at, binding)
        SELECT p_scope,
               v_base + a.o,
               a.e->>'content',
               a.e->>'role',
               coalesce(a.e->>'actor', ''),
               a.e->>'source',
               coalesce((a.e->>'occurred_at')::timestamptz, p_at),
               p_at,
               coalesce(a.e->'binding', '{}'::jsonb)
        FROM jsonb_array_elements(p_episodes) WITH ORDINALITY AS a(e, o)
        RETURNING id, seq
    )
    SELECT array_agg(i.id ORDER BY i.seq), array_agg(i.seq ORDER BY i.seq)
    INTO v_ids, v_seqs
    FROM inserted i;

    -- One job per kind for the whole batch: a turn that produced several
    -- episodes is embedded in one model call.
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    SELECT k, p_scope, v_ids, p_at, p_max_attempts, p_at, p_at
    FROM unnest(v_kinds) AS k
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING;

    RETURN QUERY SELECT u.id, u.seq FROM unnest(v_ids, v_seqs) AS u(id, seq);
END $$;

COMMENT ON FUNCTION mempher.append(text, jsonb, timestamptz, text[], int) IS
    'Appends 1 to 64 episodes to a scope, atomically, with the jobs they imply. '
    'p_episodes is a JSON array of objects: content, role and source are required; '
    'actor, occurred_at (defaults to p_at) and binding (a flat object of strings) are optional. '
    'p_at is the ingest time. p_jobs names the projections to queue, one job each for '
    'the whole batch: encode, and extract when L1 is on. Returns each episode''s id and seq, in order.';


-- ---------------------------------------------------------------------------
-- The job queue
-- ---------------------------------------------------------------------------

-- The queue's whole state machine is these functions: claiming with SKIP
-- LOCKED, the lease a worker must still hold to finish, the backoff between
-- attempts, the dead state an exhausted job lands in, and reclaiming a dead
-- worker's leases. A worker in any language is a loop around lease_jobs and one
-- of succeed_job or fail_job; it decides nothing about the queue itself.
--
-- Reading the queue needs no function: mempher.jobs is a plain table, and
-- selecting from it is part of the contract.

CREATE FUNCTION mempher._check_worker(p_worker text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_worker IS NULL OR p_worker = '' THEN
        PERFORM mempher._fail('MP013', 'worker is empty',
            'name the worker, so a lease says who holds it');
    END IF;
END $$;

CREATE FUNCTION mempher._check_at(p_at timestamptz, p_name text DEFAULT 'at')
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_at IS NULL OR NOT isfinite(p_at) THEN
        PERFORM mempher._fail('MP013', format('%s is %s, must be a finite timestamp',
            p_name, coalesce(p_at::text, 'null')));
    END IF;
END $$;

-- Says why a job the caller named could not be moved: it does not exist, or it
-- is not in the state the caller assumed. Costs a read, but only on a path that
-- is about to be investigated anyway.
CREATE FUNCTION mempher._explain_lost_lease(p_job uuid, p_worker text, p_op text)
    RETURNS void
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_state text;
    v_by    text;
BEGIN
    SELECT state, leased_by INTO v_state, v_by FROM mempher.jobs WHERE id = p_job;
    IF NOT FOUND THEN
        PERFORM mempher._fail('MP020', format('%s job %s: no such job', p_op, p_job));
    END IF;
    PERFORM mempher._fail('MP023', format(
        '%s job %s: worker %L does not hold it (state %s, held by %L)',
        p_op, p_job, p_worker, v_state, coalesce(v_by, '')),
        'the lease expired and the job was reclaimed, or another worker finished it');
END $$;

-- Writing to the queue.

CREATE FUNCTION mempher.enqueue(
    p_kind         text,
    p_scope        text,
    p_episodes     uuid[],
    p_at           timestamptz DEFAULT now(),
    p_run_after    timestamptz DEFAULT NULL,
    p_max_attempts int         DEFAULT 5
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_job_kinds(ARRAY[p_kind]);
    PERFORM mempher._check_at(p_at);
    IF p_episodes IS NULL OR cardinality(p_episodes) = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'a job must name at least one episode, and no null');
    END IF;
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM mempher.scopes WHERE id = p_scope) THEN
        PERFORM mempher._fail('MP020', format('scope %L has no episodes yet', p_scope));
    END IF;

    -- Duplicate work collapses onto the job already holding it. The loop covers
    -- the one race there is: the outstanding job finishing between the
    -- conflicting insert and the read, which frees the slot for this one.
    LOOP
        INSERT INTO mempher.jobs
            (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
        VALUES (p_kind, p_scope, p_episodes, coalesce(p_run_after, p_at), p_max_attempts, p_at, p_at)
        ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
            DO NOTHING
        RETURNING * INTO v_job;
        IF FOUND THEN
            RETURN v_job;
        END IF;
        SELECT * INTO v_job FROM mempher.jobs
        WHERE kind = p_kind AND scope_id = p_scope AND episode_ids = p_episodes
          AND state IN ('pending', 'running');
        IF FOUND THEN
            RETURN v_job;
        END IF;
    END LOOP;
END $$;

COMMENT ON FUNCTION mempher.enqueue(text, text, uuid[], timestamptz, timestamptz, int) IS
    'Queues one job of a kind over episodes of a scope. Work identical to a job still '
    'pending or running collapses onto it, and that job is returned instead.';

-- Running the queue.

CREATE FUNCTION mempher.lease_jobs(
    p_worker text,
    p_limit  int         DEFAULT 1,
    p_kinds  text[]      DEFAULT NULL,
    p_lease  interval    DEFAULT '5 minutes',
    p_at     timestamptz DEFAULT now()
)
    RETURNS SETOF mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_kinds text[];
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_lease IS NULL OR p_lease <= interval '0' THEN
        PERFORM mempher._fail('MP013', format('lease is %s, must be positive', p_lease));
    END IF;
    v_kinds := mempher._check_job_kinds(p_kinds);

    -- The attempt is counted here, at lease time, rather than when it fails: a
    -- worker that dies mid-job never reports, and its attempt must still count
    -- or a job that kills its worker would be retried for ever.
    RETURN QUERY
    WITH claimed AS (
        UPDATE mempher.jobs j SET
            state        = 'running',
            attempts     = j.attempts + 1,
            leased_until = p_at + p_lease,
            leased_by    = p_worker,
            updated_at   = p_at
        WHERE j.id IN (
            SELECT c.id FROM mempher.jobs c
            WHERE c.state = 'pending' AND c.run_after <= p_at AND c.attempts < c.max_attempts
              AND (cardinality(v_kinds) = 0 OR c.kind = ANY(v_kinds))
            ORDER BY c.run_after, c.id
            LIMIT p_limit
            FOR UPDATE SKIP LOCKED)
        RETURNING j.*
    )
    SELECT * FROM claimed ORDER BY run_after, id;
END $$;

COMMENT ON FUNCTION mempher.lease_jobs(text, int, text[], interval, timestamptz) IS
    'Claims up to p_limit ready jobs for one worker, oldest first, skipping jobs other '
    'workers hold. NULL or empty kinds means any kind. Each claim counts as an attempt. Finish '
    'each with succeed_job or fail_job before p_at + p_lease, or reclaim_jobs takes it back.';

CREATE FUNCTION mempher.succeed_job(
    p_job    uuid,
    p_worker text,
    p_at     timestamptz DEFAULT now()
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    UPDATE mempher.jobs SET
        state        = 'done',
        leased_until = NULL,
        leased_by    = NULL,
        updated_at   = p_at
    WHERE id = p_job AND state = 'running' AND leased_by = p_worker
    RETURNING * INTO v_job;
    IF NOT FOUND THEN
        PERFORM mempher._explain_lost_lease(p_job, p_worker, 'succeed');
    END IF;
    RETURN v_job;
END $$;

COMMENT ON FUNCTION mempher.succeed_job(uuid, text, timestamptz) IS
    'Marks a job the worker holds done. Raises MP023 if the worker no longer holds it.';

CREATE FUNCTION mempher.fail_job(
    p_job      uuid,
    p_worker   text,
    p_error    text,
    p_at       timestamptz DEFAULT now(),
    p_retry_at timestamptz DEFAULT NULL
)
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job mempher.jobs;
BEGIN
    PERFORM mempher._check_worker(p_worker);
    PERFORM mempher._check_at(p_at);
    IF p_retry_at IS NOT NULL AND NOT isfinite(p_retry_at) THEN
        PERFORM mempher._fail('MP013', 'retry_at must be a finite timestamp');
    END IF;

    -- The attempt was counted at lease time, so a job that has used its last
    -- one goes dead rather than back to pending: the alternative is a job that
    -- retries for ever, which is how a broken extractor becomes a bill.
    --
    -- Backoff doubles from ten seconds to a ceiling of ten minutes, unless the
    -- caller knows better (a provider's Retry-After, say). The exponent is
    -- capped before it is used, so a job with a large max_attempts cannot
    -- overflow the interval.
    UPDATE mempher.jobs SET
        state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
        run_after    = CASE
                           WHEN attempts >= max_attempts THEN run_after
                           ELSE coalesce(p_retry_at,
                                p_at + least(interval '10 seconds' * power(2, least(greatest(attempts, 1) - 1, 10)),
                                             interval '10 minutes'))
                       END,
        leased_until = NULL,
        leased_by    = NULL,
        last_error   = CASE
                           WHEN length(p_error) > 2000 THEN left(p_error, 2000) || '... (truncated)'
                           ELSE p_error
                       END,
        updated_at   = p_at
    WHERE id = p_job AND state = 'running' AND leased_by = p_worker
    RETURNING * INTO v_job;
    IF NOT FOUND THEN
        PERFORM mempher._explain_lost_lease(p_job, p_worker, 'fail');
    END IF;
    RETURN v_job;
END $$;

COMMENT ON FUNCTION mempher.fail_job(uuid, text, text, timestamptz, timestamptz) IS
    'Records that the worker''s attempt at a job failed. The job returns to pending at '
    'p_retry_at, or after a backoff doubling from 10 seconds to 10 minutes when that is '
    'NULL, unless the attempt was its last, in which case it is dead. p_error is kept, '
    'truncated to 2000 characters.';

CREATE FUNCTION mempher.reclaim_jobs(p_at timestamptz DEFAULT now())
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_count int;
BEGIN
    PERFORM mempher._check_at(p_at);
    UPDATE mempher.jobs SET
        state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
        leased_until = NULL,
        leased_by    = NULL,
        last_error   = 'lease expired; the worker holding it is presumed dead',
        updated_at   = p_at
    WHERE state = 'running' AND leased_until < p_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.reclaim_jobs(timestamptz) IS
    'Releases every lease that expired before p_at and returns how many. A job with '
    'attempts left returns to pending; one without goes dead. Safe to call from every worker.';

-- Operating the queue.

CREATE FUNCTION mempher.retry_job(p_job uuid, p_at timestamptz DEFAULT now())
    RETURNS mempher.jobs
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_job   mempher.jobs;
    v_state text;
BEGIN
    PERFORM mempher._check_at(p_at);
    -- The attempts are reset because they were spent on a cause that has since
    -- been dealt with; a revived job that could only try once more would die
    -- again on the first flake. last_error stays, so the row still says what
    -- went wrong.
    BEGIN
        UPDATE mempher.jobs SET
            state        = 'pending',
            attempts     = 0,
            run_after    = p_at,
            leased_until = NULL,
            leased_by    = NULL,
            updated_at   = p_at
        WHERE id = p_job AND state = 'dead'
        RETURNING * INTO v_job;
    EXCEPTION WHEN unique_violation THEN
        PERFORM mempher._fail('MP025', format(
            'retry job %s: the same work is already pending or running', p_job));
    END;
    IF FOUND THEN
        RETURN v_job;
    END IF;
    SELECT state INTO v_state FROM mempher.jobs WHERE id = p_job;
    IF NOT FOUND THEN
        PERFORM mempher._fail('MP020', format('retry job %s: no such job', p_job));
    END IF;
    PERFORM mempher._fail('MP024', format('retry job %s: it is %s, not dead', p_job, v_state));
    RETURN NULL;
END $$;

COMMENT ON FUNCTION mempher.retry_job(uuid, timestamptz) IS
    'Returns one dead job to pending with its attempts reset, once its cause is dealt '
    'with. Raises MP024 if it is not dead and MP025 if the same work is already queued.';

CREATE FUNCTION mempher.purge_jobs(
    p_before timestamptz,
    p_states text[] DEFAULT ARRAY['done', 'dead'],
    p_limit  int    DEFAULT 1000
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_bad   text;
    v_count int;
BEGIN
    PERFORM mempher._check_at(p_before, 'before');
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_states IS NULL OR cardinality(p_states) = 0 THEN
        PERFORM mempher._fail('MP009', 'states is empty; name done, dead or both');
    END IF;
    -- Deleting a job still owed would lose the work with no record it was lost.
    SELECT s INTO v_bad FROM unnest(p_states) AS s
    WHERE s IS NULL OR s NOT IN ('done', 'dead') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP009', format(
            'cannot purge %s jobs: only done and dead jobs are finished', coalesce(v_bad, 'null')));
    END IF;

    -- A bare DELETE cannot be bounded, and a purge that cannot be bounded holds
    -- locks for as long as the history is deep.
    DELETE FROM mempher.jobs
    WHERE id IN (
        SELECT id FROM mempher.jobs
        WHERE state = ANY(p_states) AND updated_at < p_before
        ORDER BY id
        LIMIT p_limit);
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.purge_jobs(timestamptz, text[], int) IS
    'Deletes up to p_limit finished (done or dead) jobs last updated before p_before, '
    'and returns how many went. Call it in a loop until it returns less than p_limit.';

CREATE FUNCTION mempher.job_stats(p_scope text DEFAULT NULL)
    RETURNS TABLE (kind text, state text, jobs bigint, oldest timestamptz)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    RETURN QUERY
    SELECT j.kind, j.state, count(*), min(j.created_at)
    FROM mempher.jobs j
    WHERE p_scope IS NULL OR j.scope_id = p_scope
    GROUP BY j.kind, j.state
    ORDER BY j.kind, j.state;
END $$;

COMMENT ON FUNCTION mempher.job_stats(text) IS
    'Counts the queue by kind and state, with the creation time of the oldest job in '
    'each, across every scope or within one.';


-- ---------------------------------------------------------------------------
-- Writing the projections
-- ---------------------------------------------------------------------------

-- A worker produces vectors and extractions, and each is one call here. What
-- the output means for the projection -- which episode row a vector's scope
-- comes from, when a claim supersedes another, what an extraction that has
-- already been applied does the second time -- is decided by these functions,
-- not by the worker.

-- Matches any one of a text's lexemes, stemmed under the schema's own
-- configuration exactly as the indexed columns were. A conversational question
-- has its words spread across a conversation, not gathered in one turn, so the
-- channels that use this want everything that shares anything, best overlap
-- first; requiring every word left 83% of LongMemEval questions matching
-- nothing. quote_literal makes each lexeme a literal rather than something the
-- tsquery parser could read as an operator, and text that is only stop words is
-- the empty query, which matches nothing and ranks everything zero.
CREATE FUNCTION mempher._any_lexeme_query(p_text text)
    RETURNS tsquery
    LANGUAGE sql
    STABLE
    SET search_path FROM CURRENT
RETURN coalesce(
    (SELECT string_agg(quote_literal(lexeme), ' | ')
     FROM unnest(to_tsvector('{{.TextSearchConfig}}'::regconfig, coalesce(p_text, ''))))::tsquery,
    ''::tsquery);

-- A label that ends up in operator queries and log lines: 1 to p_max bytes, no
-- control character, and for a slug no whitespace either.
CREATE FUNCTION mempher._check_label(
    p_value text, p_what text, p_code text, p_max int, p_slug boolean DEFAULT false)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_value IS NULL OR p_value = '' THEN
        PERFORM mempher._fail(p_code, format('%s is empty', p_what));
    ELSIF octet_length(p_value) > p_max THEN
        PERFORM mempher._fail(p_code, format('%s is %s bytes, limit is %s',
            p_what, octet_length(p_value), p_max));
    ELSIF p_value ~ '[\x01-\x1f\x7f-\x9f]' THEN
        PERFORM mempher._fail(p_code, format('%s holds a control character', p_what));
    ELSIF p_slug AND p_value ~ '\s' THEN
        PERFORM mempher._fail(p_code, format('%s %L holds whitespace', p_what, p_value),
            'a predicate is identity: "lives_in" and "lives in" must not be two relations');
    END IF;
END $$;

-- A JSON timestamp field, or NULL when it is absent and p_required is false.
CREATE FUNCTION mempher._json_time(
    p_object jsonb, p_key text, p_where text, p_code text, p_required boolean)
    RETURNS timestamptz
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_value jsonb := p_object->p_key;
BEGIN
    IF v_value IS NULL OR v_value = 'null'::jsonb THEN
        IF p_required THEN
            PERFORM mempher._fail(p_code, format('%s: %s is required', p_where, p_key));
        END IF;
        RETURN NULL;
    END IF;
    IF jsonb_typeof(v_value) <> 'string'
       OR NOT pg_input_is_valid(v_value #>> '{}', 'timestamptz')
       OR NOT isfinite((v_value #>> '{}')::timestamptz) THEN
        PERFORM mempher._fail(p_code, format('%s: %s %s is not a finite timestamp', p_where, p_key, v_value));
    END IF;
    RETURN (v_value #>> '{}')::timestamptz;
END $$;

-- The assertions of an extraction as typed rows, once apply_extraction has
-- checked them. A function rather than a temporary table: one per extraction
-- would churn the system catalogue at the rate the workers run.
CREATE FUNCTION mempher._assertions(p_assert jsonb)
    RETURNS TABLE (ord bigint, subject text, predicate text, object text,
                   statement text, valid tstzrange, confidence real)
    LANGUAGE sql
    STABLE
    SET search_path FROM CURRENT
AS $$
    SELECT a.o, a.e->>'subject', a.e->>'predicate', a.e->>'object', a.e->>'statement',
           tstzrange((a.e->>'valid_from')::timestamptz, (a.e->>'valid_to')::timestamptz, '[)'),
           coalesce((a.e->'confidence')::real, 1)
    FROM jsonb_array_elements(p_assert) WITH ORDINALITY AS a(e, o)
$$;

-- Episode encodings.

CREATE FUNCTION mempher.put_encodings(
    p_model    text,
    p_episodes uuid[],
    p_vectors  vector[],
    p_at       timestamptz DEFAULT now(),
    p_surprise real[]      DEFAULT NULL
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_n       int := coalesce(cardinality(p_episodes), 0);
    v_bad     int;
    v_missing uuid;
    v_count   int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    IF v_n = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'name at least one episode, and no null');
    END IF;
    IF coalesce(cardinality(p_vectors), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s vectors for %s episodes',
            coalesce(cardinality(p_vectors), 0), v_n));
    END IF;
    IF p_surprise IS NOT NULL AND cardinality(p_surprise) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s surprise values for %s episodes',
            cardinality(p_surprise), v_n));
    END IF;
    -- Caught here, by position, rather than by the column's typmod, which
    -- reports the width and not which vector or which model produced it.
    SELECT o INTO v_bad FROM unnest(p_vectors) WITH ORDINALITY AS u(v, o)
    WHERE v IS NULL OR vector_dims(v) <> {{.VectorDimensions}} LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP021', format(
            'vector %s from model %L has %s dimensions but the schema was migrated for %s',
            v_bad, p_model, vector_dims(p_vectors[v_bad]), {{.VectorDimensions}}));
    END IF;
    SELECT o INTO v_bad FROM unnest(p_surprise) WITH ORDINALITY AS u(s, o)
    WHERE s IS NULL OR s < 0 OR s > 1 LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format('surprise %s is %s, must be in [0,1]',
            v_bad, p_surprise[v_bad]));
    END IF;

    -- scope_id and ingested_at come from the episode row, never the caller:
    -- they are denormalised so the vector index can filter without a join,
    -- and deriving them here is what keeps them from disagreeing with L0.
    -- The upsert is what makes an encode job safe to retry.
    INSERT INTO mempher.episode_encodings
        (episode_id, model, scope_id, ingested_at, content_vec, surprise, encoded_at)
    SELECT e.id, p_model, e.scope_id, e.ingested_at, u.v, coalesce(u.s, 0), p_at
    FROM unnest(p_episodes, p_vectors, p_surprise) AS u(id, v, s)
    JOIN mempher.episodes e ON e.id = u.id
    ON CONFLICT (episode_id, model) DO UPDATE
    SET content_vec = excluded.content_vec,
        surprise    = excluded.surprise,
        encoded_at  = excluded.encoded_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;

    IF v_count < (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) THEN
        SELECT x INTO v_missing FROM unnest(p_episodes) AS x
        WHERE NOT EXISTS (SELECT 1 FROM mempher.episodes e WHERE e.id = x) LIMIT 1;
        PERFORM mempher._fail('MP020', format('episode %s does not exist', v_missing));
    END IF;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.put_encodings(text, uuid[], vector[], timestamptz, real[]) IS
    'Writes one vector per episode for a model, replacing any the episode already had '
    'for that model, and returns how many were written. The arrays are parallel. An '
    'episode that does not exist fails the whole call with MP020.';

-- Entities. Everything that writes L1 for a scope goes through these, under
-- _lock_l1, so a name is never resolved against aliases that another
-- transaction is in the middle of moving.

-- Serialises the writers of one scope's L1 until the transaction ends. An
-- extraction rewrites subjects to the names the aliases give, and a merge
-- rewrites the facts and the aliases together; two of them interleaved would
-- leave a fact under a name whose entity has just gone. The work it serialises
-- is the few milliseconds of applying an extraction, never the model call
-- before it.
CREATE FUNCTION mempher._lock_l1(p_scope text)
    RETURNS void
    LANGUAGE sql
    SET search_path FROM CURRENT
AS $$
    SELECT pg_advisory_xact_lock(hashtextextended('mempher.l1:' || p_scope, 0))
$$;

-- A name an entity can go by: a label of 1 to 200 bytes that is not blank.
CREATE FUNCTION mempher._check_entity_name(p_value text, p_what text, p_code text DEFAULT 'MP014')
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    PERFORM mempher._check_label(p_value, p_what, p_code, 200);
    IF mempher._entity_key(p_value) = '' THEN
        PERFORM mempher._fail(p_code, format('%s is blank', p_what));
    END IF;
END $$;

-- The entity a name leads to, created under that name when nothing goes by it
-- yet. The caller holds _lock_l1.
CREATE FUNCTION mempher._entity(
    p_scope text, p_extractor text, p_name text, p_at timestamptz, p_episodes uuid[])
    RETURNS uuid
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_key text := mempher._entity_key(p_name);
    v_id  uuid;
BEGIN
    SELECT a.entity_id INTO v_id FROM mempher.entity_aliases a
    WHERE a.scope_id = p_scope AND a.extractor = p_extractor AND a.key = v_key;
    IF FOUND THEN
        RETURN v_id;
    END IF;
    INSERT INTO mempher.entities (scope_id, extractor, name, created_at, updated_at)
    VALUES (p_scope, p_extractor, p_name, p_at, p_at)
    RETURNING id INTO v_id;
    INSERT INTO mempher.entity_aliases (scope_id, extractor, key, alias, entity_id, episode_ids, created_at)
    VALUES (p_scope, p_extractor, v_key, p_name, v_id, coalesce(p_episodes, '{}'), p_at);
    RETURN v_id;
END $$;

-- Records that p_alias is another name for whatever p_of names, and returns the
-- entity both now lead to. The caller holds _lock_l1.
--
-- A name nothing goes by yet is simply added. A name that already leads to a
-- different entity means the two were one thing all along, and that entity is
-- merged into p_of's: its names move, its facts are rewritten to the surviving
-- subject, and its row goes.
--
-- Rewriting can put two rows on one claim over overlapping windows, which the
-- temporal key forbids and which is exactly the duplicate the merge exists to
-- remove. The surviving entity's row stands as it is and gains the other's
-- episodes as provenance; the other row is deleted. Which of two windows was
-- right is not something a merge can know, so it does not combine them: the
-- side named as the real one is believed.
CREATE FUNCTION mempher._alias_entity(
    p_scope text, p_extractor text, p_alias text, p_of text, p_at timestamptz, p_episodes uuid[])
    RETURNS uuid
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_key       text := mempher._entity_key(p_alias);
    v_into      uuid := mempher._entity(p_scope, p_extractor, p_of, p_at, p_episodes);
    v_from      uuid;
    v_into_name text;
    v_from_name text;
BEGIN
    SELECT a.entity_id INTO v_from FROM mempher.entity_aliases a
    WHERE a.scope_id = p_scope AND a.extractor = p_extractor AND a.key = v_key;
    IF NOT FOUND THEN
        INSERT INTO mempher.entity_aliases (scope_id, extractor, key, alias, entity_id, episode_ids, created_at)
        VALUES (p_scope, p_extractor, v_key, p_alias, v_into, coalesce(p_episodes, '{}'), p_at);
        UPDATE mempher.entities SET updated_at = p_at WHERE id = v_into;
        RETURN v_into;
    ELSIF v_from = v_into THEN
        RETURN v_into;
    END IF;

    SELECT e.name INTO v_into_name FROM mempher.entities e WHERE e.id = v_into;
    SELECT e.name INTO v_from_name FROM mempher.entities e WHERE e.id = v_from;

    WITH clash AS (
        SELECT kept.id AS kept, gone.id AS gone, gone.episode_ids
        FROM mempher.facts kept
        JOIN mempher.facts gone
          ON gone.scope_id = kept.scope_id AND gone.extractor = kept.extractor
         AND gone.predicate = kept.predicate AND gone.object = kept.object
         AND gone.valid && kept.valid
        WHERE kept.scope_id = p_scope AND kept.extractor = p_extractor
          AND kept.subject = v_into_name AND gone.subject = v_from_name
    ), gained AS (
        SELECT c.kept, array_agg(e) AS episode_ids
        FROM clash c, unnest(c.episode_ids) AS e
        GROUP BY c.kept
    ), credited AS (
        UPDATE mempher.facts f
        SET episode_ids = (SELECT array_agg(DISTINCT e ORDER BY e)
                           FROM unnest(f.episode_ids || g.episode_ids) AS e),
            updated_at  = p_at
        FROM gained g
        WHERE f.id = g.kept
    )
    DELETE FROM mempher.facts f USING clash c WHERE f.id = c.gone;

    UPDATE mempher.facts f
    SET subject = v_into_name, updated_at = p_at
    WHERE f.scope_id = p_scope AND f.extractor = p_extractor AND f.subject = v_from_name;

    UPDATE mempher.entity_aliases SET entity_id = v_into WHERE entity_id = v_from;
    DELETE FROM mempher.entities WHERE id = v_from;
    UPDATE mempher.entities SET updated_at = p_at WHERE id = v_into;
    RETURN v_into;
END $$;

CREATE FUNCTION mempher.merge_entities(
    p_scope     text,
    p_extractor text,
    p_alias     text,
    p_into      text,
    p_at        timestamptz DEFAULT now()
)
    RETURNS uuid
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    PERFORM mempher._check_entity_name(p_alias, 'alias');
    PERFORM mempher._check_entity_name(p_into, 'into');
    PERFORM mempher._check_at(p_at);
    IF NOT EXISTS (SELECT 1 FROM mempher.scopes s WHERE s.id = p_scope) THEN
        PERFORM mempher._fail('MP020', format('no such scope %L', p_scope));
    END IF;
    PERFORM mempher._lock_l1(p_scope);
    RETURN mempher._alias_entity(p_scope, p_extractor, p_alias, p_into, p_at, NULL);
END $$;

COMMENT ON FUNCTION mempher.merge_entities(text, text, text, text, timestamptz) IS
    'Declares that p_alias and p_into name the same thing in one extractor''s view of a '
    'scope, and returns the id of the entity both now lead to. A name nothing goes by '
    'yet is added; if p_alias already names another entity, that entity is merged into '
    'p_into''s: its names move and its facts take the surviving subject, and where both '
    'held the same claim over overlapping windows the surviving entity''s fact stands. '
    'Declaring it twice changes nothing the second time. There is no unmerge: a wrong '
    'merge is repaired by rebuilding L1 from L0.';

-- Applying an extraction.

CREATE FUNCTION mempher.apply_extraction(
    p_scope     text,
    p_extractor text,
    p_episodes  uuid[],
    p_assert    jsonb       DEFAULT '[]',
    p_retract   jsonb       DEFAULT '[]',
    p_at        timestamptz DEFAULT now(),
    p_encode    boolean     DEFAULT true,
    p_aliases   jsonb       DEFAULT '[]'
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_elem     jsonb;
    v_ord      bigint;
    v_where    text;
    v_from     timestamptz;
    v_to       timestamptz;
    v_conf     jsonb;
    v_found    int;
    v_marked   int;
    v_fact     uuid;
    v_lower    timestamptz;
    v_clash    record;
    v_unknown  text;
    v_name     text;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP012', 200);
    PERFORM mempher._check_at(p_at);
    IF p_episodes IS NULL OR cardinality(p_episodes) = 0 OR array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP012', 'an extraction names the episodes it read, and no null');
    END IF;
    IF jsonb_typeof(p_assert) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP012', 'assert is not a JSON array');
    ELSIF jsonb_array_length(p_assert) > 64 THEN
        PERFORM mempher._fail('MP012', format('extraction asserts %s facts, limit is 64',
            jsonb_array_length(p_assert)));
    END IF;
    IF jsonb_typeof(p_retract) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP012', 'retract is not a JSON array');
    ELSIF jsonb_array_length(p_retract) > 64 THEN
        PERFORM mempher._fail('MP012', format('extraction retracts %s facts, limit is 64',
            jsonb_array_length(p_retract)));
    END IF;
    IF jsonb_typeof(p_aliases) IS DISTINCT FROM 'array' THEN
        PERFORM mempher._fail('MP012', 'aliases is not a JSON array');
    ELSIF jsonb_array_length(p_aliases) > 64 THEN
        PERFORM mempher._fail('MP012', format('extraction declares %s aliases, limit is 64',
            jsonb_array_length(p_aliases)));
    END IF;

    -- Every assertion is checked before anything is written.
    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_assert) WITH ORDINALITY AS a(e, o)
    LOOP
        v_where := format('assertion %s', v_ord);
        IF jsonb_typeof(v_elem) <> 'object' THEN
            PERFORM mempher._fail('MP012', format('%s is not a JSON object', v_where));
        END IF;
        SELECT k INTO v_unknown FROM jsonb_object_keys(v_elem) AS k
        WHERE k NOT IN ('subject', 'predicate', 'object', 'statement', 'valid_from', 'valid_to', 'confidence')
        LIMIT 1;
        IF FOUND THEN
            PERFORM mempher._fail('MP012', format('%s has unknown key %L', v_where, v_unknown),
                'an assertion takes subject, predicate, object, statement, valid_from, valid_to and confidence');
        END IF;
        IF jsonb_typeof(v_elem->'subject') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'predicate') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'object') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'statement') IS DISTINCT FROM 'string' THEN
            PERFORM mempher._fail('MP010', format(
                '%s: subject, predicate, object and statement must all be strings', v_where));
        END IF;
        PERFORM mempher._check_entity_name(v_elem->>'subject', v_where || ': subject', 'MP010');
        PERFORM mempher._check_label(v_elem->>'predicate', v_where || ': predicate', 'MP010', 100, true);
        IF v_elem->>'object' = '' OR octet_length(v_elem->>'object') > 1000 THEN
            PERFORM mempher._fail('MP010', format('%s: object must be 1 to 1000 bytes', v_where));
        END IF;
        IF btrim(v_elem->>'statement') = '' OR octet_length(v_elem->>'statement') > 4000 THEN
            PERFORM mempher._fail('MP010', format('%s: statement must be 1 to 4000 bytes, not blank', v_where));
        END IF;
        v_conf := coalesce(v_elem->'confidence', '1'::jsonb);
        IF jsonb_typeof(v_conf) <> 'number' OR v_conf::numeric < 0 OR v_conf::numeric > 1 THEN
            PERFORM mempher._fail('MP010', format('%s: confidence %s must be a number in [0,1]', v_where, v_conf));
        END IF;
        v_from := mempher._json_time(v_elem, 'valid_from', v_where, 'MP011', true);
        v_to := mempher._json_time(v_elem, 'valid_to', v_where, 'MP011', false);
        IF v_to IS NOT NULL AND v_to <= v_from THEN
            PERFORM mempher._fail('MP011', format(
                '%s: the window ends (%s) at or before it starts (%s)', v_where, v_to, v_from));
        END IF;
    END LOOP;

    -- The same claim twice over the same window is one claim, and the first
    -- wording of it wins. Over two windows that overlap it is a contradiction
    -- the model made within one answer, which the database would reject anyway,
    -- less legibly. Two spellings of a subject that differ only in case or
    -- spacing are one subject here; two that an alias joins are caught by the
    -- temporal key below.
    SELECT a.subject, a.predicate, a.object INTO v_clash
    FROM mempher._assertions(p_assert) a JOIN mempher._assertions(p_assert) b
      ON a.ord < b.ord AND mempher._entity_key(a.subject) = mempher._entity_key(b.subject)
     AND (a.predicate, a.object) = (b.predicate, b.object)
     AND a.valid && b.valid AND a.valid <> b.valid
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP022', format(
            '%L %L %L is asserted over two overlapping windows in one extraction',
            v_clash.subject, v_clash.predicate, v_clash.object));
    END IF;

    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_retract) WITH ORDINALITY AS a(e, o)
    LOOP
        v_where := format('retraction %s', v_ord);
        IF jsonb_typeof(v_elem) <> 'object'
           OR jsonb_typeof(v_elem->'fact') IS DISTINCT FROM 'string'
           OR NOT pg_input_is_valid(v_elem->>'fact', 'uuid') THEN
            PERFORM mempher._fail('MP012', format('%s must name a fact by its uuid', v_where));
        END IF;
        PERFORM mempher._json_time(v_elem, 'at', v_where, 'MP012', true);
    END LOOP;

    FOR v_elem, v_ord IN SELECT e, o FROM jsonb_array_elements(p_aliases) WITH ORDINALITY AS a(e, o)
    LOOP
        v_where := format('alias %s', v_ord);
        IF jsonb_typeof(v_elem) <> 'object'
           OR EXISTS (SELECT 1 FROM jsonb_object_keys(v_elem) AS k WHERE k NOT IN ('alias', 'of'))
           OR jsonb_typeof(v_elem->'alias') IS DISTINCT FROM 'string'
           OR jsonb_typeof(v_elem->'of') IS DISTINCT FROM 'string' THEN
            PERFORM mempher._fail('MP012', format('%s must be {"alias": name, "of": name}', v_where));
        END IF;
        PERFORM mempher._check_entity_name(v_elem->>'alias', v_where || ': alias');
        PERFORM mempher._check_entity_name(v_elem->>'of', v_where || ': of');
    END LOOP;

    -- From here on the scope's names are read and written, so its other L1
    -- writers wait.
    PERFORM mempher._lock_l1(p_scope);

    -- The markers say which episodes this extractor has read. They are what
    -- makes a retried job a no-op rather than a second application: if every
    -- episode was already marked, the extraction has been applied, and its
    -- facts are returned as they stand.
    WITH found AS (
        SELECT e.id, e.scope_id, e.ingested_at
        FROM mempher.episodes e
        WHERE e.id = ANY(p_episodes) AND e.scope_id = p_scope
    ), marked AS (
        INSERT INTO mempher.fact_extractions
            (episode_id, extractor, scope_id, ingested_at, fact_count, extracted_at)
        SELECT f.id, p_extractor, f.scope_id, f.ingested_at,
               (SELECT count(*) FROM (SELECT DISTINCT subject, predicate, object, valid
                                      FROM mempher._assertions(p_assert)) d), p_at
        FROM found f
        ON CONFLICT (episode_id, extractor) DO NOTHING
        RETURNING episode_id
    )
    SELECT (SELECT count(*) FROM found), (SELECT count(*) FROM marked)
    INTO v_found, v_marked;

    IF v_found < (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) THEN
        PERFORM mempher._fail('MP020', format('%s of %s episodes are not in scope %L',
            (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x) - v_found,
            (SELECT count(DISTINCT x) FROM unnest(p_episodes) AS x), p_scope));
    END IF;

    IF v_marked > 0 THEN
        -- Retraction closes a window; it never deletes. A fact that had
        -- already stopped being true by then is left as it is, which is what
        -- makes replaying an old extraction harmless.
        FOR v_elem IN SELECT e FROM jsonb_array_elements(p_retract) AS a(e)
        LOOP
            v_fact := (v_elem->>'fact')::uuid;
            v_to := (v_elem->>'at')::timestamptz;
            UPDATE mempher.facts
            SET valid = tstzrange(lower(valid), v_to, '[)'), updated_at = p_at
            WHERE id = v_fact AND scope_id = p_scope
              AND lower(valid) < v_to
              AND (upper_inf(valid) OR upper(valid) > v_to);
            IF FOUND THEN
                CONTINUE;
            END IF;
            SELECT lower(valid) INTO v_lower FROM mempher.facts WHERE id = v_fact AND scope_id = p_scope;
            IF NOT FOUND THEN
                PERFORM mempher._fail('MP020', format('retract fact %s: no such fact in scope %L', v_fact, p_scope));
            ELSIF v_to <= v_lower THEN
                PERFORM mempher._fail('MP011', format(
                    'fact %s stops being true at %s, but only became true at %s', v_fact, v_to, v_lower));
            END IF;
        END LOOP;

        -- Names next, in the order given, so that a later alias can build on
        -- an earlier one. They come after the retractions, which name facts
        -- by ids a merge may remove, and before the assertions, which are
        -- written under whatever name these leave standing.
        FOR v_elem IN SELECT e FROM jsonb_array_elements(p_aliases) AS a(e)
        LOOP
            PERFORM mempher._alias_entity(p_scope, p_extractor,
                v_elem->>'alias', v_elem->>'of', p_at, p_episodes);
        END LOOP;

        -- Every subject is an entity, created under the first spelling it is
        -- given.
        FOR v_name IN
            SELECT DISTINCT ON (mempher._entity_key(x.subject)) x.subject
            FROM mempher._assertions(p_assert) x
            ORDER BY mempher._entity_key(x.subject), x.ord
        LOOP
            PERFORM mempher._entity(p_scope, p_extractor, v_name, p_at, p_episodes);
        END LOOP;

        -- A claim already held over exactly this window gains this
        -- extraction's episodes as provenance and its latest wording; any
        -- other claim is a new row. A new row that overlaps a window the
        -- scope already believes is the conflict WITHOUT OVERLAPS exists to
        -- catch. Each is written under its entity's name, whichever of the
        -- entity's names the assertion used, which is what makes the same
        -- claim about "Anna" and about "the user's sister" one row.
        BEGIN
            WITH a AS (
                SELECT DISTINCT ON (n.name, x.predicate, x.object, x.valid)
                       x.ord, n.name AS subject, x.predicate, x.object,
                       x.statement, x.valid, x.confidence
                FROM mempher._assertions(p_assert) x
                JOIN mempher.entity_aliases al
                  ON al.scope_id = p_scope AND al.extractor = p_extractor
                 AND al.key = mempher._entity_key(x.subject)
                JOIN mempher.entities n ON n.id = al.entity_id
                ORDER BY n.name, x.predicate, x.object, x.valid, x.ord
            ), updated AS (
                UPDATE mempher.facts f
                SET statement   = a.statement,
                    confidence  = a.confidence,
                    episode_ids = (SELECT array_agg(DISTINCT e ORDER BY e) FROM unnest(f.episode_ids || p_episodes) AS e),
                    updated_at  = p_at
                FROM a
                WHERE f.scope_id = p_scope AND f.extractor = p_extractor
                  AND (f.subject, f.predicate, f.object) = (a.subject, a.predicate, a.object)
                  AND f.valid = a.valid
                RETURNING a.ord
            )
            INSERT INTO mempher.facts
                (scope_id, extractor, subject, predicate, object, statement, valid,
                 confidence, episode_ids, asserted_at, updated_at)
            SELECT p_scope, p_extractor, a.subject, a.predicate, a.object, a.statement, a.valid,
                   a.confidence, p_episodes, p_at, p_at
            FROM a
            WHERE a.ord NOT IN (SELECT u.ord FROM updated u);
        EXCEPTION WHEN exclusion_violation THEN
            PERFORM mempher._fail('MP022',
                'an assertion overlaps a window the scope already believes for the same claim',
                'retract the existing fact first, or assert the window it does not cover');
        END;
    END IF;

    -- The facts this leaves with no vector are work, and the database queues
    -- it. "No vector" is asked of every model at once, because the database
    -- does not know which one the deployment embeds with. It is exact for the
    -- usual deployment, which has one; a second model is
    -- enqueue_fact_encodings' job.
    IF coalesce(p_encode, true) AND EXISTS (
        SELECT 1 FROM mempher.facts f
        WHERE f.scope_id = p_scope AND f.episode_ids && p_episodes
          AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x WHERE x.fact_id = f.id)
    ) THEN
        INSERT INTO mempher.jobs
            (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
        VALUES ('encode_facts', p_scope, p_episodes, p_at, 5, p_at, p_at)
        ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
            DO NOTHING;
    END IF;

    RETURN QUERY
    SELECT * FROM mempher.facts
    WHERE scope_id = p_scope AND extractor = p_extractor AND episode_ids && p_episodes
    ORDER BY subject, predicate, object, lower(valid);
END $$;

COMMENT ON FUNCTION mempher.apply_extraction(text, text, uuid[], jsonb, jsonb, timestamptz, boolean, jsonb) IS
    'Records one extraction of some episodes atomically: their read markers, the '
    'retractions (each {"fact": uuid, "at": timestamp} closes that fact''s window), the '
    'aliases (each {"alias": name, "of": name} declares two names for one entity, as '
    'merge_entities does), and the assertions (each {"subject", "predicate", "object", '
    '"statement", "valid_from", "valid_to"?, "confidence"?}). Every subject becomes an '
    'entity, and a fact is stored under its entity''s name whichever of its names the '
    'assertion used. Applying the same extraction twice changes nothing the '
    'second time. Returns every fact the extraction''s episodes support, and, unless '
    'p_encode is false, queues an encode_facts job when one of them has no vector.';

-- Fact encodings. An encode_facts job is a worker reading
-- pending_fact_encodings, embedding the statements, and handing them to
-- put_fact_encodings.
CREATE FUNCTION mempher.pending_fact_encodings(
    p_model    text,
    p_scope    text   DEFAULT NULL,
    p_episodes uuid[] DEFAULT NULL,
    p_after    uuid   DEFAULT NULL,
    p_limit    int    DEFAULT 100
)
    RETURNS TABLE (id uuid, scope_id text, statement text)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    IF array_position(p_episodes, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'pending_fact_encodings names a null episode');
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;

    RETURN QUERY
    SELECT f.id, f.scope_id, f.statement FROM mempher.facts f
    WHERE (p_scope IS NULL OR f.scope_id = p_scope)
      AND (p_episodes IS NULL OR cardinality(p_episodes) = 0 OR f.episode_ids && p_episodes)
      AND (p_after IS NULL OR f.id > p_after)
      AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x
                      WHERE x.fact_id = f.id AND x.model = p_model)
    ORDER BY f.id
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.pending_fact_encodings(text, text, uuid[], uuid, int) IS
    'Lists facts with no vector by model p_model, in id order, with the statement to '
    'embed. p_scope and p_episodes narrow it to one scope and to facts those episodes '
    'support, which is what an encode_facts job asks for. Page with p_after.';

CREATE FUNCTION mempher.put_fact_encodings(
    p_model      text,
    p_facts      uuid[],
    p_statements text[],
    p_vectors    vector[],
    p_at         timestamptz DEFAULT now()
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_n     int := coalesce(cardinality(p_facts), 0);
    v_bad   int;
    v_count int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    IF v_n = 0 OR array_position(p_facts, NULL) IS NOT NULL THEN
        PERFORM mempher._fail('MP013', 'name at least one fact, and no null');
    END IF;
    IF (SELECT count(DISTINCT x) FROM unnest(p_facts) AS x) <> v_n THEN
        PERFORM mempher._fail('MP013', 'a fact is named twice');
    END IF;
    IF coalesce(cardinality(p_statements), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s statements for %s facts',
            coalesce(cardinality(p_statements), 0), v_n));
    END IF;
    IF coalesce(cardinality(p_vectors), 0) <> v_n THEN
        PERFORM mempher._fail('MP013', format('%s vectors for %s facts',
            coalesce(cardinality(p_vectors), 0), v_n));
    END IF;
    SELECT o INTO v_bad FROM unnest(p_vectors) WITH ORDINALITY AS u(v, o)
    WHERE v IS NULL OR vector_dims(v) <> {{.VectorDimensions}} LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP021', format(
            'vector %s from model %L has %s dimensions but the schema was migrated for %s',
            v_bad, p_model, vector_dims(p_vectors[v_bad]), {{.VectorDimensions}}));
    END IF;

    -- The statement is compared under a row lock. The caller read it before
    -- its model call, and since then an extraction may have reworded the fact
    -- or an erasure removed it. The lock makes that change either visible
    -- here, so the vector is skipped, or wait until this row exists for the
    -- change's trigger to remove.
    WITH held AS (
        SELECT f.id, f.statement FROM mempher.facts f
        WHERE f.id = ANY(p_facts)
        ORDER BY f.id
        FOR SHARE
    )
    INSERT INTO mempher.fact_encodings (fact_id, model, statement_vec, encoded_at)
    SELECT h.id, p_model, u.v, p_at
    FROM unnest(p_facts, p_statements, p_vectors) AS u(id, s, v)
    JOIN held h ON h.id = u.id AND h.statement = u.s
    ON CONFLICT (fact_id, model) DO UPDATE
    SET statement_vec = excluded.statement_vec,
        encoded_at    = excluded.encoded_at;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.put_fact_encodings(text, uuid[], text[], vector[], timestamptz) IS
    'Writes one vector per fact for a model and returns how many were written. The '
    'arrays are parallel, and p_statements is the text each vector was made from: a fact '
    'that has gone, or whose statement has changed since it was read, is skipped rather '
    'than given a vector of text it does not hold.';

CREATE FUNCTION mempher.enqueue_fact_encodings(
    p_model        text,
    p_scope        text        DEFAULT NULL,
    p_at           timestamptz DEFAULT now(),
    p_run_after    timestamptz DEFAULT NULL,
    p_max_attempts int         DEFAULT 5
)
    RETURNS int
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_count int;
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    PERFORM mempher._check_at(p_at);
    IF p_max_attempts IS NULL OR p_max_attempts < 1 THEN
        PERFORM mempher._fail('MP013', format('max_attempts is %s, must be positive', p_max_attempts));
    END IF;

    -- A job names the first episode that supports a pending fact, not all of
    -- them: a fact many episodes agree on would otherwise name a list too long
    -- for the index that collapses duplicate jobs. The job embeds every
    -- pending fact that episode supports, so the grouping loses nothing.
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    SELECT DISTINCT 'encode_facts', f.scope_id, ARRAY[f.episode_ids[1]],
           coalesce(p_run_after, p_at), p_max_attempts, p_at, p_at
    FROM mempher.facts f
    WHERE (p_scope IS NULL OR f.scope_id = p_scope)
      AND NOT EXISTS (SELECT 1 FROM mempher.fact_encodings x
                      WHERE x.fact_id = f.id AND x.model = p_model)
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END $$;

COMMENT ON FUNCTION mempher.enqueue_fact_encodings(text, text, timestamptz, timestamptz, int) IS
    'Queues encode_facts jobs for every fact with no vector by model p_model, in one '
    'scope or in all of them, and returns how many jobs it added. This is the backfill: '
    'after a change of embedding model, or when a job died. Running it twice adds '
    'nothing the second time.';


-- ---------------------------------------------------------------------------
-- Reading: facts, and recall
-- ---------------------------------------------------------------------------

-- The weights a fusion runs with: the caller's, checked, with the lexical
-- channel at a quarter of a vote unless they say otherwise. That default was
-- calibrated on LongMemEval for episodes, where an equal vote let keyword
-- overlap outrank meaning; facts use it unmeasured. "entity" weighs the vote a
-- fact gets for touching an entity the question names, and is unmeasured too.
CREATE FUNCTION mempher._fusion_weights(p_weights jsonb)
    RETURNS jsonb
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_weights jsonb := coalesce(p_weights, '{}');
    v_bad     text;
BEGIN
    IF jsonb_typeof(v_weights) <> 'object' THEN
        PERFORM mempher._fail('MP013', 'weights is not a JSON object');
    END IF;
    SELECT key INTO v_bad FROM jsonb_each(v_weights)
    WHERE key NOT IN ('semantic', 'lexical', 'fact', 'entity')
       OR jsonb_typeof(value) <> 'number' OR value::float8 < 0
    LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP013', format(
            'weight %L must name a channel and be a number at least 0', v_bad));
    END IF;
    IF NOT v_weights ? 'lexical' THEN
        v_weights := v_weights || '{"lexical": 0.25}';
    END IF;
    RETURN v_weights;
END $$;

CREATE FUNCTION mempher._check_fusion_k(p_fusion_k float8)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    IF p_fusion_k IS NULL OR p_fusion_k < 0 OR p_fusion_k = 'Infinity' OR p_fusion_k = 'NaN' THEN
        PERFORM mempher._fail('MP013', format('fusion_k is %s, must be a finite number at least 0', p_fusion_k));
    END IF;
END $$;

-- A query's embedding comes with the model that made it and has the width the
-- schema was migrated for.
CREATE FUNCTION mempher._check_query_embedding(p_embedding vector, p_model text)
    RETURNS void
    LANGUAGE plpgsql
    IMMUTABLE
    SET search_path FROM CURRENT
AS $$
BEGIN
    PERFORM mempher._check_label(p_model, 'model', 'MP013', 200);
    IF vector_dims(p_embedding) <> {{.VectorDimensions}} THEN
        PERFORM mempher._fail('MP021', format(
            'query vector has %s dimensions but the schema was migrated for %s',
            vector_dims(p_embedding), {{.VectorDimensions}}));
    END IF;
END $$;

-- A text as the words in it, lowercased and single-spaced with a space at each
-- end, so that one text names another's subject exactly when it contains it:
-- " anna " is in " where does anna live " and not in " where does hanna live ".
CREATE FUNCTION mempher._mention_key(p_text text)
    RETURNS text
    LANGUAGE sql
    IMMUTABLE
    PARALLEL SAFE
    SET search_path FROM CURRENT
RETURN ' ' || btrim(regexp_replace(lower(coalesce(p_text, '')), '[^[:alnum:]]+', ' ', 'g')) || ' ';

-- The entities a text names, by any of the names each went by at p_as_of.
-- Every alias of the scope is tried against the text, which is a scan of one
-- scope's names and nothing wider.
CREATE FUNCTION mempher._mentioned_entities(
    p_scope text, p_extractor text, p_text text, p_as_of timestamptz)
    RETURNS uuid[]
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_text text := mempher._mention_key(p_text);
BEGIN
    IF v_text = '  ' THEN
        RETURN '{}';
    END IF;
    RETURN coalesce((
        SELECT array_agg(DISTINCT a.entity_id)
        FROM mempher.entity_aliases a
        WHERE a.scope_id = p_scope AND a.extractor = p_extractor AND a.created_at <= p_as_of
          AND mempher._mention_key(a.alias) <> '  '
          AND strpos(v_text, mempher._mention_key(a.alias)) > 0), '{}');
END $$;

CREATE FUNCTION mempher.search_facts(
    p_scope          text,
    p_extractor      text,
    p_at             timestamptz DEFAULT now(),
    p_as_of          timestamptz DEFAULT now(),
    p_text           text        DEFAULT NULL,
    p_limit          int         DEFAULT 200,
    p_subjects       text[]      DEFAULT NULL,
    p_predicates     text[]      DEFAULT NULL,
    p_min_confidence real        DEFAULT 0,
    p_include_closed boolean     DEFAULT false,
    p_embedding      vector      DEFAULT NULL,
    p_model          text        DEFAULT NULL,
    p_weights        jsonb       DEFAULT '{}',
    p_fusion_k       float8      DEFAULT 60
)
    RETURNS SETOF mempher.facts
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
DECLARE
    v_query     tsquery;
    v_weights   jsonb;
    v_subjects  text[];
    v_mentioned uuid[];
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    PERFORM mempher._check_at(p_as_of, 'as_of');
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_min_confidence IS NULL OR p_min_confidence < 0 OR p_min_confidence > 1 THEN
        PERFORM mempher._fail('MP013', format('min_confidence is %s, must be in [0,1]', p_min_confidence));
    END IF;
    IF octet_length(p_text) > 1048576 THEN
        PERFORM mempher._fail('MP002', 'query text is longer than 1048576 bytes');
    END IF;
    IF p_embedding IS NOT NULL THEN
        PERFORM mempher._check_query_embedding(p_embedding, p_model);
    END IF;
    PERFORM mempher._check_fusion_k(p_fusion_k);
    v_weights := mempher._fusion_weights(p_weights);
    v_query := mempher._any_lexeme_query(p_text);

    -- A subject filter may use any name the entity goes by; the facts
    -- themselves carry one. A name nothing goes by is kept as given, and
    -- matches nothing.
    IF cardinality(p_subjects) > 0 THEN
        SELECT array_agg(coalesce(e.name, s)) INTO v_subjects
        FROM unnest(p_subjects) AS s
        LEFT JOIN mempher.entity_aliases a
               ON a.scope_id = p_scope AND a.extractor = p_extractor
              AND a.key = mempher._entity_key(s)
        LEFT JOIN mempher.entities e ON e.id = a.entity_id;
    END IF;
    v_mentioned := mempher._mentioned_entities(p_scope, p_extractor, p_text, p_as_of);

    -- The question ranks, it never filters: relevance decides which facts
    -- survive the limit, not which are true. A fact the question does not
    -- mention is still a fact about the scope.
    --
    -- Three rankings vote, by reciprocal rank as recall fuses episodes: nearness
    -- of the statement's vector to p_embedding, the words the statement shares
    -- with p_text, and whether the fact touches an entity p_text names, as its
    -- subject or as its object. The last is what finds "the user's sister lives
    -- in Paris" for a question about Anna: the name is in neither the statement
    -- nor, necessarily, anywhere near it in meaning. Every fact that touches a
    -- named entity ties for first in that ranking. A fact with no vector yet,
    -- sharing no word, or touching no named entity has no vote from that
    -- ranking. Subject, predicate and object close the ordering, so equal
    -- scores come back in the same order every run.
    RETURN QUERY
    WITH candidates AS (
        SELECT f AS fact,
               ts_rank_cd(f.statement_tsv, v_query) AS lexical,
               x.statement_vec <=> p_embedding AS distance,
               cardinality(v_mentioned) > 0 AND EXISTS (
                   SELECT 1 FROM mempher.entity_aliases a
                   WHERE a.scope_id = p_scope AND a.extractor = p_extractor
                     AND a.key IN (mempher._entity_key(f.subject), mempher._entity_key(f.object))
                     AND a.entity_id = ANY(v_mentioned)) AS named
        FROM mempher.facts f
        LEFT JOIN mempher.fact_encodings x
               ON p_embedding IS NOT NULL AND x.fact_id = f.id AND x.model = p_model
        WHERE f.scope_id = p_scope AND f.extractor = p_extractor
          AND f.asserted_at <= p_as_of
          AND CASE WHEN p_include_closed THEN lower(f.valid) <= p_at ELSE f.valid @> p_at END
          AND (v_subjects IS NULL OR f.subject = ANY(v_subjects))
          AND (p_predicates IS NULL OR cardinality(p_predicates) = 0 OR f.predicate = ANY(p_predicates))
          AND f.confidence >= p_min_confidence
    ), scored AS (
        SELECT c.fact, c.lexical,
               CASE WHEN c.distance IS NOT NULL
                    THEN coalesce((v_weights->>'semantic')::float8, 1)
                         / (p_fusion_k + rank() OVER (ORDER BY c.distance))
                    ELSE 0 END
             + CASE WHEN c.lexical > 0
                    THEN (v_weights->>'lexical')::float8
                         / (p_fusion_k + rank() OVER (ORDER BY c.lexical DESC))
                    ELSE 0 END
             + CASE WHEN c.named
                    THEN coalesce((v_weights->>'entity')::float8, 1) / (p_fusion_k + 1)
                    ELSE 0 END AS score
        FROM candidates c
    )
    SELECT (s.fact).* FROM scored s
    ORDER BY s.score DESC, s.lexical DESC,
             (s.fact).subject, (s.fact).predicate, (s.fact).object, lower((s.fact).valid)
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.search_facts(text, text, timestamptz, timestamptz, text, int, text[], text[],
                                         real, boolean, vector, text, jsonb, float8) IS
    'Returns the facts one extractor holds about a scope that are true at p_at, as known '
    'by p_as_of. With p_include_closed, also those that began by p_at and have since '
    'stopped being true. p_subjects may use any name an entity goes by. The question '
    'orders the facts and never excludes one: by meaning when p_embedding is given (the '
    'question embedded by p_model), by shared words and by the entities it names when '
    'p_text is, all fused by reciprocal rank, weighted by p_weights (lexical defaults to '
    '0.25, entity to 1).';

CREATE FUNCTION mempher.list_entities(
    p_scope        text,
    p_extractor    text,
    p_mentioned_in text        DEFAULT NULL,
    p_as_of        timestamptz DEFAULT now(),
    p_limit        int         DEFAULT 200
)
    RETURNS TABLE (id uuid, name text, aliases text[], facts bigint,
                   created_at timestamptz, updated_at timestamptz)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
DECLARE
    v_mentioned uuid[];
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    PERFORM mempher._check_at(p_as_of, 'as_of');
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF octet_length(p_mentioned_in) > 1048576 THEN
        PERFORM mempher._fail('MP002', 'mentioned_in is longer than 1048576 bytes');
    END IF;
    IF p_mentioned_in IS NOT NULL THEN
        v_mentioned := mempher._mentioned_entities(p_scope, p_extractor, p_mentioned_in, p_as_of);
    END IF;

    RETURN QUERY
    SELECT e.id, e.name,
           coalesce((SELECT array_agg(a.alias ORDER BY a.alias)
                     FROM mempher.entity_aliases a
                     WHERE a.entity_id = e.id AND a.created_at <= p_as_of
                       AND a.key <> mempher._entity_key(e.name)), '{}'),
           (SELECT count(*) FROM mempher.facts f
            WHERE f.scope_id = e.scope_id AND f.extractor = e.extractor
              AND f.subject = e.name AND f.asserted_at <= p_as_of),
           e.created_at, e.updated_at
    FROM mempher.entities e
    WHERE e.scope_id = p_scope AND e.extractor = p_extractor AND e.created_at <= p_as_of
      AND (v_mentioned IS NULL OR e.id = ANY(v_mentioned))
    ORDER BY e.name, e.id
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.list_entities(text, text, text, timestamptz, int) IS
    'Lists the entities one extractor knows in a scope as of p_as_of, by name: each with '
    'its other names and how many facts are about it. With p_mentioned_in, only the '
    'entities that text names, by any name, as whole words and whatever the case.';

-- The facts around some entities. A fact whose object is one of an entity's
-- names is an edge between its subject and that entity, in both directions, so
-- the facts of a scope are a graph and "what do we know around Anna" is a walk
-- of it: Anna, then whoever a fact ties to Anna, then whoever a fact ties to
-- them. The walk stops at p_hops, which is also what ends it on a cycle.
CREATE FUNCTION mempher.related_facts(
    p_scope     text,
    p_extractor text,
    p_entities  text[],
    p_hops      int         DEFAULT 1,
    p_at        timestamptz DEFAULT now(),
    p_as_of     timestamptz DEFAULT now(),
    p_limit     int         DEFAULT 200
)
    RETURNS TABLE (hops int, id uuid, scope_id text, extractor text, subject text,
                   predicate text, object text, statement text, valid tstzrange,
                   confidence real, episode_ids uuid[], asserted_at timestamptz,
                   updated_at timestamptz)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
DECLARE
    v_name text;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    PERFORM mempher._check_at(p_at);
    PERFORM mempher._check_at(p_as_of, 'as_of');
    IF p_entities IS NULL OR cardinality(p_entities) = 0 THEN
        PERFORM mempher._fail('MP013', 'related_facts needs at least one entity to start from');
    ELSIF cardinality(p_entities) > 64 THEN
        PERFORM mempher._fail('MP013', format('related_facts starts from %s entities, limit is 64',
            cardinality(p_entities)));
    END IF;
    FOREACH v_name IN ARRAY p_entities LOOP
        PERFORM mempher._check_entity_name(v_name, 'entity');
    END LOOP;
    IF p_hops IS NULL OR p_hops < 0 OR p_hops > 8 THEN
        PERFORM mempher._fail('MP013', format('hops is %s, must be 0 to 8', p_hops));
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;

    RETURN QUERY
    WITH RECURSIVE visible AS MATERIALIZED (
        -- Every fact true at p_at, with the entity it is about and the entity
        -- its object names, if it names one.
        SELECT f.id AS fact_id, s.entity_id AS subject_entity, o.entity_id AS object_entity
        FROM mempher.facts f
        LEFT JOIN mempher.entity_aliases s
               ON s.scope_id = f.scope_id AND s.extractor = f.extractor
              AND s.key = mempher._entity_key(f.subject)
        LEFT JOIN mempher.entity_aliases o
               ON o.scope_id = f.scope_id AND o.extractor = f.extractor
              AND o.key = mempher._entity_key(f.object)
        WHERE f.scope_id = p_scope AND f.extractor = p_extractor
          AND f.asserted_at <= p_as_of AND f.valid @> p_at
    ), reach (entity_id, hops) AS (
        SELECT a.entity_id, 0
        FROM mempher.entity_aliases a
        WHERE a.scope_id = p_scope AND a.extractor = p_extractor
          AND a.key IN (SELECT mempher._entity_key(n) FROM unnest(p_entities) AS n)
      UNION
        SELECT CASE WHEN v.subject_entity = r.entity_id THEN v.object_entity
                    ELSE v.subject_entity END,
               r.hops + 1
        FROM reach r
        JOIN visible v ON r.entity_id IN (v.subject_entity, v.object_entity)
        WHERE r.hops < p_hops AND v.subject_entity <> v.object_entity
    ), nearest AS (
        SELECT r.entity_id, min(r.hops) AS hops FROM reach r GROUP BY r.entity_id
    ), touched AS (
        -- A fact is as near as the nearer of the two entities it ties.
        SELECT v.fact_id, min(n.hops) AS hops
        FROM visible v
        JOIN nearest n ON n.entity_id IN (v.subject_entity, v.object_entity)
        GROUP BY v.fact_id
    )
    SELECT t.hops, f.id, f.scope_id, f.extractor, f.subject, f.predicate, f.object,
           f.statement, f.valid, f.confidence, f.episode_ids, f.asserted_at, f.updated_at
    FROM touched t
    JOIN mempher.facts f ON f.id = t.fact_id
    ORDER BY t.hops, f.subject, f.predicate, f.object, lower(f.valid)
    LIMIT p_limit;
END $$;

COMMENT ON FUNCTION mempher.related_facts(text, text, text[], int, timestamptz, timestamptz, int) IS
    'Returns the facts true at p_at, as known by p_as_of, that lie within p_hops of the '
    'entities named in p_entities (by any of their names), nearest first. A fact whose '
    'object is the name of an entity ties its subject to that entity; hops 0 is every '
    'fact that touches a named entity as subject or object, and each further hop adds '
    'the facts of the entities those tie it to. A name nothing goes by contributes '
    'nothing.';

-- mempher.recall runs every retrieval channel the caller asked for, fuses the
-- episode channels by reciprocal rank, reads what the scope believes from L1,
-- spends a token budget on both -- facts first -- and returns the result as one
-- JSON document, with the provenance of every hit. The only part of reading
-- that happens outside the database is embedding the question, because that is
-- a model call.
CREATE FUNCTION mempher.recall(
    p_scope               text,
    p_query               text,
    p_embedding           vector      DEFAULT NULL,
    p_model               text        DEFAULT NULL,
    p_extractor           text        DEFAULT NULL,
    p_channels            text[]      DEFAULT NULL,
    p_limit               int         DEFAULT 10,
    p_candidates          int         DEFAULT 50,
    p_fact_limit          int         DEFAULT 32,
    p_max_tokens          int         DEFAULT NULL,
    p_as_of               timestamptz DEFAULT now(),
    p_occurred_from       timestamptz DEFAULT NULL,
    p_occurred_to         timestamptz DEFAULT NULL,
    p_binding             jsonb       DEFAULT NULL,
    p_roles               text[]      DEFAULT NULL,
    p_subjects            text[]      DEFAULT NULL,
    p_min_fact_confidence real        DEFAULT 0,
    p_weights             jsonb       DEFAULT '{}',
    p_fusion_k            float8      DEFAULT 60,
    p_bytes_per_token     int         DEFAULT 4
)
    RETURNS jsonb
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
    -- Without iterative index scans an HNSW search filtered to one scope
    -- returns fewer neighbours than asked for, and on a large database can
    -- return none at all: the index finds the nearest vectors anywhere, and the
    -- scope filter then discards them. strict_order keeps scanning until the
    -- limit is met, in distance order.
    SET hnsw.iterative_scan = strict_order
AS $$
DECLARE
    v_channels    text[];
    v_bad         text;
    v_weights     jsonb;
    v_candidates  int;
    v_tsq         tsquery;
    v_t0          timestamptz;
    v_reports     jsonb := '[]';
    v_sem_ids     uuid[]   := '{}';
    v_sem_scores  float8[] := '{}';
    v_lex_ids     uuid[]   := '{}';
    v_lex_scores  float8[] := '{}';
    v_fact_ids    uuid[]   := '{}';
    v_facts       jsonb    := '[]';
    v_episodes    jsonb    := '[]';
    v_fact_tokens bigint   := 0;
    v_ep_tokens   bigint   := 0;
    v_truncated   boolean  := false;
BEGIN
    -- ---------------------------------------------------------------- checks
    PERFORM mempher._check_scope(p_scope);
    IF octet_length(p_query) > 1048576 THEN
        PERFORM mempher._fail('MP002', format('recall query is %s bytes, limit is 1048576',
            octet_length(p_query)));
    END IF;
    PERFORM mempher._check_at(p_as_of, 'as_of');

    SELECT c INTO v_bad FROM unnest(p_channels) AS c
    WHERE c IS NULL OR c NOT IN ('semantic', 'lexical', 'fact') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP007', format('channel %s is not one of semantic, lexical, fact',
            coalesce(quote_literal(v_bad), 'null')));
    END IF;
    IF p_channels IS NULL OR cardinality(p_channels) = 0 THEN
        -- Every channel the arguments make possible.
        v_channels := array_remove(ARRAY[
            CASE WHEN p_embedding IS NOT NULL THEN 'semantic' END,
            CASE WHEN btrim(p_query) <> '' THEN 'lexical' END,
            CASE WHEN p_extractor IS NOT NULL THEN 'fact' END], NULL);
        IF cardinality(v_channels) = 0 THEN
            PERFORM mempher._fail('MP002', 'nothing to recall with: give a query, an embedding, or both');
        END IF;
    ELSE
        v_channels := ARRAY(SELECT c FROM unnest(p_channels) WITH ORDINALITY AS u(c, o)
                            GROUP BY c ORDER BY min(o));
    END IF;
    -- Words are what the lexical channel matches; the fact channel only
    -- ranks by them, and the semantic channel has its embedding instead.
    IF 'lexical' = ANY(v_channels) AND coalesce(btrim(p_query), '') = '' THEN
        PERFORM mempher._fail('MP002', 'the lexical channel needs a query');
    END IF;

    IF 'semantic' = ANY(v_channels) AND p_embedding IS NULL THEN
        PERFORM mempher._fail('MP013', 'the semantic channel needs the query''s embedding');
    END IF;
    -- Checked whichever channels run, because the fact channel ranks by it too.
    IF p_embedding IS NOT NULL THEN
        PERFORM mempher._check_query_embedding(p_embedding, p_model);
    END IF;
    IF 'fact' = ANY(v_channels) THEN
        IF p_extractor IS NULL THEN
            PERFORM mempher._fail('MP013', 'the fact channel needs the extractor whose facts to read');
        END IF;
        PERFORM mempher._check_label(p_extractor, 'extractor', 'MP013', 200);
    END IF;

    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;
    IF p_candidates IS NULL OR p_candidates < 1 THEN
        PERFORM mempher._fail('MP013', format('candidates is %s, must be positive', p_candidates));
    END IF;
    IF p_fact_limit IS NULL OR p_fact_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('fact_limit is %s, must be positive', p_fact_limit));
    END IF;
    IF p_max_tokens IS NOT NULL AND p_max_tokens < 1 THEN
        PERFORM mempher._fail('MP013', format('max_tokens is %s; leave it null for no budget', p_max_tokens));
    END IF;
    IF p_bytes_per_token IS NULL OR p_bytes_per_token < 1 THEN
        PERFORM mempher._fail('MP013', format('bytes_per_token is %s, must be positive', p_bytes_per_token));
    END IF;
    IF p_min_fact_confidence IS NULL OR p_min_fact_confidence < 0 OR p_min_fact_confidence > 1 THEN
        PERFORM mempher._fail('MP013', format('min_fact_confidence is %s, must be in [0,1]', p_min_fact_confidence));
    END IF;
    PERFORM mempher._check_fusion_k(p_fusion_k);
    -- An inverted window matches nothing, which is far more likely to be
    -- swapped arguments than an intentional empty question.
    IF p_occurred_from IS NOT NULL AND p_occurred_to IS NOT NULL AND p_occurred_to < p_occurred_from THEN
        PERFORM mempher._fail('MP013', format('the event-time window ends (%s) before it starts (%s)',
            p_occurred_to, p_occurred_from));
    END IF;
    IF p_binding IS NOT NULL THEN
        PERFORM mempher._check_binding(p_binding, 'recall');
    END IF;
    SELECT r INTO v_bad FROM unnest(p_roles) AS r
    WHERE r IS NULL OR r NOT IN ('user', 'assistant', 'system', 'tool', 'observation') LIMIT 1;
    IF FOUND THEN
        PERFORM mempher._fail('MP003', format('role filter %s is not a role', coalesce(quote_literal(v_bad), 'null')));
    END IF;
    FOR v_bad IN SELECT s FROM unnest(p_subjects) AS s LOOP
        PERFORM mempher._check_label(v_bad, 'subject', 'MP010', 200);
    END LOOP;

    v_weights := mempher._fusion_weights(p_weights);

    -- A channel is asked for at least as many candidates as will be returned,
    -- or a limit above the depth could never be met.
    v_candidates := greatest(p_candidates, p_limit);
    v_tsq := mempher._any_lexeme_query(p_query);

    -- -------------------------------------------------------------- channels
    IF 'semantic' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        SELECT coalesce(array_agg(c.id ORDER BY c.score DESC, c.id DESC), '{}'),
               coalesce(array_agg(c.score ORDER BY c.score DESC, c.id DESC), '{}')
        INTO v_sem_ids, v_sem_scores
        FROM (
            SELECT e.id, 1 - (enc.content_vec <=> p_embedding) AS score
            FROM mempher.episode_encodings enc
            JOIN mempher.episodes e ON e.id = enc.episode_id
            WHERE enc.model = p_model AND enc.scope_id = p_scope AND enc.ingested_at <= p_as_of
              AND (p_occurred_from IS NULL OR e.occurred_at >= p_occurred_from)
              AND (p_occurred_to IS NULL OR e.occurred_at <= p_occurred_to)
              AND (p_binding IS NULL OR e.binding @> p_binding)
              AND (p_roles IS NULL OR cardinality(p_roles) = 0 OR e.role = ANY(p_roles))
            ORDER BY enc.content_vec <=> p_embedding
            LIMIT v_candidates
        ) c;
        v_reports := v_reports || jsonb_build_object('channel', 'semantic',
            'candidates', cardinality(v_sem_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    IF 'lexical' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        SELECT coalesce(array_agg(c.id ORDER BY c.score DESC, c.id DESC), '{}'),
               coalesce(array_agg(c.score ORDER BY c.score DESC, c.id DESC), '{}')
        INTO v_lex_ids, v_lex_scores
        FROM (
            SELECT e.id, ts_rank_cd(e.content_tsv, v_tsq)::float8 AS score
            FROM mempher.episodes e
            WHERE e.scope_id = p_scope AND e.ingested_at <= p_as_of AND e.content_tsv @@ v_tsq
              AND (p_occurred_from IS NULL OR e.occurred_at >= p_occurred_from)
              AND (p_occurred_to IS NULL OR e.occurred_at <= p_occurred_to)
              AND (p_binding IS NULL OR e.binding @> p_binding)
              AND (p_roles IS NULL OR cardinality(p_roles) = 0 OR e.role = ANY(p_roles))
            ORDER BY score DESC, e.seq DESC
            LIMIT v_candidates
        ) c;
        v_reports := v_reports || jsonb_build_object('channel', 'lexical',
            'candidates', cardinality(v_lex_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    -- Facts are not ranked against episodes: they are what the scope currently
    -- believes, true at the end of the event-time window or else at as_of, and
    -- the question only decides which survive the limit. It is asked by
    -- meaning as well as by words whenever its embedding was given.
    IF 'fact' = ANY(v_channels) THEN
        v_t0 := clock_timestamp();
        WITH f AS (
            SELECT x.*, x.ordinality AS pos
            FROM mempher.search_facts(p_scope, p_extractor, coalesce(p_occurred_to, p_as_of), p_as_of,
                                      p_query, p_fact_limit, p_subjects, NULL, p_min_fact_confidence,
                                      false, p_embedding, p_model, v_weights, p_fusion_k)
                 WITH ORDINALITY AS x
        ), costed AS (
            SELECT f.*, (octet_length(f.statement) + p_bytes_per_token - 1) / p_bytes_per_token AS tokens,
                   sum((octet_length(f.statement) + p_bytes_per_token - 1) / p_bytes_per_token)
                       OVER (ORDER BY f.pos) AS spent
            FROM f
        )
        SELECT coalesce(jsonb_agg(jsonb_build_object(
                   'id', c.id, 'scope', c.scope_id, 'extractor', c.extractor,
                   'subject', c.subject, 'predicate', c.predicate, 'object', c.object,
                   'statement', c.statement,
                   'valid_from', lower(c.valid), 'valid_to', upper(c.valid),
                   'confidence', c.confidence, 'episodes', to_jsonb(c.episode_ids),
                   'asserted_at', c.asserted_at, 'updated_at', c.updated_at,
                   'tokens', c.tokens) ORDER BY c.pos)
                   FILTER (WHERE p_max_tokens IS NULL OR c.spent <= p_max_tokens), '[]'),
               coalesce(max(c.spent) FILTER (WHERE p_max_tokens IS NULL OR c.spent <= p_max_tokens), 0),
               bool_or(p_max_tokens IS NOT NULL AND c.spent > p_max_tokens),
               coalesce(array_agg(c.id), '{}')
        INTO v_facts, v_fact_tokens, v_truncated, v_fact_ids
        FROM costed c;
        v_truncated := coalesce(v_truncated, false);
        v_reports := v_reports || jsonb_build_object('channel', 'fact',
            'candidates', cardinality(v_fact_ids),
            'duration_us', (extract(epoch FROM clock_timestamp() - v_t0) * 1e6)::bigint);
    END IF;

    -- ---------------------------------------------------------------- fusion
    -- Reciprocal rank fusion: an episode scores the sum, over the channels
    -- that found it, of its channel's weight over K plus its rank there. Ranks
    -- rather than scores, because a cosine similarity and a ts_rank_cd are not
    -- on any common scale. Ties go to the newer episode.
    WITH hits AS (
        SELECT 'semantic'::text AS channel, u.id, u.rank::int AS rank, u.score
        FROM unnest(v_sem_ids, v_sem_scores) WITH ORDINALITY AS u(id, score, rank)
        UNION ALL
        SELECT 'lexical', u.id, u.rank::int, u.score
        FROM unnest(v_lex_ids, v_lex_scores) WITH ORDINALITY AS u(id, score, rank)
    ), fused AS (
        SELECT h.id,
               sum(coalesce((v_weights->>h.channel)::float8, 1) / (p_fusion_k + h.rank)) AS score,
               jsonb_agg(jsonb_build_object('channel', h.channel, 'rank', h.rank, 'score', h.score)
                         ORDER BY h.channel) AS hits
        FROM hits h
        GROUP BY h.id
    ), ranked AS (
        SELECT f.*, row_number() OVER (ORDER BY f.score DESC, f.id DESC) AS pos
        FROM fused f
    ), costed AS (
        SELECT r.*, e.scope_id, e.seq, e.content, e.role, e.actor, e.source,
               e.occurred_at, e.ingested_at, e.binding,
               (octet_length(e.content) + p_bytes_per_token - 1) / p_bytes_per_token AS tokens,
               sum((octet_length(e.content) + p_bytes_per_token - 1) / p_bytes_per_token)
                   OVER (ORDER BY r.pos) AS spent
        FROM ranked r
        JOIN mempher.episodes e ON e.id = r.id
        WHERE r.pos <= p_limit
    )
    SELECT coalesce(jsonb_agg(jsonb_build_object(
               'id', c.id, 'scope', c.scope_id, 'seq', c.seq, 'content', c.content,
               'role', c.role, 'actor', c.actor, 'source', c.source,
               'occurred_at', c.occurred_at, 'ingested_at', c.ingested_at, 'binding', c.binding,
               'score', c.score, 'hits', c.hits, 'tokens', c.tokens) ORDER BY c.pos)
               FILTER (WHERE p_max_tokens IS NULL OR v_fact_tokens + c.spent <= p_max_tokens), '[]'),
           coalesce(max(c.spent) FILTER (WHERE p_max_tokens IS NULL OR v_fact_tokens + c.spent <= p_max_tokens), 0),
           v_truncated
               OR coalesce(bool_or(p_max_tokens IS NOT NULL AND v_fact_tokens + c.spent > p_max_tokens), false)
               OR (SELECT count(*) FROM ranked) > p_limit
    INTO v_episodes, v_ep_tokens, v_truncated
    FROM costed c;

    RETURN jsonb_build_object(
        'scope',     p_scope,
        'as_of',     p_as_of,
        'facts',     v_facts,
        'episodes',  v_episodes,
        'channels',  v_reports,
        'tokens',    v_fact_tokens + v_ep_tokens,
        'truncated', v_truncated);
END $$;

COMMENT ON FUNCTION mempher.recall(text, text, vector, text, text, text[], int, int, int, int,
                                   timestamptz, timestamptz, timestamptz, jsonb, text[], text[],
                                   real, jsonb, float8, int) IS
    'Recalls what a scope holds that bears on p_query, as of p_as_of, as one JSON document: '
    '{"facts": [...], "episodes": [...], "channels": [...], "tokens": n, "truncated": bool}. '
    'Channels default to every one the arguments allow: semantic needs p_embedding (the '
    'query embedded by p_model), lexical needs p_query, fact needs p_extractor and ranks '
    'facts by p_embedding as well as p_query when it is given. Episodes are '
    'fused by reciprocal rank, weighted by p_weights (lexical defaults to 0.25), and each '
    'carries the channels that found it. p_max_tokens, counted at p_bytes_per_token, is '
    'spent on facts first, then on episodes in order; what does not fit is left out and '
    'the result says it was truncated.';


-- ---------------------------------------------------------------------------
-- Erasure, and what L0 says is still owed
-- ---------------------------------------------------------------------------

-- mempher.forget is the one function here that destroys anything, and the only
-- one the append-only trigger on mempher.episodes lets through: the declaration
-- that opens the trigger is a SET clause on the function itself, so it holds
-- for exactly the duration of the call and cannot leak into the rest of the
-- caller's transaction.
--
-- mempher.pending_episodes is the safety net behind the queue: it asks L0 and a
-- projection table, never the queue's own history, which episodes still lack
-- an encoding or an extraction.

CREATE FUNCTION mempher.forget(p_scope text, p_episodes uuid[] DEFAULT NULL)
    RETURNS TABLE (episodes int, encodings int, extractions int, facts int, entities int,
                   jobs int, scope_removed boolean)
    LANGUAGE plpgsql
    SET search_path FROM CURRENT
    SET mempher.erasing = 'on'
AS $$
#variable_conflict use_column
DECLARE
    v_whole boolean := p_episodes IS NULL OR cardinality(p_episodes) = 0;
BEGIN
    PERFORM mempher._check_scope(p_scope);
    IF NOT v_whole THEN
        IF cardinality(p_episodes) > 1000 THEN
            PERFORM mempher._fail('MP013', format('forget names %s episodes, limit is 1000',
                cardinality(p_episodes)), 'erase a longer list in several calls, or the whole scope');
        END IF;
        IF array_position(p_episodes, NULL) IS NOT NULL THEN
            PERFORM mempher._fail('MP013', 'forget names a null episode');
        END IF;
    END IF;

    -- An extraction applied while this runs must not resolve a name against an
    -- entity that is on its way out.
    PERFORM mempher._lock_l1(p_scope);

    -- Everything derived goes before what it was derived from. A fact is
    -- deleted rather than closed, because a closed window still says what it
    -- said; if what remains still supports the claim, the next extraction
    -- derives it again. The counts are the receipt.
    --
    -- A name is content too. Erasing a whole scope takes every entity; erasing
    -- episodes takes the entities they introduced that no remaining fact is
    -- about, and from the entities that stay, the other names those episodes
    -- gave them.
    IF v_whole THEN
        DELETE FROM mempher.facts f WHERE f.scope_id = p_scope;
        GET DIAGNOSTICS facts = ROW_COUNT;
        DELETE FROM mempher.entities n WHERE n.scope_id = p_scope;
        GET DIAGNOSTICS entities = ROW_COUNT;
        DELETE FROM mempher.fact_extractions x WHERE x.scope_id = p_scope;
        GET DIAGNOSTICS extractions = ROW_COUNT;
        DELETE FROM mempher.episode_encodings c WHERE c.scope_id = p_scope;
        GET DIAGNOSTICS encodings = ROW_COUNT;
        DELETE FROM mempher.jobs j WHERE j.scope_id = p_scope;
        GET DIAGNOSTICS jobs = ROW_COUNT;
        DELETE FROM mempher.episodes e WHERE e.scope_id = p_scope;
        GET DIAGNOSTICS episodes = ROW_COUNT;
        DELETE FROM mempher.scopes s WHERE s.id = p_scope;
        scope_removed := FOUND;
    ELSE
        DELETE FROM mempher.facts f WHERE f.scope_id = p_scope AND f.episode_ids && p_episodes;
        GET DIAGNOSTICS facts = ROW_COUNT;
        DELETE FROM mempher.entities n
        WHERE n.scope_id = p_scope
          AND EXISTS (SELECT 1 FROM mempher.entity_aliases a
                      WHERE a.entity_id = n.id AND a.episode_ids && p_episodes)
          AND NOT EXISTS (SELECT 1 FROM mempher.facts f
                          WHERE f.scope_id = n.scope_id AND f.extractor = n.extractor
                            AND f.subject = n.name);
        GET DIAGNOSTICS entities = ROW_COUNT;
        DELETE FROM mempher.entity_aliases a USING mempher.entities n
        WHERE a.entity_id = n.id AND a.scope_id = p_scope AND a.episode_ids && p_episodes
          AND a.key <> mempher._entity_key(n.name);
        DELETE FROM mempher.fact_extractions x WHERE x.scope_id = p_scope AND x.episode_id = ANY(p_episodes);
        GET DIAGNOSTICS extractions = ROW_COUNT;
        DELETE FROM mempher.episode_encodings c WHERE c.scope_id = p_scope AND c.episode_id = ANY(p_episodes);
        GET DIAGNOSTICS encodings = ROW_COUNT;
        DELETE FROM mempher.jobs j WHERE j.scope_id = p_scope AND j.episode_ids && p_episodes;
        GET DIAGNOSTICS jobs = ROW_COUNT;
        DELETE FROM mempher.episodes e WHERE e.scope_id = p_scope AND e.id = ANY(p_episodes);
        GET DIAGNOSTICS episodes = ROW_COUNT;
        scope_removed := false;
    END IF;
    RETURN NEXT;
END $$;

COMMENT ON FUNCTION mempher.forget(text, uuid[]) IS
    'Erases a whole scope, or the named episodes of one (at most 1000), together with '
    'every encoding, extraction marker, fact, entity and job derived from them, in one '
    'transaction, and returns what went. Erasing what is not there removes nothing and '
    'is not an error. The only way rows leave mempher.episodes.';

CREATE FUNCTION mempher.pending_episodes(
    p_kind      text,
    p_target    text,
    p_scope     text   DEFAULT NULL,
    p_after_seq bigint DEFAULT 0,
    p_limit     int    DEFAULT 100
)
    RETURNS TABLE (scope_id text, seq bigint, id uuid)
    LANGUAGE plpgsql
    STABLE
    SET search_path FROM CURRENT
AS $$
#variable_conflict use_column
BEGIN
    IF p_kind IS NULL OR p_kind NOT IN ('encode', 'extract') THEN
        PERFORM mempher._fail('MP008', format(
            'pending_episodes answers for encode and extract, not %s',
            coalesce(quote_literal(p_kind), 'null')),
            'facts with no vector are listed by pending_fact_encodings');
    END IF;
    PERFORM mempher._check_label(p_target,
        CASE p_kind WHEN 'encode' THEN 'model' ELSE 'extractor' END, 'MP013', 200);
    IF p_scope IS NOT NULL THEN
        PERFORM mempher._check_scope(p_scope);
    END IF;
    IF p_after_seq IS NULL OR p_after_seq < 0 THEN
        PERFORM mempher._fail('MP013', format('after_seq is %s, must be at least 0', p_after_seq));
    ELSIF p_after_seq > 0 AND p_scope IS NULL THEN
        PERFORM mempher._fail('MP013', 'after_seq needs a scope, because seq is only ordered within one');
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        PERFORM mempher._fail('MP013', format('limit is %s, must be positive', p_limit));
    END IF;

    IF p_kind = 'encode' THEN
        RETURN QUERY
        SELECT e.scope_id, e.seq, e.id FROM mempher.episodes e
        WHERE (p_scope IS NULL OR e.scope_id = p_scope) AND e.seq > p_after_seq
          AND NOT EXISTS (SELECT 1 FROM mempher.episode_encodings x
                          WHERE x.episode_id = e.id AND x.model = p_target)
        ORDER BY e.scope_id, e.seq
        LIMIT p_limit;
    ELSE
        RETURN QUERY
        SELECT e.scope_id, e.seq, e.id FROM mempher.episodes e
        WHERE (p_scope IS NULL OR e.scope_id = p_scope) AND e.seq > p_after_seq
          AND NOT EXISTS (SELECT 1 FROM mempher.fact_extractions x
                          WHERE x.episode_id = e.id AND x.extractor = p_target)
        ORDER BY e.scope_id, e.seq
        LIMIT p_limit;
    END IF;
END $$;

COMMENT ON FUNCTION mempher.pending_episodes(text, text, text, bigint, int) IS
    'Lists episodes with no encoding by model p_target (kind encode), or no extraction '
    'by extractor p_target (kind extract), in scope and seq order: the work L0 implies '
    'whatever the queue remembers. Page within a scope with p_after_seq.';
