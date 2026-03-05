# Memory2 API — Unified Stream

## Core Idea

One type: `Stream[T]`. Everything is a stream — stored, filtered, transformed. The user never thinks about Query vs ObservationSet vs Stream. They just chain operations.

## Creating Streams

```python
store = SqliteStore("/data/robot.db")
session = store.session()

# Root stored stream — backed by DB
images = session.stream("images", Image,
                        pose_provider=lambda: tf.get_pose("world", "base_link"))

logs = session.text_stream("logs", str,
                           pose_provider=lambda: tf.get_pose("world", "base_link"))
```

## Writing

```python
images.append(frame)                    # ts + pose auto-filled
logs.append("Motor fault on joint 3")   # ts + pose auto-filled
images.append(frame, pose=explicit_pose, tags={"cam": "front"})
```

Only meaningful on stored (DB-backed) streams.

## Filtering

Every filter returns a new `Stream[T]`. Lazy — nothing executes until a terminal.

```python
recent = images.after(one_hour_ago)
kitchen = recent.near(kitchen_pose, 5.0)
tagged = kitchen.filter_tags(cam="front")

# Or chained
images.after(one_hour_ago).near(kitchen_pose, 5.0).filter_tags(cam="front")
```

### Filter methods

```python
class Stream(Generic[T]):
    # Temporal
    def after(self, t: float) -> Stream[T]: ...
    def before(self, t: float) -> Stream[T]: ...
    def time_range(self, t1: float, t2: float) -> Stream[T]: ...
    def at(self, t: float, *, tolerance: float = 1.0) -> Stream[T]: ...

    # Spatial
    def near(self, pose: PoseLike, radius: float) -> Stream[T]: ...

    # Tags
    def filter_tags(self, **tags: Any) -> Stream[T]: ...

class EmbeddingStream(Stream[T]):
    def search_embedding(self, query: Embedding | list[float], *, k: int) -> EmbeddingStream[T]: ...

class TextStream(Stream[T]):
    def search_text(self, text: str, *, k: int | None = None) -> TextStream[T]: ...
```

## Terminals & Iteration

`Stream` is directly iterable — pages internally, never loads everything at once.

```python
# Direct iteration (lazy, memory-efficient — uses fetch_pages internally)
for row in images.after(t).near(kitchen_pose, 5.0):
    print(row.data)

# Explicit fetch when you want the full list in memory
all_rows = images.after(t).fetch()

# Other terminals
row = images.after(t).one()                # single best match
row = images.last()                        # most recent
n = images.after(t).count()                # count without fetching

# Pagination
page = images.order_by("ts").limit(50).offset(100).fetch()
```

### Terminal methods

```python
class Stream(Generic[T]):
    def __iter__(self) -> Iterator[Observation]: ...  # lazy, pages internally
    def fetch(self) -> list[Observation]: ...          # all results in memory
    def fetch_pages(self, batch_size: int = 128) -> Iterator[list[Observation]]: ...
    def one(self) -> Observation: ...
    def last(self) -> Observation: ...
    def count(self) -> int: ...
    def order_by(self, field: str, *, desc: bool = False) -> Stream[T]: ...
    def limit(self, k: int) -> Stream[T]: ...
    def offset(self, n: int) -> Stream[T]: ...
```

## Observation

```python
from dimos.models.embedding.base import Embedding, EmbeddingModel

@dataclass
class Observation:
    id: int
    ts: float | None = None
    pose: PoseStamped | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        """Lazy payload. Pre-populated from append/transform, fetched on demand from query."""
        ...
```

## Transformer

A `Transformer` receives the full source stream and decides what to do — which items to process, how to batch, whether to use embeddings as a cheap proxy, etc.

```python
class Transformer(ABC, Generic[T, R]):
    """Transforms a source stream into results on a target stream."""

    def process(self, source: Stream[T], target: Stream[R]) -> None:
        """Batch/historical processing. Has full access to source — can query,
        filter, use embeddings, batch, skip frames, etc."""
        ...

    def on_append(self, obs: Observation, target: Stream[R]) -> None:
        """Reactive processing. Called per new item. Default: process([obs])."""
        ...

    supports_backfill: bool = True
    supports_live: bool = True
```

### Simple lambdas (sugar)

`Callable[[T], R | list[R] | None]` is auto-wrapped into a naive per-item Transformer:

```python
# These are equivalent:
images.transform(lambda img: vlm.detect(img, "cigarettes"))
images.transform(PerItemTransformer(lambda img: vlm.detect(img, "cigarettes")))
```

- `R` → single result
- `list[R]` → multiple results (e.g., multiple detections per frame)
- `None` → skip (no result for this input)

### EmbeddingTransformer

`EmbeddingTransformer` wraps an `EmbeddingModel` as a `Transformer[T, Embedding]`. When the output type is `Embedding`, `.store()` creates an `EmbeddingStream` (vec0 index, `search_embedding`, `EmbeddingObservation`).

```python
# EmbeddingTransformer wraps the model
img_emb = images.transform(EmbeddingTransformer(CLIPModel())).store("img_emb")

# Now img_emb is an EmbeddingStream
results = img_emb.search_embedding(query_emb, k=20).fetch()
# results[0].data → Image (auto-projected from source)
# results[0].embedding → Embedding (supports @ for cosine similarity)
```

### Smart Transformer example

Chains after an embedding transform — receives `EmbeddingObservation` with `.data` (Image) and `.embedding` (vector), so it can use similarity to skip irrelevant frames:

```python
class CigaretteDetector(Transformer[EmbeddingObservation, Detection]):
    def __init__(self, vlm, clip):
        self.vlm = vlm
        self.clip = clip

    def process(self, source: Stream[EmbeddingObservation], target: Stream[Detection]):
        query = self.clip.embed_text("person smoking cigarette")
        for page in source.fetch_pages(batch_size=16):
            # Use embedding similarity as cheap proxy — skip distant frames
            promising = [obs for obs in page if obs.embedding @ query > 0.3]
            if not promising:
                continue
            detections = self.vlm.detect_batch(
                [obs.data for obs in promising], "cigarettes"
            )
            for obs, dets in zip(promising, detections):
                for det in dets:
                    target.append(det, ts=obs.ts, pose=obs.pose)

    def on_append(self, obs: EmbeddingObservation, target: Stream[Detection]):
        dets = self.vlm.detect(obs.data, "cigarettes")
        for det in dets:
            target.append(det, ts=obs.ts, pose=obs.pose)
```

### Chaining transforms

```python
# Filter → transform → store
images.after(one_hour_ago) \
    .near(kitchen_pose, 5.0) \
    .transform(EmbeddingTransformer(CLIPModel())) \
    .store("kitchen_embeddings")

# Filter → transform → fetch (in-memory, not persisted)
results = images.after(one_hour_ago) \
    .near(kitchen_pose, 5.0) \
    .transform(EmbeddingTransformer(CLIPModel())) \
    .fetch()

# Filter → embed → detect → store (chained: detector gets EmbeddingObservation)
images.near(kitchen_pose, 5.0) \
    .transform(EmbeddingTransformer(CLIPModel())) \
    .transform(CigaretteDetector(vlm, clip)) \
    .store("kitchen_cigarette_detections")
```

### Backfill / Live modes

```python
# Both (default): backfill existing + subscribe to new
images.transform(detector).store("detections")

# Live only: skip backfill, only process new items
images.transform(detector, live=True).store("detections")

# Backfill only: process existing, don't subscribe
images.transform(detector, backfill=True).store("detections")

# Backfill only: process existing, and subscribe
images.transform(detector, backfill=True, live=True).store("detections")

# Incremental: re-running a stored transform resumes from last processed item
# (uses lineage parent_id to skip already-processed source rows)
```

## Storing

`.store(name)` materializes a stream to DB. After storing, results are queryable and persistent.

```python
# In-memory transform result — not persisted
detections = images.transform(detect_fn)

# Persist it
detections.store("detections")

# Now it's a DB-backed stream, queryable
stored = session.stream("detections")
rows = stored.after(t).fetch()
```

`.store()` also sets up lineage — every stored row gets `parent_id` pointing back to its source.

Stream type is determined by what the Transformer produces:
- `Embedding` output → `EmbeddingStream` (vec0 index)
- Everything else → `Stream` (blob)
- `TextStream` is created explicitly via `session.text_stream()` (not auto-detected)

## Reactive

```python
# .appended emits Observation with .data pre-populated
images.appended.subscribe(lambda row: print(f"New image at {row.pose}"))

# Stored transforms propagate reactively by default
detections = images.transform(detect_fn).store("detections")
# Now every images.append(frame) → detect_fn runs → result stored in "detections"

# Filtered appended — only kitchen images
images.near(kitchen_pose, 5.0).appended.subscribe(...)
```

## Join (cross-stream lineage)

```python
# Join detections with their source images — returns tuples
for det, img in detections.after(t).join(images):
    print(f"Detected {det.data} in image at {img.pose}")
```

## Full Example: Cigarette Detection Pipeline

```python
session = SqliteStore("/data/robot.db").session()

# Root stream
images = session.stream("images", Image,
                        pose_provider=lambda: tf.get_pose("world", "base_link"))

# Embedding index — EmbeddingModel is a Transformer
img_emb = images.transform(EmbeddingTransformer(CLIPModel())).store("img_emb")

# VLM detection pipeline (live-only, no backfill)
images.transform(
    lambda img: vlm.detect(img, "people with cigarettes"),
    live=True,
).store("cigarette_detections")

# Smart detection — reuse existing embeddings, detector gets EmbeddingObservation
img_emb.near(kitchen_pose, 10.0) \
    .transform(CigaretteDetector(vlm, clip)) \
    .store("kitchen_cigarette_detections")

# # Worse: re-embeds from scratch (redundant if img_emb already exists)
# images.near(kitchen_pose, 10.0) \
#     .transform(EmbeddingTransformer(CLIPModel())) \
#     .transform(CigaretteDetector(vlm, clip)) \
#     .store("kitchen_cigarette_detections")

# --- Later, querying ---

# "Where did we see people with cigarettes in the kitchen?"
for row in session.stream("cigarette_detections") \
        .after(one_hour_ago).near(kitchen_pose, 10.0):
    print(f"t={row.ts} pose={row.pose}: {row.data}")

# "Show me the source images alongside detections"
for det, img in session.stream("cigarette_detections") \
        .after(one_hour_ago).join(images):
    print(f"Detection: {det.data}, Source image at {img.pose}")

# "Find images similar to 'red shoes'"
query_emb = clip.embed_text("red shoes")
similar = img_emb.search_embedding(query_emb, k=20).fetch()
# similar[0].data → Image (auto-projected from source)
# similar[0].embedding → Embedding (supports @ for cosine similarity)
```

## Full API

```python
from dimos.models.embedding.base import Embedding, EmbeddingModel

# --- Data types ---

@dataclass
class Observation:
    id: int
    ts: float | None = None
    pose: PoseStamped | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        """Lazy payload. Pre-populated from append, fetched on demand from query."""
        ...

@dataclass
class EmbeddedObservation(Observation):
    """Returned by EmbeddingStream terminals. Auto-projects .data to source stream."""

    @property
    def data(self) -> Any:
        """Lazily loads from the source stream (e.g., Image), not the embedding."""
        ...

    @property
    def embedding(self) -> Embedding:
        """The Embedding object (has .vector, supports @ for cosine similarity)."""
        ...

# --- Transformer ---

class Transformer(ABC, Generic[T, R]):
    """Transforms a source stream into results on a target stream."""

    def process(self, source: Stream[T], target: Stream[R]) -> None:
        """Batch/historical processing. Full access to source stream."""
        ...

    def on_append(self, obs: Observation, target: Stream[R]) -> None:
        """Reactive processing. Called per new item."""
        ...

    supports_backfill: bool = True
    supports_live: bool = True

# --- Streams ---

class Stream(Generic[T]):
    # Write (DB-backed only)
    def append(self, payload: T, *,
               ts: float | None = None,
               pose: PoseLike | None = None,
               tags: dict[str, Any] | None = None,
               ) -> Observation: ...

    # Filter (returns new Stream, lazy)
    def after(self, t: float) -> Stream[T]: ...
    def before(self, t: float) -> Stream[T]: ...
    def time_range(self, t1: float, t2: float) -> Stream[T]: ...
    def at(self, t: float, *, tolerance: float = 1.0) -> Stream[T]: ...
    def near(self, pose: PoseLike, radius: float) -> Stream[T]: ...
    def filter_tags(self, **tags: Any) -> Stream[T]: ...

    # Order / paginate
    def order_by(self, field: str, *, desc: bool = False) -> Stream[T]: ...
    def limit(self, k: int) -> Stream[T]: ...
    def offset(self, n: int) -> Stream[T]: ...

    # Transform
    def transform(self,
                  xf: Transformer[T, R] | Callable[[T], R | list[R] | None],
                  *, live: bool = False,
                  backfill_only: bool = False,
                  ) -> Stream[R]: ...

    # Materialize (on TransformStream, accepts optional session= fallback)
    def store(self, name: str | None = None, session: Session | None = None) -> Stream[T]: ...

    # Cross-stream (lineage join — returns tuples of (self_obs, target_obs))
    def join(self, target: Stream) -> Stream[tuple[Observation, Observation]]: ...

    # Iteration & Terminals
    def __iter__(self) -> Iterator[Observation]: ...  # lazy, pages internally
    def fetch(self) -> list[Observation]: ...          # all results in memory
    def fetch_pages(self, batch_size: int = 128) -> Iterator[list[Observation]]: ...
    def one(self) -> Observation: ...
    def last(self) -> Observation: ...
    def count(self) -> int: ...

    # Reactive
    @property
    def appended(self) -> Observable[Observation]: ...

class EmbeddingStream(Stream[T]):
    """Created automatically when a Transformer produces Embedding output.
    Terminals return EmbeddedObservation (auto-projects .data to source stream)."""
    def search_embedding(self, query: Embedding | list[float], *, k: int) -> EmbeddingStream[T]: ...
    def fetch(self) -> list[EmbeddedObservation]: ...
    def one(self) -> EmbeddedObservation: ...
    def last(self) -> EmbeddedObservation: ...

class TextStream(Stream[T]):
    """Stream with FTS index."""
    def search_text(self, text: str, *, k: int | None = None) -> TextStream[T]: ...

# --- Session / Store ---

PoseProvider = Callable[[], PoseLike | None]

class Session:
    def stream(self, name: str, payload_type: type | None = None, *,
               pose_provider: PoseProvider | None = None) -> Stream: ...
    def text_stream(self, name: str, payload_type: type | None = None, *,
                    tokenizer: str = "unicode61",
                    pose_provider: PoseProvider | None = None) -> TextStream: ...
    def materialize_transform(self, name: str, source: Stream,
                              transformer: Transformer,
                              *, live: bool = False,
                              backfill_only: bool = False) -> Stream: ...
    def list_streams(self) -> list[StreamInfo]: ...
    def close(self) -> None: ...

class Store:
    def session(self) -> Session: ...
    def close(self) -> None: ...
```

## Internal Backing (impl detail)

A `Stream` can be backed by different things — the user never sees this:

- **DB table** — from `session.stream()`. Has `_meta`, `_payload`, indexes.
- **Predicate** — from `.after()`, `.near()`, etc. Lazy SQL WHERE.
- **Transform** — from `.transform(t)`. Source stream + Transformer.

The impl decides how to execute based on the backing chain.

## Implementation Notes

- **No ORM** — raw `sqlite3` with direct SQL. The `Stream` filter chain *is* the query builder.
- **Session threading** — streams created by `session.stream()` get `_session` set. `TransformStream` inherits it from its source. `store()` also accepts an explicit `session=` fallback.
- **Serialization** — payloads are `pickle`, poses are `pickle`, tags are JSON.
- **Near filter** — compiled as no-op SQL (`1=1`), filtered post-query in Python via pose distance.

## Resolved Questions

1. **`.append()` on non-stored streams?** → `TypeError` (requires backend).
2. **Multiple `.store()` calls?** → Idempotent — returns existing stream if already stored.
3. ~~**Memory pressure from in-memory transforms?**~~ → Solved via `fetch_pages`.

## Open Questions

1. **`project_to` / lineage** — `parent_id` column exists but not yet wired.
2. **Incremental transforms** — re-running a stored transform should resume from last processed item.
3. **`__iter__`** — spec shows `Stream.__iter__` but not yet implemented.
