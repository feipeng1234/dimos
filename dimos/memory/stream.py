# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    overload,
)

from .types import (
    AfterFilter,
    AtFilter,
    BeforeFilter,
    EmbeddingObservation,
    EmbeddingSearchFilter,
    Filter,
    NearFilter,
    Observation,
    StreamQuery,
    TagsFilter,
    TextSearchFilter,
    TimeRangeFilter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from reactivex import Observable
    from reactivex.subject import Subject

    from dimos.models.embedding.base import Embedding
    from dimos.msgs.geometry_msgs.Pose import PoseLike

    from .store import Session
    from .transformer import Transformer

T = TypeVar("T")
R = TypeVar("R")


class StreamBackend(Protocol):
    """Backend protocol — implemented by SqliteStreamBackend etc."""

    def execute_fetch(self, query: StreamQuery) -> list[Observation]: ...
    def execute_count(self, query: StreamQuery) -> int: ...
    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
        parent_id: int | None = None,
    ) -> Observation: ...
    @property
    def appended_subject(self) -> Subject[Observation]: ...  # type: ignore[type-arg]
    @property
    def stream_name(self) -> str: ...


class Stream(Generic[T]):
    """Lazy, chainable stream over stored observations.

    Created by Session.stream(). Filter methods return new Stream instances.
    Terminals (.fetch(), .count(), etc.) execute the query.
    """

    def __init__(
        self,
        backend: StreamBackend | None = None,
        *,
        query: StreamQuery | None = None,
        session: Session | None = None,
    ) -> None:
        self._backend = backend
        self._query = query or StreamQuery()
        self._session: Session | None = session

    def _clone(self, **overrides: Any) -> Stream[T]:
        """Return a new Stream with updated query fields."""
        q = self._query
        new_query = StreamQuery(
            filters=overrides.get("filters", q.filters),
            order_field=overrides.get("order_field", q.order_field),
            order_desc=overrides.get("order_desc", q.order_desc),
            limit_val=overrides.get("limit_val", q.limit_val),
            offset_val=overrides.get("offset_val", q.offset_val),
        )
        clone: Stream[T] = self.__class__.__new__(self.__class__)
        clone._backend = self._backend
        clone._query = new_query
        clone._session = self._session
        return clone

    def _with_filter(self, f: Filter) -> Stream[T]:
        return self._clone(filters=(*self._query.filters, f))

    def _require_backend(self) -> StreamBackend:
        if self._backend is None:
            raise TypeError(
                "Operation requires a stored stream. Call .store() first or use session.stream()."
            )
        return self._backend

    # ── Write ─────────────────────────────────────────────────────────

    def append(
        self,
        payload: T,
        *,
        ts: float | None = None,
        pose: PoseLike | None = None,
        tags: dict[str, Any] | None = None,
        parent_id: int | None = None,
    ) -> Observation:
        backend = self._require_backend()
        return backend.do_append(payload, ts, pose, tags, parent_id)

    # ── Temporal filters ──────────────────────────────────────────────

    def after(self, t: float) -> Stream[T]:
        return self._with_filter(AfterFilter(t))

    def before(self, t: float) -> Stream[T]:
        return self._with_filter(BeforeFilter(t))

    def time_range(self, t1: float, t2: float) -> Stream[T]:
        return self._with_filter(TimeRangeFilter(t1, t2))

    def at(self, t: float, *, tolerance: float = 1.0) -> Stream[T]:
        return self._with_filter(AtFilter(t, tolerance))

    # ── Spatial filter ────────────────────────────────────────────────

    def near(self, pose: PoseLike, radius: float) -> Stream[T]:
        return self._with_filter(NearFilter(pose, radius))

    # ── Tag filter ────────────────────────────────────────────────────

    def filter_tags(self, **tags: Any) -> Stream[T]:
        return self._with_filter(TagsFilter(tags))

    # ── Ordering / pagination ─────────────────────────────────────────

    def order_by(self, field: str, *, desc: bool = False) -> Stream[T]:
        return self._clone(order_field=field, order_desc=desc)

    def limit(self, k: int) -> Stream[T]:
        return self._clone(limit_val=k)

    def offset(self, n: int) -> Stream[T]:
        return self._clone(offset_val=n)

    # ── Transform ─────────────────────────────────────────────────────

    @overload
    def transform(
        self,
        xf: Transformer[T, R],
        *,
        live: bool = ...,
        backfill_only: bool = ...,
    ) -> Stream[R]: ...

    @overload
    def transform(
        self,
        xf: Callable[[T], Any],
        *,
        live: bool = ...,
        backfill_only: bool = ...,
    ) -> Stream[Any]: ...

    def transform(
        self,
        xf: Transformer[Any, Any] | Callable[..., Any],
        *,
        live: bool = False,
        backfill_only: bool = False,
    ) -> Stream[Any]:
        from .transformer import PerItemTransformer, Transformer as TransformerABC

        transformer: TransformerABC[Any, Any]
        if not isinstance(xf, TransformerABC):
            transformer = PerItemTransformer(xf)
        else:
            transformer = xf

        return TransformStream(
            source=self,
            transformer=transformer,
            live=live,
            backfill_only=backfill_only,
        )

    # ── Materialize ───────────────────────────────────────────────────

    def store(self, name: str | None = None) -> Stream[T]:
        # Already stored streams are a no-op
        if self._backend is not None and name is None:
            return self
        raise TypeError(
            "store() requires a session context. This stream is not associated with a session."
        )

    # ── Cross-stream lineage ──────────────────────────────────────────

    def project_to(self, target: Stream[Any]) -> Stream[Any]:
        raise NotImplementedError("project_to requires a stored stream with lineage")

    # ── Iteration ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Observation]:
        for page in self.fetch_pages():
            yield from page

    # ── Terminals ─────────────────────────────────────────────────────

    def fetch(self) -> list[Observation]:
        backend = self._require_backend()
        return backend.execute_fetch(self._query)

    def fetch_pages(self, batch_size: int = 128) -> Iterator[list[Observation]]:
        offset = self._query.offset_val or 0
        while True:
            q = StreamQuery(
                filters=self._query.filters,
                order_field=self._query.order_field or "id",
                order_desc=self._query.order_desc,
                limit_val=batch_size,
                offset_val=offset,
            )
            backend = self._require_backend()
            page = backend.execute_fetch(q)
            if not page:
                break
            yield page
            if len(page) < batch_size:
                break
            offset += batch_size

    def one(self) -> Observation:
        results = self.limit(1).fetch()
        if not results:
            raise LookupError("No matching observation")
        return results[0]

    def last(self) -> Observation:
        results = self.order_by("ts", desc=True).limit(1).fetch()
        if not results:
            raise LookupError("No matching observation")
        return results[0]

    def count(self) -> int:
        backend = self._require_backend()
        return backend.execute_count(self._query)

    # ── Reactive ──────────────────────────────────────────────────────

    @property
    def appended(self) -> Observable[Observation]:  # type: ignore[type-arg]
        backend = self._require_backend()
        return backend.appended_subject  # type: ignore[return-value]


class EmbeddingStream(Stream[T]):
    """Stream with a vector index. Adds search_embedding()."""

    def search_embedding(
        self,
        query: Embedding | list[float],
        *,
        k: int,
    ) -> EmbeddingStream[T]:
        from dimos.models.embedding.base import Embedding as EmbeddingCls

        if isinstance(query, EmbeddingCls):
            vec = query.to_numpy().tolist()
        else:
            vec = list(query)
        clone = self._with_filter(EmbeddingSearchFilter(vec, k))
        # Preserve EmbeddingStream type
        es: EmbeddingStream[T] = EmbeddingStream(backend=clone._backend, query=clone._query)
        return es

    def fetch(self) -> list[EmbeddingObservation]:  # type: ignore[override]
        backend = self._require_backend()
        return backend.execute_fetch(self._query)  # type: ignore[return-value]

    def one(self) -> EmbeddingObservation:  # type: ignore[override]
        q = StreamQuery(
            filters=self._query.filters,
            order_field=self._query.order_field,
            order_desc=self._query.order_desc,
            limit_val=1,
            offset_val=self._query.offset_val,
        )
        backend = self._require_backend()
        results = backend.execute_fetch(q)
        if not results:
            raise LookupError("No matching observation")
        return results[0]  # type: ignore[return-value]

    def last(self) -> EmbeddingObservation:  # type: ignore[override]
        q = StreamQuery(
            filters=self._query.filters,
            order_field="ts",
            order_desc=True,
            limit_val=1,
            offset_val=self._query.offset_val,
        )
        backend = self._require_backend()
        results = backend.execute_fetch(q)
        if not results:
            raise LookupError("No matching observation")
        return results[0]  # type: ignore[return-value]


class TextStream(Stream[T]):
    """Stream with an FTS5 index. Adds search_text()."""

    def search_text(self, text: str, *, k: int | None = None) -> TextStream[T]:
        clone = self._with_filter(TextSearchFilter(text, k))
        ts: TextStream[T] = TextStream(backend=clone._backend, query=clone._query)
        return ts


class TransformStream(Stream[R]):
    """In-memory stream produced by .transform(). Not yet stored."""

    def __init__(
        self,
        source: Stream[Any],
        transformer: Transformer[Any, R],
        *,
        live: bool = False,
        backfill_only: bool = False,
    ) -> None:
        super().__init__(backend=None)
        self._source = source
        self._transformer = transformer
        self._live = live
        self._backfill_only = backfill_only

    def fetch(self) -> list[Observation]:
        """Execute transform in memory, collecting results."""
        collector = _CollectorStream[R]()
        if self._transformer.supports_backfill and not self._live:
            self._transformer.process(self._source, collector)
        return collector.results

    def store(
        self,
        name: str | None = None,
        payload_type: type | None = None,
        session: Session | None = None,
    ) -> Stream[R]:
        resolved = session or self._source._session
        if resolved is None:
            raise TypeError(
                "Cannot store: no session available. "
                "Either use session.stream() to create the source, "
                "or pass session= to store()."
            )
        if name is None:
            raise TypeError("store() requires a name for transform outputs")
        resolved_type = payload_type or self._transformer.output_type
        return resolved.materialize_transform(
            name=name,
            source=self._source,
            transformer=self._transformer,
            payload_type=resolved_type,
            live=self._live,
            backfill_only=self._backfill_only,
        )


class _CollectorStream(Stream[R]):
    """Ephemeral stream that collects appended observations in a list."""

    def __init__(self) -> None:
        super().__init__(backend=None)
        self.results: list[Observation] = []
        self._next_id = 0

    def append(
        self,
        payload: R,
        *,
        ts: float | None = None,
        pose: PoseLike | None = None,
        tags: dict[str, Any] | None = None,
        parent_id: int | None = None,
    ) -> Observation:
        obs = Observation(
            id=self._next_id,
            ts=ts,
            tags=tags or {},
            parent_id=parent_id,
            _data=payload,
        )
        self._next_id += 1
        self.results.append(obs)
        return obs
