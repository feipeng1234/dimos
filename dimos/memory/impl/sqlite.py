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

"""SQLite-backed memory store implementation.

Each stream maps to a table:
    {name}       — id INTEGER PK, ts REAL, pose BLOB, tags TEXT (JSON), payload BLOB
    {name}_fts   — FTS5 virtual table (TextStream only)
    {name}_vec   — vec0 virtual table (EmbeddingStream only)

Payloads are pickled. Poses are pickled PoseStamped. Tags are JSON.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from reactivex.subject import Subject

from dimos.memory.store import Session, Store
from dimos.memory.stream import EmbeddingStream, Stream, TextStream
from dimos.memory.transformer import EmbeddingTransformer, Transformer
from dimos.memory.types import (
    AfterFilter,
    AtFilter,
    BeforeFilter,
    EmbeddingObservation,
    EmbeddingSearchFilter,
    Filter,
    NearFilter,
    Observation,
    StreamInfo,
    StreamQuery,
    TagsFilter,
    TextSearchFilter,
    TimeRangeFilter,
)

if TYPE_CHECKING:
    from dimos.memory.types import PoseProvider


# ── Serialization helpers ─────────────────────────────────────────────


def _serialize_payload(payload: Any) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_payload(blob: bytes) -> Any:
    return pickle.loads(blob)


def _serialize_pose(pose: Any) -> bytes | None:
    if pose is None:
        return None
    return pickle.dumps(pose, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_pose(blob: bytes | None) -> Any:
    if blob is None:
        return None
    return pickle.loads(blob)


def _serialize_tags(tags: dict[str, Any] | None) -> str:
    if not tags:
        return "{}"
    return json.dumps(tags, separators=(",", ":"))


def _deserialize_tags(text: str) -> dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)  # type: ignore[no-any-return]


# ── SQL building ──────────────────────────────────────────────────────


def _compile_filter(f: Filter, table: str) -> tuple[str, list[Any]]:
    """Compile a single filter to (SQL fragment, params)."""
    if isinstance(f, AfterFilter):
        return "ts > ?", [f.t]
    if isinstance(f, BeforeFilter):
        return "ts < ?", [f.t]
    if isinstance(f, TimeRangeFilter):
        return "ts >= ? AND ts <= ?", [f.t1, f.t2]
    if isinstance(f, AtFilter):
        return "ABS(ts - ?) <= ?", [f.t, f.tolerance]
    if isinstance(f, TagsFilter):
        clauses: list[str] = []
        params: list[Any] = []
        for key, val in f.tags.items():
            clauses.append(f"json_extract(tags, '$.{key}') = ?")
            params.append(val)
        return " AND ".join(clauses), params
    if isinstance(f, NearFilter):
        # Spatial filtering requires pose deserialization — done post-query
        # Return a no-op SQL clause; filtering happens in Python
        return "1=1", []
    if isinstance(f, EmbeddingSearchFilter):
        # Handled specially by EmbeddingStream backend
        return "1=1", []
    if isinstance(f, TextSearchFilter):
        # Handled specially by TextStream backend
        return "1=1", []
    raise TypeError(f"Unknown filter type: {type(f)}")


def _compile_query(query: StreamQuery, table: str) -> tuple[str, list[Any]]:
    """Compile a StreamQuery to (SQL, params) for a SELECT."""
    where_parts: list[str] = []
    params: list[Any] = []

    for f in query.filters:
        sql, p = _compile_filter(f, table)
        where_parts.append(sql)
        params.extend(p)

    where = " AND ".join(where_parts) if where_parts else "1=1"
    order = f"ORDER BY {query.order_field}"
    if query.order_field:
        if query.order_desc:
            order += " DESC"
    else:
        order = "ORDER BY id"

    sql = f"SELECT id, ts, pose, tags, payload FROM {table} WHERE {where} {order}"
    if query.limit_val is not None:
        sql += f" LIMIT {query.limit_val}"
    if query.offset_val is not None:
        sql += f" OFFSET {query.offset_val}"
    return sql, params


def _compile_count(query: StreamQuery, table: str) -> tuple[str, list[Any]]:
    where_parts: list[str] = []
    params: list[Any] = []
    for f in query.filters:
        sql, p = _compile_filter(f, table)
        where_parts.append(sql)
        params.extend(p)
    where = " AND ".join(where_parts) if where_parts else "1=1"
    return f"SELECT COUNT(*) FROM {table} WHERE {where}", params


# ── Near-filter post-processing ───────────────────────────────────────


def _has_near_filter(query: StreamQuery) -> NearFilter | None:
    for f in query.filters:
        if isinstance(f, NearFilter):
            return f
    return None


def _apply_near_filter(rows: list[Observation], near: NearFilter) -> list[Observation]:
    """Post-filter observations by spatial distance."""
    from dimos.msgs.geometry_msgs.Pose import to_pose

    target = to_pose(near.pose)
    result: list[Observation] = []
    for obs in rows:
        if obs.pose is None:
            continue
        obs_pose = to_pose(obs.pose)
        dist = (target - obs_pose).position.norm()
        if dist <= near.radius:
            result.append(obs)
    return result


# ── Backend ───────────────────────────────────────────────────────────


class SqliteStreamBackend:
    """StreamBackend implementation for a single SQLite-backed stream."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        table: str,
        *,
        pose_provider: PoseProvider | None = None,
    ) -> None:
        self._conn = conn
        self._table = table
        self._pose_provider = pose_provider
        self._subject: Subject[Observation] = Subject()  # type: ignore[type-arg]

    @property
    def appended_subject(self) -> Subject[Observation]:  # type: ignore[type-arg]
        return self._subject

    @property
    def stream_name(self) -> str:
        return self._table

    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
    ) -> Observation:
        if ts is None:
            ts = time.time()
        if pose is None and self._pose_provider is not None:
            pose = self._pose_provider()

        payload_blob = _serialize_payload(payload)
        pose_blob = _serialize_pose(pose)
        tags_json = _serialize_tags(tags)

        cur = self._conn.execute(
            f"INSERT INTO {self._table} (ts, pose, tags, payload) VALUES (?, ?, ?, ?)",
            (ts, pose_blob, tags_json, payload_blob),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        assert row_id is not None

        obs = Observation(
            id=row_id,
            ts=ts,
            pose=pose,
            tags=tags or {},
            _data=payload,
        )
        self._subject.on_next(obs)
        return obs

    def execute_fetch(self, query: StreamQuery) -> list[Observation]:
        sql, params = _compile_query(query, self._table)
        rows = self._conn.execute(sql, params).fetchall()

        observations = [self._row_to_obs(r) for r in rows]

        near = _has_near_filter(query)
        if near is not None:
            observations = _apply_near_filter(observations, near)

        return observations

    def execute_count(self, query: StreamQuery) -> int:
        sql, params = _compile_count(query, self._table)
        result = self._conn.execute(sql, params).fetchone()
        return result[0] if result else 0  # type: ignore[no-any-return]

    def _row_to_obs(self, row: Any) -> Observation:
        row_id, ts, pose_blob, tags_json, payload_blob = row
        return Observation(
            id=row_id,
            ts=ts,
            pose=_deserialize_pose(pose_blob),
            tags=_deserialize_tags(tags_json),
            _data=_deserialize_payload(payload_blob),
        )


class SqliteEmbeddingBackend(SqliteStreamBackend):
    """Backend for EmbeddingStream — stores vectors in a vec0 virtual table."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        table: str,
        *,
        vec_dimensions: int | None = None,
        pose_provider: PoseProvider | None = None,
        parent_table: str | None = None,
    ) -> None:
        super().__init__(conn, table, pose_provider=pose_provider)
        self._vec_dimensions = vec_dimensions
        self._parent_table = parent_table

    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
    ) -> Observation:
        from dimos.models.embedding.base import Embedding

        obs = super().do_append(payload, ts, pose, tags)

        # Also insert into vec0 table if payload is an Embedding
        if isinstance(payload, Embedding):
            vec = payload.to_numpy().tolist()
            if self._vec_dimensions is None:
                self._vec_dimensions = len(vec)
                self._ensure_vec_table()
            self._conn.execute(
                f"INSERT INTO {self._table}_vec (rowid, embedding) VALUES (?, ?)",
                (obs.id, json.dumps(vec)),
            )
            self._conn.commit()

        return obs

    def _ensure_vec_table(self) -> None:
        if self._vec_dimensions is None:
            return
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec "
            f"USING vec0(embedding float[{self._vec_dimensions}])"
        )
        self._conn.commit()

    def execute_fetch(self, query: StreamQuery) -> list[Observation]:
        # Check for embedding search filter
        emb_filter = None
        for f in query.filters:
            if isinstance(f, EmbeddingSearchFilter):
                emb_filter = f
                break

        if emb_filter is not None:
            return self._fetch_by_vector(query, emb_filter)

        return super().execute_fetch(query)

    def _fetch_by_vector(
        self, query: StreamQuery, emb_filter: EmbeddingSearchFilter
    ) -> list[Observation]:
        """Fetch using vec0 similarity search, then apply remaining filters."""
        # First, get candidate rowids from vec0
        vec_sql = (
            f"SELECT rowid, distance FROM {self._table}_vec "
            f"WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
        )
        vec_rows = self._conn.execute(
            vec_sql, (json.dumps(emb_filter.query), emb_filter.k)
        ).fetchall()

        if not vec_rows:
            return []

        rowids = [r[0] for r in vec_rows]
        placeholders = ",".join("?" * len(rowids))

        # Build remaining WHERE clauses (skip the embedding filter)
        where_parts: list[str] = [f"id IN ({placeholders})"]
        params: list[Any] = list(rowids)

        for f in query.filters:
            if isinstance(f, EmbeddingSearchFilter):
                continue
            sql_frag, p = _compile_filter(f, self._table)
            where_parts.append(sql_frag)
            params.extend(p)

        where = " AND ".join(where_parts)
        sql = f"SELECT id, ts, pose, tags, payload FROM {self._table} WHERE {where}"
        rows = self._conn.execute(sql, params).fetchall()

        observations = [self._row_to_obs(r) for r in rows]

        near = _has_near_filter(query)
        if near is not None:
            observations = _apply_near_filter(observations, near)

        return observations

    def _row_to_obs(self, row: Any) -> Observation:
        row_id, ts, pose_blob, tags_json, payload_blob = row
        return EmbeddingObservation(
            id=row_id,
            ts=ts,
            pose=_deserialize_pose(pose_blob),
            tags=_deserialize_tags(tags_json),
            _data=_deserialize_payload(payload_blob),
        )


class SqliteTextBackend(SqliteStreamBackend):
    """Backend for TextStream — maintains an FTS5 index."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        table: str,
        *,
        tokenizer: str = "unicode61",
        pose_provider: PoseProvider | None = None,
    ) -> None:
        super().__init__(conn, table, pose_provider=pose_provider)
        self._tokenizer = tokenizer

    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
    ) -> Observation:
        obs = super().do_append(payload, ts, pose, tags)

        # Insert into FTS table
        text = str(payload) if payload is not None else ""
        self._conn.execute(
            f"INSERT INTO {self._table}_fts (rowid, content) VALUES (?, ?)",
            (obs.id, text),
        )
        self._conn.commit()
        return obs

    def execute_fetch(self, query: StreamQuery) -> list[Observation]:
        text_filter = None
        for f in query.filters:
            if isinstance(f, TextSearchFilter):
                text_filter = f
                break

        if text_filter is not None:
            return self._fetch_by_text(query, text_filter)

        return super().execute_fetch(query)

    def _fetch_by_text(
        self, query: StreamQuery, text_filter: TextSearchFilter
    ) -> list[Observation]:
        # Get matching rowids from FTS
        fts_sql = f"SELECT rowid, rank FROM {self._table}_fts WHERE content MATCH ? ORDER BY rank"
        fts_params: list[Any] = [text_filter.text]
        if text_filter.k is not None:
            fts_sql += " LIMIT ?"
            fts_params.append(text_filter.k)

        fts_rows = self._conn.execute(fts_sql, fts_params).fetchall()
        if not fts_rows:
            return []

        rowids = [r[0] for r in fts_rows]
        placeholders = ",".join("?" * len(rowids))

        where_parts: list[str] = [f"id IN ({placeholders})"]
        params: list[Any] = list(rowids)

        for f in query.filters:
            if isinstance(f, TextSearchFilter):
                continue
            sql_frag, p = _compile_filter(f, self._table)
            where_parts.append(sql_frag)
            params.extend(p)

        where = " AND ".join(where_parts)
        sql = f"SELECT id, ts, pose, tags, payload FROM {self._table} WHERE {where}"
        rows = self._conn.execute(sql, params).fetchall()

        observations = [self._row_to_obs(r) for r in rows]

        near = _has_near_filter(query)
        if near is not None:
            observations = _apply_near_filter(observations, near)

        return observations


# ── Session ───────────────────────────────────────────────────────────


class SqliteSession(Session):
    """Session against a SQLite database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._streams: dict[str, Stream[Any]] = {}
        self._ensure_meta_table()

    def _ensure_meta_table(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS _streams ("
            "  name TEXT PRIMARY KEY,"
            "  payload_type TEXT,"
            "  stream_kind TEXT DEFAULT 'stream'"
            ")"
        )
        self._conn.commit()

    def stream(
        self,
        name: str,
        payload_type: type | None = None,
        *,
        pose_provider: PoseProvider | None = None,
    ) -> Stream[Any]:
        if name in self._streams:
            return self._streams[name]

        self._ensure_stream_table(name)
        self._register_stream(name, payload_type, "stream")

        backend = SqliteStreamBackend(self._conn, name, pose_provider=pose_provider)
        s: Stream[Any] = Stream(backend=backend, session=self)
        self._streams[name] = s
        return s

    def text_stream(
        self,
        name: str,
        payload_type: type | None = None,
        *,
        tokenizer: str = "unicode61",
        pose_provider: PoseProvider | None = None,
    ) -> TextStream[Any]:
        if name in self._streams:
            return self._streams[name]  # type: ignore[return-value]

        self._ensure_stream_table(name)
        self._ensure_fts_table(name, tokenizer)
        self._register_stream(name, payload_type, "text")

        backend = SqliteTextBackend(
            self._conn, name, tokenizer=tokenizer, pose_provider=pose_provider
        )
        ts: TextStream[Any] = TextStream(backend=backend, session=self)
        self._streams[name] = ts
        return ts

    def embedding_stream(
        self,
        name: str,
        payload_type: type | None = None,
        *,
        vec_dimensions: int | None = None,
        pose_provider: PoseProvider | None = None,
        parent_table: str | None = None,
    ) -> EmbeddingStream[Any]:
        if name in self._streams:
            return self._streams[name]  # type: ignore[return-value]

        self._ensure_stream_table(name)
        self._register_stream(name, payload_type, "embedding")

        backend = SqliteEmbeddingBackend(
            self._conn,
            name,
            vec_dimensions=vec_dimensions,
            pose_provider=pose_provider,
            parent_table=parent_table,
        )
        if vec_dimensions is not None:
            backend._ensure_vec_table()

        es: EmbeddingStream[Any] = EmbeddingStream(backend=backend, session=self)
        self._streams[name] = es
        return es

    def list_streams(self) -> list[StreamInfo]:
        rows = self._conn.execute("SELECT name, payload_type FROM _streams").fetchall()
        result: list[StreamInfo] = []
        for name, ptype in rows:
            count_row = self._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
            count = count_row[0] if count_row else 0
            result.append(StreamInfo(name=name, payload_type=ptype, count=count))
        return result

    def materialize_transform(
        self,
        name: str,
        source: Stream[Any],
        transformer: Transformer[Any, Any],
        *,
        live: bool = False,
        backfill_only: bool = False,
    ) -> Stream[Any]:
        # Determine stream type from transformer
        target: Stream[Any]
        if isinstance(transformer, EmbeddingTransformer):
            target = self.embedding_stream(name)
        else:
            target = self.stream(name)

        # Backfill existing data
        if transformer.supports_backfill and not live:
            transformer.process(source, target)

        # Subscribe to live updates
        if transformer.supports_live and not backfill_only:
            source.appended.subscribe(on_next=lambda obs: transformer.on_append(obs, target))

        return target

    def close(self) -> None:
        for s in self._streams.values():
            if s._backend is not None:
                s._backend.appended_subject.on_completed()
        self._streams.clear()

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_stream_table(self, name: str) -> None:
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name} ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts REAL,"
            "  pose BLOB,"
            "  tags TEXT DEFAULT '{}',"
            "  payload BLOB,"
            "  parent_id INTEGER"
            ")"
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_ts ON {name}(ts)")
        self._conn.commit()

    def _ensure_fts_table(self, name: str, tokenizer: str) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name}_fts "
            f"USING fts5(content, tokenize='{tokenizer}')"
        )
        self._conn.commit()

    def _register_stream(self, name: str, payload_type: type | None, kind: str) -> None:
        type_name = payload_type.__qualname__ if payload_type else None
        self._conn.execute(
            "INSERT OR IGNORE INTO _streams (name, payload_type, stream_kind) VALUES (?, ?, ?)",
            (name, type_name, kind),
        )
        self._conn.commit()


# ── Store ─────────────────────────────────────────────────────────────


class SqliteStore(Store):
    """SQLite-backed memory store."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def session(self) -> SqliteSession:
        return SqliteSession(self._conn)

    def close(self) -> None:
        self._conn.close()
