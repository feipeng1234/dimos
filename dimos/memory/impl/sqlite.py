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

Schema per stream ``{name}``:

    {name}          — id, ts, pose columns (x/y/z + quaternion), tags, parent_id
    {name}_payload  — id, data BLOB (loaded lazily)
    {name}_rtree    — R*Tree spatial index on position
    {name}_fts      — FTS5 full-text index (TextStream only)
    {name}_vec      — vec0 vector index (EmbeddingStream only)

Payloads use Codec (LCM for DimosMsg types, pickle otherwise).
Poses are decomposed into columns. Tags are JSON.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from reactivex.subject import Subject

from dimos.memory.codec import (
    JpegCodec,
    LcmCodec,
    PickleCodec,
    codec_for_type,
    module_path_to_type,
    type_to_module_path,
)
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


# ── Pose helpers (column-based) ──────────────────────────────────────


def _decompose_pose(pose: Any) -> tuple[float, float, float, float, float, float, float] | None:
    """Extract (x, y, z, qx, qy, qz, qw) from a PoseStamped or similar."""
    if pose is None:
        return None
    # PoseStamped has .pose.position and .pose.orientation
    p = pose.pose.position
    q = pose.pose.orientation
    return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def _reconstruct_pose(
    x: float | None,
    y: float | None,
    z: float | None,
    qx: float | None,
    qy: float | None,
    qz: float | None,
    qw: float | None,
) -> Any | None:
    """Rebuild a PoseStamped from column values."""
    if x is None:
        return None
    from dimos.msgs.geometry_msgs.Point import Point
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.std_msgs.Header import Header

    return PoseStamped(
        header=Header(),
        pose=Pose(
            position=Point(x=x, y=y or 0.0, z=z or 0.0),
            orientation=Quaternion(x=qx or 0.0, y=qy or 0.0, z=qz or 0.0, w=qw or 1.0),
        ),
    )


def _serialize_tags(tags: dict[str, Any] | None) -> str:
    if not tags:
        return "{}"
    return json.dumps(tags, separators=(",", ":"))


def _deserialize_tags(text: str) -> dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)  # type: ignore[no-any-return]


# ── SQL building ──────────────────────────────────────────────────────

# Columns selected from the meta table (no payload).
_META_COLS = "id, ts, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, tags, parent_id"


def _compile_filter(f: Filter, table: str) -> tuple[str, list[Any]]:
    """Compile a single filter to (SQL fragment, params)."""
    if isinstance(f, AfterFilter):
        return f"{table}.ts > ?", [f.t]
    if isinstance(f, BeforeFilter):
        return f"{table}.ts < ?", [f.t]
    if isinstance(f, TimeRangeFilter):
        return f"{table}.ts >= ? AND {table}.ts <= ?", [f.t1, f.t2]
    if isinstance(f, AtFilter):
        return f"ABS({table}.ts - ?) <= ?", [f.t, f.tolerance]
    if isinstance(f, TagsFilter):
        clauses: list[str] = []
        params: list[Any] = []
        for key, val in f.tags.items():
            clauses.append(f"json_extract({table}.tags, '$.{key}') = ?")
            params.append(val)
        return " AND ".join(clauses), params
    if isinstance(f, NearFilter):
        # Handled via R*Tree JOIN — see _compile_query
        return "1=1", []
    if isinstance(f, EmbeddingSearchFilter):
        return "1=1", []
    if isinstance(f, TextSearchFilter):
        return "1=1", []
    raise TypeError(f"Unknown filter type: {type(f)}")


def _has_near_filter(query: StreamQuery) -> NearFilter | None:
    for f in query.filters:
        if isinstance(f, NearFilter):
            return f
    return None


def _compile_query(query: StreamQuery, table: str) -> tuple[str, list[Any]]:
    """Compile a StreamQuery to (SQL, params) for a metadata SELECT."""
    where_parts: list[str] = []
    params: list[Any] = []
    joins: list[str] = []

    _has_near_filter(query)

    for f in query.filters:
        if isinstance(f, NearFilter):
            # R*Tree bounding-box join
            joins.append(f"JOIN {table}_rtree AS r ON r.id = {table}.id")
            where_parts.append(
                "r.min_x >= ? AND r.max_x <= ? AND "
                "r.min_y >= ? AND r.max_y <= ? AND "
                "r.min_z >= ? AND r.max_z <= ?"
            )
            pose_parts = _decompose_pose(f.pose)
            if pose_parts is not None:
                x, y, z = pose_parts[0], pose_parts[1], pose_parts[2]
            else:
                x, y, z = 0.0, 0.0, 0.0
            params.extend(
                [
                    x - f.radius,
                    x + f.radius,
                    y - f.radius,
                    y + f.radius,
                    z - f.radius,
                    z + f.radius,
                ]
            )
        else:
            sql_frag, p = _compile_filter(f, table)
            where_parts.append(sql_frag)
            params.extend(p)

    where = " AND ".join(where_parts) if where_parts else "1=1"
    join_clause = " ".join(joins)

    order = f"ORDER BY {query.order_field}"
    if query.order_field:
        if query.order_desc:
            order += " DESC"
    else:
        order = f"ORDER BY {table}.id"

    sql = f"SELECT {table}.{_META_COLS.replace(', ', f', {table}.')} FROM {table}"
    if join_clause:
        sql += f" {join_clause}"
    sql += f" WHERE {where} {order}"
    if query.limit_val is not None:
        sql += f" LIMIT {query.limit_val}"
    if query.offset_val is not None:
        sql += f" OFFSET {query.offset_val}"
    return sql, params


def _compile_count(query: StreamQuery, table: str) -> tuple[str, list[Any]]:
    where_parts: list[str] = []
    params: list[Any] = []
    joins: list[str] = []

    for f in query.filters:
        if isinstance(f, NearFilter):
            joins.append(f"JOIN {table}_rtree AS r ON r.id = {table}.id")
            pose_parts = _decompose_pose(f.pose)
            if pose_parts is not None:
                x, y, z = pose_parts[0], pose_parts[1], pose_parts[2]
            else:
                x, y, z = 0.0, 0.0, 0.0
            where_parts.append(
                "r.min_x >= ? AND r.max_x <= ? AND "
                "r.min_y >= ? AND r.max_y <= ? AND "
                "r.min_z >= ? AND r.max_z <= ?"
            )
            params.extend(
                [
                    x - f.radius,
                    x + f.radius,
                    y - f.radius,
                    y + f.radius,
                    z - f.radius,
                    z + f.radius,
                ]
            )
        else:
            sql_frag, p = _compile_filter(f, table)
            where_parts.append(sql_frag)
            params.extend(p)

    where = " AND ".join(where_parts) if where_parts else "1=1"
    join_clause = " ".join(joins)
    sql = f"SELECT COUNT(*) FROM {table}"
    if join_clause:
        sql += f" {join_clause}"
    sql += f" WHERE {where}"
    return sql, params


# ── Near-filter post-processing (exact distance after R*Tree bbox) ───


def _apply_near_post_filter(rows: list[Observation], near: NearFilter) -> list[Observation]:
    """Post-filter R*Tree candidates by exact Euclidean distance."""
    pose_parts = _decompose_pose(near.pose)
    if pose_parts is None:
        return []
    tx, ty, tz = pose_parts[0], pose_parts[1], pose_parts[2]
    result: list[Observation] = []
    for obs in rows:
        if obs.pose is None:
            continue
        op = obs.pose.pose.position
        dist = ((op.x - tx) ** 2 + (op.y - ty) ** 2 + (op.z - tz) ** 2) ** 0.5
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
        codec: LcmCodec | JpegCodec | PickleCodec | None = None,
    ) -> None:
        self._conn = conn
        self._table = table
        self._pose_provider = pose_provider
        self._codec = codec or PickleCodec()
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
        parent_id: int | None = None,
    ) -> Observation:
        if ts is None:
            ts = time.time()
        if pose is None and self._pose_provider is not None:
            pose = self._pose_provider()

        pose_cols = _decompose_pose(pose)
        tags_json = _serialize_tags(tags)

        # 1. Insert into meta table
        if pose_cols is not None:
            cur = self._conn.execute(
                f"INSERT INTO {self._table} "
                "(ts, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, tags, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, *pose_cols, tags_json, parent_id),
            )
        else:
            cur = self._conn.execute(
                f"INSERT INTO {self._table} (ts, tags, parent_id) VALUES (?, ?, ?)",
                (ts, tags_json, parent_id),
            )
        row_id = cur.lastrowid
        assert row_id is not None

        # 2. Insert into payload table
        payload_blob = self._codec.encode(payload)
        self._conn.execute(
            f"INSERT INTO {self._table}_payload (id, data) VALUES (?, ?)",
            (row_id, payload_blob),
        )

        # 3. Insert into R*Tree (if pose)
        if pose_cols is not None:
            x, y, z = pose_cols[0], pose_cols[1], pose_cols[2]
            self._conn.execute(
                f"INSERT INTO {self._table}_rtree (id, min_x, max_x, min_y, max_y, min_z, max_z) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row_id, x, x, y, y, z, z),
            )

        self._conn.commit()

        obs = Observation(
            id=row_id,
            ts=ts,
            pose=pose,
            tags=tags or {},
            parent_id=parent_id,
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
            observations = _apply_near_post_filter(observations, near)

        return observations

    def execute_count(self, query: StreamQuery) -> int:
        sql, params = _compile_count(query, self._table)
        result = self._conn.execute(sql, params).fetchone()
        return result[0] if result else 0  # type: ignore[no-any-return]

    def _row_to_obs(self, row: Any) -> Observation:
        row_id, ts, px, py, pz, qx, qy, qz, qw, tags_json, pid = row
        pose = _reconstruct_pose(px, py, pz, qx, qy, qz, qw)
        conn = self._conn
        table = self._table
        codec = self._codec

        def loader() -> Any:
            r = conn.execute(f"SELECT data FROM {table}_payload WHERE id = ?", (row_id,)).fetchone()
            if r is None:
                raise LookupError(f"No payload for id={row_id}")
            return codec.decode(r[0])

        return Observation(
            id=row_id,
            ts=ts,
            pose=pose,
            tags=_deserialize_tags(tags_json),
            parent_id=pid,
            _data_loader=loader,
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
        codec: LcmCodec | JpegCodec | PickleCodec | None = None,
    ) -> None:
        super().__init__(conn, table, pose_provider=pose_provider, codec=codec)
        self._vec_dimensions = vec_dimensions
        self._parent_table = parent_table

    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
        parent_id: int | None = None,
    ) -> Observation:
        from dimos.models.embedding.base import Embedding

        obs = super().do_append(payload, ts, pose, tags, parent_id)

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

        where_parts: list[str] = [f"{self._table}.id IN ({placeholders})"]
        params: list[Any] = list(rowids)

        for f in query.filters:
            if isinstance(f, EmbeddingSearchFilter):
                continue
            sql_frag, p = _compile_filter(f, self._table)
            where_parts.append(sql_frag)
            params.extend(p)

        where = " AND ".join(where_parts)
        sql = (
            f"SELECT {self._table}.{_META_COLS.replace(', ', f', {self._table}.')} "
            f"FROM {self._table} WHERE {where}"
        )
        rows = self._conn.execute(sql, params).fetchall()

        observations = [self._row_to_obs(r) for r in rows]

        near = _has_near_filter(query)
        if near is not None:
            observations = _apply_near_post_filter(observations, near)

        return observations

    def _row_to_obs(self, row: Any) -> Observation:
        row_id, ts, px, py, pz, qx, qy, qz, qw, tags_json, pid = row
        pose = _reconstruct_pose(px, py, pz, qx, qy, qz, qw)
        conn = self._conn
        table = self._table
        codec = self._codec
        parent_table = self._parent_table

        def loader() -> Any:
            r = conn.execute(f"SELECT data FROM {table}_payload WHERE id = ?", (row_id,)).fetchone()
            if r is None:
                raise LookupError(f"No payload for id={row_id}")
            return codec.decode(r[0])

        source_loader = None
        if pid is not None and parent_table is not None:

            def _source_loader(parent_tbl: str = parent_table, parent_row_id: int = pid) -> Any:
                r = conn.execute(
                    f"SELECT data FROM {parent_tbl}_payload WHERE id = ?", (parent_row_id,)
                ).fetchone()
                if r is None:
                    raise LookupError(f"No parent payload for id={parent_row_id}")
                # Resolve parent codec from _streams metadata
                meta = conn.execute(
                    "SELECT payload_module FROM _streams WHERE name = ?", (parent_tbl,)
                ).fetchone()
                if meta and meta[0]:
                    parent_type = module_path_to_type(meta[0])
                    parent_codec = codec_for_type(parent_type)
                else:
                    parent_codec = codec
                return parent_codec.decode(r[0])

            source_loader = _source_loader

        return EmbeddingObservation(
            id=row_id,
            ts=ts,
            pose=pose,
            tags=_deserialize_tags(tags_json),
            parent_id=pid,
            _data_loader=loader,
            _source_data_loader=source_loader,
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
        codec: LcmCodec | JpegCodec | PickleCodec | None = None,
    ) -> None:
        super().__init__(conn, table, pose_provider=pose_provider, codec=codec)
        self._tokenizer = tokenizer

    def do_append(
        self,
        payload: Any,
        ts: float | None,
        pose: Any | None,
        tags: dict[str, Any] | None,
        parent_id: int | None = None,
    ) -> Observation:
        obs = super().do_append(payload, ts, pose, tags, parent_id)

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

        where_parts: list[str] = [f"{self._table}.id IN ({placeholders})"]
        params: list[Any] = list(rowids)

        for f in query.filters:
            if isinstance(f, TextSearchFilter):
                continue
            sql_frag, p = _compile_filter(f, self._table)
            where_parts.append(sql_frag)
            params.extend(p)

        where = " AND ".join(where_parts)
        sql = (
            f"SELECT {self._table}.{_META_COLS.replace(', ', f', {self._table}.')} "
            f"FROM {self._table} WHERE {where}"
        )
        rows = self._conn.execute(sql, params).fetchall()

        observations = [self._row_to_obs(r) for r in rows]

        near = _has_near_filter(query)
        if near is not None:
            observations = _apply_near_post_filter(observations, near)

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
            "  payload_module TEXT,"
            "  stream_kind TEXT DEFAULT 'stream',"
            "  parent_stream TEXT,"
            "  embedding_dim INTEGER"
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

        if payload_type is None:
            payload_type = self._resolve_payload_type(name)

        if payload_type is None:
            raise TypeError(
                f"stream({name!r}): payload_type is required when creating a new stream. "
                "Pass the type explicitly, e.g. session.stream('images', Image)."
            )

        self._ensure_stream_tables(name)
        self._register_stream(name, payload_type, "stream")

        codec = codec_for_type(payload_type)
        backend = SqliteStreamBackend(self._conn, name, pose_provider=pose_provider, codec=codec)
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

        if payload_type is None:
            payload_type = self._resolve_payload_type(name)

        self._ensure_stream_tables(name)
        self._ensure_fts_table(name, tokenizer)
        self._register_stream(name, payload_type, "text")

        codec = codec_for_type(payload_type)
        backend = SqliteTextBackend(
            self._conn, name, tokenizer=tokenizer, pose_provider=pose_provider, codec=codec
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

        if payload_type is None:
            payload_type = self._resolve_payload_type(name)

        self._ensure_stream_tables(name)
        self._register_stream(name, payload_type, "embedding", embedding_dim=vec_dimensions)

        codec = codec_for_type(payload_type)
        backend = SqliteEmbeddingBackend(
            self._conn,
            name,
            vec_dimensions=vec_dimensions,
            pose_provider=pose_provider,
            parent_table=parent_table,
            codec=codec,
        )
        if vec_dimensions is not None:
            backend._ensure_vec_table()

        es: EmbeddingStream[Any] = EmbeddingStream(backend=backend, session=self)
        self._streams[name] = es
        return es

    def list_streams(self) -> list[StreamInfo]:
        rows = self._conn.execute("SELECT name, payload_module FROM _streams").fetchall()
        result: list[StreamInfo] = []
        for name, pmodule in rows:
            count_row = self._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
            count = count_row[0] if count_row else 0
            result.append(StreamInfo(name=name, payload_type=pmodule, count=count))
        return result

    def materialize_transform(
        self,
        name: str,
        source: Stream[Any],
        transformer: Transformer[Any, Any],
        *,
        payload_type: type | None = None,
        live: bool = False,
        backfill_only: bool = False,
    ) -> Stream[Any]:
        # Resolve source table name for parent lineage
        source_table = None
        if source._backend is not None:
            source_table = source._backend.stream_name

        target: Stream[Any]
        if isinstance(transformer, EmbeddingTransformer):
            target = self.embedding_stream(name, payload_type, parent_table=source_table)
        else:
            target = self.stream(name, payload_type)

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

    def _ensure_stream_tables(self, name: str) -> None:
        """Create the meta table, payload table, and R*Tree for a stream."""
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name} ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts REAL,"
            "  pose_x REAL,"
            "  pose_y REAL,"
            "  pose_z REAL,"
            "  pose_qx REAL,"
            "  pose_qy REAL,"
            "  pose_qz REAL,"
            "  pose_qw REAL,"
            "  tags TEXT DEFAULT '{}',"
            "  parent_id INTEGER"
            ")"
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_ts ON {name}(ts)")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name}_payload (  id INTEGER PRIMARY KEY,  data BLOB)"
        )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name}_rtree USING rtree("
            "  id,"
            "  min_x, max_x,"
            "  min_y, max_y,"
            "  min_z, max_z"
            ")"
        )
        self._conn.commit()

    def _ensure_fts_table(self, name: str, tokenizer: str) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name}_fts "
            f"USING fts5(content, tokenize='{tokenizer}')"
        )
        self._conn.commit()

    def _register_stream(
        self,
        name: str,
        payload_type: type | None,
        kind: str,
        *,
        embedding_dim: int | None = None,
    ) -> None:
        module_path = type_to_module_path(payload_type) if payload_type else None
        self._conn.execute(
            "INSERT OR IGNORE INTO _streams (name, payload_module, stream_kind, embedding_dim) "
            "VALUES (?, ?, ?, ?)",
            (name, module_path, kind, embedding_dim),
        )
        self._conn.commit()

    def _resolve_payload_type(self, name: str) -> type | None:
        """Look up payload type from _streams metadata (for restart case)."""
        row = self._conn.execute(
            "SELECT payload_module FROM _streams WHERE name = ?", (name,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return module_path_to_type(row[0])


# ── Store ─────────────────────────────────────────────────────────────


class SqliteStore(Store):
    """SQLite-backed memory store."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._load_extensions()

    def session(self) -> SqliteSession:
        return SqliteSession(self._conn)

    def _load_extensions(self) -> None:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except ImportError:
            pass

    def close(self) -> None:
        self._conn.close()
