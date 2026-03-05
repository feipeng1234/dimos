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

"""Tests for SQLite-backed memory store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dimos.memory.impl.sqlite import SqliteSession, SqliteStore

if TYPE_CHECKING:
    from dimos.memory.types import Observation


@pytest.fixture
def store(tmp_path: object) -> SqliteStore:
    # tmp_path is a pathlib.Path
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    return SqliteStore(str(tmp_path / "test.db"))


@pytest.fixture
def session(store: SqliteStore) -> SqliteSession:
    return store.session()


class TestStreamBasics:
    def test_create_stream(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        assert s is not None

    def test_append_and_fetch(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        obs = s.append(b"frame1")
        assert obs.id == 1
        assert obs.data == b"frame1"
        assert obs.ts is not None

        rows = s.fetch()
        assert len(rows) == 1
        assert rows[0].data == b"frame1"
        assert rows[0].id == 1

    def test_append_multiple(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        s.append(b"frame1")
        s.append(b"frame2")
        s.append(b"frame3")

        assert s.count() == 3
        rows = s.fetch()
        assert [r.data for r in rows] == [b"frame1", b"frame2", b"frame3"]

    def test_append_with_tags(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        s.append(b"frame1", tags={"cam": "front", "quality": "high"})

        rows = s.fetch()
        assert rows[0].tags == {"cam": "front", "quality": "high"}

    def test_last(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        s.append(b"frame1", ts=1.0)
        s.append(b"frame2", ts=2.0)
        s.append(b"frame3", ts=3.0)

        obs = s.last()
        assert obs.data == b"frame3"
        assert obs.ts == 3.0

    def test_one(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        s.append(b"only")

        obs = s.one()
        assert obs.data == b"only"

    def test_one_empty_raises(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        with pytest.raises(LookupError):
            s.one()


class TestFilters:
    def test_after(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("old", ts=1.0)
        s.append("new", ts=10.0)

        rows = s.after(5.0).fetch()
        assert len(rows) == 1
        assert rows[0].data == "new"

    def test_before(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("old", ts=1.0)
        s.append("new", ts=10.0)

        rows = s.before(5.0).fetch()
        assert len(rows) == 1
        assert rows[0].data == "old"

    def test_time_range(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("a", ts=1.0)
        s.append("b", ts=5.0)
        s.append("c", ts=10.0)

        rows = s.time_range(3.0, 7.0).fetch()
        assert len(rows) == 1
        assert rows[0].data == "b"

    def test_at(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("a", ts=1.0)
        s.append("b", ts=5.0)
        s.append("c", ts=10.0)

        rows = s.at(5.5, tolerance=1.0).fetch()
        assert len(rows) == 1
        assert rows[0].data == "b"

    def test_filter_tags(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("front", tags={"cam": "front"})
        s.append("rear", tags={"cam": "rear"})

        rows = s.filter_tags(cam="front").fetch()
        assert len(rows) == 1
        assert rows[0].data == "front"

    def test_chained_filters(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("a", ts=1.0, tags={"cam": "front"})
        s.append("b", ts=5.0, tags={"cam": "front"})
        s.append("c", ts=5.0, tags={"cam": "rear"})

        rows = s.after(3.0).filter_tags(cam="front").fetch()
        assert len(rows) == 1
        assert rows[0].data == "b"


class TestOrdering:
    def test_order_by_ts(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("b", ts=2.0)
        s.append("a", ts=1.0)
        s.append("c", ts=3.0)

        rows = s.order_by("ts").fetch()
        assert [r.data for r in rows] == ["a", "b", "c"]

    def test_order_by_desc(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("a", ts=1.0)
        s.append("b", ts=2.0)
        s.append("c", ts=3.0)

        rows = s.order_by("ts", desc=True).fetch()
        assert [r.data for r in rows] == ["c", "b", "a"]

    def test_limit_offset(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        for i in range(10):
            s.append(f"item{i}", ts=float(i))

        rows = s.order_by("ts").limit(3).offset(2).fetch()
        assert [r.data for r in rows] == ["item2", "item3", "item4"]


class TestFetchPages:
    def test_basic_pagination(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        for i in range(10):
            s.append(f"item{i}", ts=float(i))

        pages = list(s.fetch_pages(batch_size=3))
        assert len(pages) == 4  # 3+3+3+1
        assert len(pages[0]) == 3
        assert len(pages[-1]) == 1

        all_items = [obs.data for page in pages for obs in page]
        assert all_items == [f"item{i}" for i in range(10)]


class TestTextStream:
    def test_create_and_append(self, session: SqliteSession) -> None:
        s = session.text_stream("logs", str)
        s.append("Motor fault on joint 3")
        s.append("Battery low warning")

        assert s.count() == 2

    def test_text_search(self, session: SqliteSession) -> None:
        s = session.text_stream("logs", str)
        s.append("Motor fault on joint 3")
        s.append("Battery low warning")
        s.append("Motor overheating on joint 5")

        rows = s.search_text("motor", k=10).fetch()
        assert len(rows) == 2
        assert all("Motor" in r.data for r in rows)


class TestListStreams:
    def test_list_empty(self, session: SqliteSession) -> None:
        assert session.list_streams() == []

    def test_list_after_create(self, session: SqliteSession) -> None:
        session.stream("images", bytes)
        session.text_stream("logs", str)

        infos = session.list_streams()
        names = {i.name for i in infos}
        assert names == {"images", "logs"}


class TestReactive:
    def test_appended_observable(self, session: SqliteSession) -> None:
        s = session.stream("images", bytes)
        received: list[Observation] = []
        s.appended.subscribe(on_next=received.append)

        s.append(b"frame1")
        s.append(b"frame2")

        assert len(received) == 2
        assert received[0].data == b"frame1"
        assert received[1].data == b"frame2"


class TestTransformInMemory:
    def test_lambda_transform(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("hello", ts=1.0)
        s.append("world", ts=2.0)

        upper = s.transform(lambda x: x.upper())
        results = upper.fetch()
        assert len(results) == 2
        assert results[0].data == "HELLO"
        assert results[1].data == "WORLD"

    def test_lambda_filter_none(self, session: SqliteSession) -> None:
        s = session.stream("data", int)
        s.append(1, ts=1.0)
        s.append(2, ts=2.0)
        s.append(3, ts=3.0)

        evens = s.transform(lambda x: x * 2 if x % 2 == 0 else None)
        results = evens.fetch()
        assert len(results) == 1
        assert results[0].data == 4

    def test_lambda_expand_list(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("a,b,c", ts=1.0)

        split = s.transform(lambda x: x.split(","))
        results = split.fetch()
        assert len(results) == 3
        assert [r.data for r in results] == ["a", "b", "c"]


class TestTransformStore:
    def test_transform_store_backfill(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("hello", ts=1.0)
        s.append("world", ts=2.0)

        stored = s.transform(lambda x: x.upper()).store("upper_data")
        rows = stored.fetch()
        assert len(rows) == 2
        assert rows[0].data == "HELLO"
        assert rows[1].data == "WORLD"

        # Also queryable by name
        reloaded = session.stream("upper_data")
        assert reloaded.count() == 2

    def test_transform_store_live(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("existing", ts=1.0)

        # live=True skips backfill, only processes new items
        stored = s.transform(lambda x: x.upper(), live=True).store("live_upper")
        assert stored.count() == 0  # no backfill

        s.append("new", ts=2.0)
        assert stored.count() == 1
        assert stored.last().data == "NEW"

    def test_transform_store_backfill_only(self, session: SqliteSession) -> None:
        s = session.stream("data", str)
        s.append("existing", ts=1.0)

        stored = s.transform(lambda x: x.upper(), backfill_only=True).store("backfill_upper")
        assert stored.count() == 1
        assert stored.one().data == "EXISTING"

        # New appends should NOT propagate
        s.append("new", ts=2.0)
        assert stored.count() == 1  # still 1


class TestStoreReopen:
    def test_data_persists(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        db_path = str(tmp_path / "persist.db")

        # Write
        store1 = SqliteStore(db_path)
        s1 = store1.session()
        s1.stream("data", str).append("hello", ts=1.0)
        s1.close()
        store1.close()

        # Re-open and read
        store2 = SqliteStore(db_path)
        s2 = store2.session()
        rows = s2.stream("data", str).fetch()
        assert len(rows) == 1
        assert rows[0].data == "hello"
        s2.close()
        store2.close()
