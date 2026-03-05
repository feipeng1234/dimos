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

"""Ingest 5min robot video → sharpness filter → CLIP embed → export top matches.

Caches the DB — re-run to just search without re-ingesting/embedding.
"""

from __future__ import annotations

from pathlib import Path

from dimos.memory.impl.sqlite import SqliteStore
from dimos.memory.ingest import ingest
from dimos.memory.transformer import EmbeddingTransformer, QualityWindowTransformer
from dimos.models.embedding.clip import CLIPModel
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.testing import TimedSensorReplay

OUT_DIR = Path(__file__).parent / "e2e_matches"
OUT_DIR.mkdir(exist_ok=True)

db_path = OUT_DIR / "e2e.db"
store = SqliteStore(str(db_path))
session = store.session()

# Check if we already have data
existing = {s.name for s in session.list_streams()}
need_build = "clip_embeddings" not in existing

if need_build:
    replay = TimedSensorReplay("unitree_go2_bigoffice/video")

    print("Loading CLIP...")
    clip = CLIPModel()
    clip.start()

    # 1. Ingest 5 minutes
    print("Ingesting 5 min of video...")
    raw = session.stream("raw_video", Image)
    n = ingest(raw, replay.iterate_ts(seek=5.0, duration=300.0))
    print(f"  {n} frames ingested")

    # 2. Sharpness filter
    print("Filtering by sharpness (0.5s windows)...")
    sharp = raw.transform(QualityWindowTransformer(lambda img: img.sharpness, window=0.5)).store(
        "sharp_frames", Image
    )
    n_sharp = sharp.count()
    print(f"  {n_sharp} sharp frames (from {n}, {n_sharp / n:.0%} kept)")

    # 3. Embed
    print("Embedding with CLIP...")
    embeddings = sharp.transform(EmbeddingTransformer(clip)).store("clip_embeddings")
    print(f"  {embeddings.count()} embeddings stored")
else:
    print(f"Using cached DB ({db_path})")
    clip = CLIPModel()
    clip.start()
    sharp = session.stream("sharp_frames")
    embeddings = session.embedding_stream("clip_embeddings")
    print(f"  {sharp.count()} sharp frames, {embeddings.count()} embeddings")

# 4. Search and export
queries = [
    "a hallway in an office",
    "a person standing",
    "a door",
    "a desk",
    "supermarket",
    "large room",
]

for query_text in queries:
    print(f"\nQuery: '{query_text}'")
    query_emb = clip.embed_text(query_text)
    results = embeddings.search_embedding(query_emb, k=5).fetch()

    slug = query_text.replace(" ", "_")[:30]
    for rank, result in enumerate(results):
        # .data auto-projects to parent image via parent_id lineage
        fname = OUT_DIR / f"{slug}_{rank + 1}_id{result.id}_ts{result.ts:.0f}.jpg"
        result.data.save(str(fname))
        print(f"  [{rank + 1}] id={result.id} ts={result.ts:.2f} → {fname.name}")

session.close()
store.close()
print(f"\nDone. Results in {OUT_DIR}/")
