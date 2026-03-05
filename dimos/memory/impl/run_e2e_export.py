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

"""Ingest 5min robot video → sharpness filter → CLIP embed → search & visualize.

Caches the DB — re-run to just search without re-ingesting/embedding.
Outputs heatmaps and timelines to Rerun, images to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import rerun as rr

from dimos.memory.impl.sqlite import SqliteStore
from dimos.memory.ingest import ingest
from dimos.memory.transformer import (
    CaptionTransformer,
    EmbeddingTransformer,
    QualityWindowTransformer,
)
from dimos.memory.viz import log_similarity_timeline, similarity_heatmap
from dimos.models.embedding.clip import CLIPModel
from dimos.models.vl.florence import Florence2Model
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.testing import TimedSensorReplay

if TYPE_CHECKING:
    from dimos.memory.stream import EmbeddingStream

OUT_DIR = Path(__file__).parent / "e2e_matches"
OUT_DIR.mkdir(exist_ok=True)

rr.init("memory_e2e", spawn=True)

db_path = OUT_DIR / "e2e.db"
store = SqliteStore(str(db_path))
session = store.session()

# Check if we already have data
existing = {s.name for s in session.list_streams()}
need_build = "clip_embeddings" not in existing

if need_build:
    replay = TimedSensorReplay("unitree_go2_bigoffice/video")
    odom = TimedSensorReplay("unitree_go2_bigoffice/odom")

    print("Loading CLIP...")
    clip = CLIPModel()
    clip.start()

    # 1. Ingest 5 minutes with odom poses
    print("Ingesting 5 min of video with odom poses...")
    raw = session.stream("raw_video", Image)
    n = ingest(raw, replay.iterate_ts(seek=5.0, duration=300.0), pose_source=odom)
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
    embeddings: EmbeddingStream[Any] = sharp.transform(EmbeddingTransformer(clip)).store(
        "clip_embeddings"
    )  # type: ignore[assignment]
    print(f"  {embeddings.count()} embeddings stored")
else:
    print(f"Using cached DB ({db_path})")
    print("loading Clip")
    clip = CLIPModel()
    clip.start()
    print("done")
    sharp = session.stream("sharp_frames")
    embeddings = session.embedding_stream("clip_embeddings", embedding_model=clip)
    print(f"  {sharp.count()} sharp frames, {embeddings.count()} embeddings")

# 4. Search, visualize, export
queries = [
    "a hallway in an office",
    "a person standing",
    "a door",
    "a desk",
    "supermarket",
    "large room",
]

print("\nLoading Florence2 for captioning...")
captioner = Florence2Model()
captioner.start()

caption_xf = CaptionTransformer(captioner)

for query_text in queries:
    print(f"\nQuery: '{query_text}'")
    slug = query_text.replace(" ", "_")[:30]

    # raw=True: get EmbeddingObservation with .similarity and .pose
    raw_results = embeddings.search_embedding(query_text, k=50, raw=True).fetch()

    # Spatial heatmap → Rerun
    grid = similarity_heatmap(raw_results, resolution=0.5)
    print(f"  Heatmap: {grid}")
    rr.log(f"world/{slug}/heatmap", grid.to_rerun(colormap="inferno"))

    # Temporal timeline → Rerun
    log_similarity_timeline(raw_results, entity_path=f"plots/{slug}")

    # Caption top 5 (auto-projected results for image access)
    results = embeddings.search_embedding(query_text, k=5).fetch()
    captions = results.transform(caption_xf).fetch()

    for rank, (cap, img) in enumerate(zip(captions, results, strict=False)):
        fname = OUT_DIR / f"{slug}_{rank + 1}_id{img.id}_ts{img.ts:.0f}.jpg"
        img.data.save(str(fname))
        print(f"  [{rank + 1}] id={img.id} ts={img.ts:.2f} — {cap.data}")

session.close()
store.close()
print(f"\nDone. Results in {OUT_DIR}/")
