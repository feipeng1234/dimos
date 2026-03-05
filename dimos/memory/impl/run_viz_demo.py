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

"""Visual demo: similarity heatmap + timeline in Rerun.

Run with:  python -m dimos.memory.impl.run_viz_demo
Then open Rerun viewer to see the output.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

from dimos.memory.impl.sqlite import SqliteStore
from dimos.memory.types import EmbeddingObservation
from dimos.memory.viz import log_similarity_timeline, similarity_heatmap
from dimos.models.embedding.base import Embedding
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

# ── Rerun setup ───────────────────────────────────────────────────────
rr.init("memory_viz_demo", spawn=True)

# ── Build a small DB with posed embeddings ────────────────────────────
store = SqliteStore(":memory:")
session = store.session()

es = session.embedding_stream("demo_emb", vec_dimensions=4)

# Simulate a robot path with embeddings at various positions
np.random.seed(42)
n_obs = 60
for i in range(n_obs):
    angle = 2 * np.pi * i / n_obs
    radius = 3.0 + 0.5 * np.sin(3 * angle)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    # Embedding: mix of two basis vectors depending on position
    mix = (np.sin(angle) + 1) / 2  # 0..1
    vec = np.array([mix, 1.0 - mix, 0.1 * np.cos(angle), 0.0], dtype=np.float32)
    vec /= np.linalg.norm(vec)

    pose = PoseStamped(
        ts=float(i),
        frame_id="world",
        position=[x, y, 0.0],
        orientation=[0.0, 0.0, 0.0, 1.0],
    )
    es.append(Embedding(vec), ts=float(i), pose=pose)

print(f"Created {es.count()} observations on a circular path")

# ── Search and visualize ──────────────────────────────────────────────
query = [1.0, 0.0, 0.0, 0.0]
results = es.search_embedding(query, k=n_obs).fetch()

print(f"Search returned {len(results)} results")
for obs in results[:5]:
    assert isinstance(obs, EmbeddingObservation)
    print(f"  id={obs.id} ts={obs.ts:.0f} similarity={obs.similarity:.3f}")

# 1. Similarity heatmap → OccupancyGrid → Rerun mesh
grid = similarity_heatmap(results, resolution=0.2, padding=2.0)
print(f"\nHeatmap: {grid}")
rr.log("world/heatmap", grid.to_rerun(colormap="inferno"))

# 2. Similarity timeline → Rerun scalar plot
log_similarity_timeline(results, entity_path="plots/similarity")
print("Logged similarity timeline")

# 3. Also log poses as arrows for spatial context
for obs in results:
    if obs.pose is not None and obs.ts is not None:
        rr.set_time("memory_time", timestamp=obs.ts)
        rr.log("world/poses", obs.pose.to_rerun_arrow(length=0.3))

print("\nDone — check Rerun viewer")

session.close()
store.close()
