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

"""Per-feature dataset statistics — written once, read by trainers + inference.

LeRobot expects `meta/stats.json` next to the dataset with mean/std/min/max/q01/q99
for every observation and action key. The same dict is consumed by:
  - the trainer (normalize inputs / unnormalize predicted actions)
  - the inference `ObsBuilder` and `ActionReplayer` (same normalization, live)

`Stats` is a streaming Welford accumulator: feed it `Sample` instances one at a
time and call `.result()` at the end. Used both inside `formats.lerobot.write`
(so the dataset is self-describing on first build) and as a standalone pass via
`compute_stats(dp)` if a dataset on disk needs stats recomputed.

Image stats are computed on a subsample (every Nth frame) to bound cost.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dimos.learning.spec import Sample

if TYPE_CHECKING:
    from dimos.learning.dataprep import DataPrep


class Stats:
    """Streaming Welford accumulator for per-feature mean/std/min/max/q01/q99.

    Update with one Sample at a time; call `.result()` at the end to get the
    serializable dict written to `meta/stats.json`.

    Quantiles (q01, q99) are computed from a reservoir sample of size
    `quantile_reservoir` per feature — bounded memory for unbounded streams.
    """

    def __init__(
        self,
        image_subsample: int = 10,
        quantile_reservoir: int = 10_000,
        seed: int = 0,
    ) -> None:
        """Configure cost knobs.

        Args:
            image_subsample: include every Nth image frame in stats; N=1 for full
                accuracy, larger N for faster builds on long sessions.
            quantile_reservoir: reservoir size per feature for q01/q99.
            seed: for the reservoir sampler.
        """
        raise NotImplementedError

    def update(self, sample: Sample) -> None:
        """Fold one Sample into the running statistics for every obs/action key."""
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        """Return the LeRobot-compatible stats dict.

        Schema:
            {
                "observation.<key>": {"mean": [...], "std": [...], "min": [...],
                                       "max": [...], "q01": [...], "q99": [...]},
                "action.<key>":      {... same keys ...},
                ...
            }
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write `result()` to `path` as JSON."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Read a stats JSON from disk. Returns the raw dict, not a Stats instance."""
        raise NotImplementedError


def compute_stats(samples: Iterator[Sample], **kw: Any) -> dict[str, Any]:
    """One-shot helper: drain `samples`, return the stats dict.

    Equivalent to `s = Stats(**kw); for x in samples: s.update(x); return s.result()`.
    """
    raise NotImplementedError


def compute_stats_from_prep(dp: DataPrep, **kw: Any) -> dict[str, Any]:
    """One-shot helper that pulls samples from a DataPrep instance.

    Convenience for "I have a built dataset on disk and need to recompute stats."
    """
    raise NotImplementedError
