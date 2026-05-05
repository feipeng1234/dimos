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

"""Dataset-build blueprints.

Wraps `DataPrepModule` so users can run::

    dimos run learning-dataprep
    dimos run learning-dataprep -o dataprepmodule.source=data/recordings/foo.db \\
                                -o dataprepmodule.output.path=data/datasets/foo

The defaults below target the included pickplace_001 demo. For single-demo
recordings without an `episode_status` stream, `learning_dataprep_whole_session`
treats the entire recording as one episode.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.learning.dataprep import (
    EpisodeExtractor,
    OutputConfig,
    StreamField,
    SyncConfig,
)
from dimos.learning.dataprep_module import DataPrepModule

learning_dataprep = autoconnect(
    DataPrepModule.blueprint(
        source="data/recordings/pickplace_001.db",
        episodes=EpisodeExtractor(
            extractor="ranges",
            ranges=[(1777931622.11, 1777931646.61)],
        ),
        observation={
            "image":       StreamField(stream="color_image", field="data"),
            "joint_state": StreamField(stream="joint_state", field="position"),
        },
        action={
            "joint_target": StreamField(stream="joint_state", field="position"),
        },
        sync=SyncConfig(anchor="image", rate_hz=14.0, tolerance_ms=80.0),
        output=OutputConfig(
            format="lerobot",
            path="data/datasets/pickplace_001",
            metadata={"fps": 14, "robot": "xarm7", "default_task_label": "pick_and_place"},
        ),
        auto_run=True,
    ),
).transports({})


learning_dataprep_whole_session = autoconnect(
    DataPrepModule.blueprint(
        source="data/session.db",
        episodes=EpisodeExtractor(extractor="whole_session"),
        observation={
            "image":       StreamField(stream="camera_color_image", field="data"),
            "joint_state": StreamField(stream="coordinator_joint_state", field="position"),
        },
        action={
            "joint_target": StreamField(stream="coordinator_joint_command", field="position"),
        },
        sync=SyncConfig(anchor="image", rate_hz=30.0, tolerance_ms=50.0),
        output=OutputConfig(
            format="lerobot",
            path="data/datasets/default",
            metadata={"fps": 30, "robot": "xarm7"},
        ),
        auto_run=True,
    ),
).transports({})


__all__ = ["learning_dataprep", "learning_dataprep_whole_session"]
