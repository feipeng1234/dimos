# Copyright 2025-2026 Dimensional Inc.
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

"""Vendored fitted FOPDT plant models for the Go2 base.

Source artifacts (per-session, not in repo):
  ~/char_runs/session_20260425-143030/modeling/model_summary.json  (vx, rage)
  ~/char_runs/session_20260425-131525/modeling/model_summary.json  (wz, default)

Both produced by the characterization fitting pipeline at
:mod:`dimos.utils.characterization.modeling.session.fit_session`.

Caveats:
  - Only "rise" params are vendored. Real plant has rise/fall asymmetry,
    notably L_fall ~= 0.20 s vs L_rise ~= 0.04 s on vx. The current
    :class:`FOPDTChannel` is single-regime; asymmetric modeling is a
    follow-up if sim/hw rankings disagree.
  - vy is a placeholder copy of vx params: Go2 has no native lateral
    velocity, so any controller commanding vy on the real robot will
    behave very differently from the sim. Treat vy commands as a sim
    artifact, not a hardware-relevant signal.
"""

from __future__ import annotations

from dimos.utils.benchmarking.plant import FopdtChannelParams, Go2PlantParams

GO2_VX_RISE = FopdtChannelParams(K=1.008, tau=0.346, L=0.035)
GO2_VX_FALL = FopdtChannelParams(K=0.654, tau=0.259, L=0.202)
GO2_WZ_RISE = FopdtChannelParams(K=2.175, tau=0.258, L=0.085)
GO2_WZ_FALL = FopdtChannelParams(K=2.380, tau=0.258, L=0.083)

GO2_PLANT_FITTED = Go2PlantParams(
    vx=GO2_VX_RISE,
    vy=GO2_VX_RISE,  # placeholder - see module docstring
    wz=GO2_WZ_RISE,
)
