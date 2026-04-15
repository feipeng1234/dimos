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

"""Tests for PCTPlanner NativeModule wrapper."""

from pathlib import Path
from typing import get_origin, get_type_hints

from dimos.core.stream import In, Out
from dimos.navigation.smart_nav.modules.pct_planner.pct_planner import (
    PCTPlanner,
    PCTPlannerConfig,
)


class TestPCTPlannerConfig:
    """Test PCTPlanner configuration."""

    def test_default_config(self):
        config = PCTPlannerConfig()
        assert config.resolution == 0.075
        assert config.slice_dh == 0.4
        assert config.slope_max == 0.45
        assert config.step_max == 0.5
        assert config.lookahead_distance == 1.25
        assert config.cost_barrier == 100.0
        assert config.kernel_size == 11

    def test_cli_args_generation(self):
        config = PCTPlannerConfig(
            resolution=0.1,
            slice_dh=0.5,
            lookahead_distance=2.0,
        )
        args = config.to_cli_args()
        assert "--resolution" in args
        assert "0.1" in args
        assert "--slice_dh" in args
        assert "0.5" in args
        assert "--lookahead_distance" in args
        assert "2.0" in args

    def test_every_config_field_has_cli_arg(self):
        """Every declared PCT config field must appear as a --flag on the CLI.

        Iterates model_fields so adding a new field to the config
        automatically tightens this test.
        """
        config = PCTPlannerConfig()
        args = config.to_cli_args()
        parent = PCTPlannerConfig.__mro__[1]
        parent_fields = set(getattr(parent, "model_fields", {}).keys())
        for name in PCTPlannerConfig.model_fields:
            if name in parent_fields:
                continue
            assert f"--{name}" in args, f"Missing CLI arg for config field: {name}"


class TestPCTPlannerModule:
    """Test PCTPlanner module declaration."""

    def test_ports_declared(self):
        hints = get_type_hints(PCTPlanner)
        in_ports = {k for k, v in hints.items() if get_origin(v) is In}
        out_ports = {k for k, v in hints.items() if get_origin(v) is Out}

        assert "explored_areas" in in_ports
        assert "odometry" in in_ports
        assert "goal" in in_ports
        assert "way_point" in out_ports
        assert "goal_path" in out_ports
        assert "tomogram" in out_ports

    def test_executable_path_is_relative_to_module(self):
        """Config cwd/executable resolves against the module's own directory.

        Does not require the binary to exist — pure-metadata check that
        runs on CI without any nix build.
        """
        config = PCTPlannerConfig()
        assert config.cwd is not None
        cwd = Path(config.cwd)
        assert cwd.is_dir(), f"Module cwd must exist: {cwd}"
        assert cwd.name == "pct_planner"
        # executable stays as a relative path in config until the native
        # module resolves it at build/start time.
        assert not Path(config.executable).is_absolute()
        assert config.executable.endswith("pct_planner")
