# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Exercise recipe composition and the shell contract for verl's native entry point."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra._internal.utils import get_args_parser

from verl.trainer import main_ppo

RECIPE_ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    ("dynamo_trainer", False, None),
    ("dynamo_trainer_v1_colocate", True, "colocate_async"),
    ("dynamo_trainer_v1_colocate_sglang", True, "colocate_async"),
    ("dynamo_trainer_v1_separate", True, "separate_async"),
]


@pytest.mark.parametrize("name,use_v1,mode", PRESETS)
@pytest.mark.parametrize("registry_source", ["default", "override"])
def test_presets_compose_with_worker_registration(name, use_v1, mode, registry_source, monkeypatch, tmp_path):
    # Work outside the verl checkout: imports must use Hydra's package searchpath.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERL_USE_EXTERNAL_MODULES", raising=False)
    expected = "recipe.dynamo.register"
    overrides = []
    if registry_source == "override":
        expected = "recipe.dynamo.register,other.module"
        overrides.append("++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='" + expected + "'")
    with initialize_config_dir(config_dir=str(RECIPE_ROOT / "config"), version_base=None):
        config = compose(config_name=name, overrides=overrides)
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES == expected

    # Exercise the native runner through its actual Ray handoff. Merely reading
    # an OmegaConf property resolves it, but to_container defaults to resolve=False.
    # An oc.env interpolation would reach Ray as literal ${...} and break imports.
    class ReachedRayInit(Exception):
        pass

    received = {}

    def capture_ray_init(**kwargs):
        received.update(kwargs)
        raise ReachedRayInit

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_ppo.ray, "init", capture_ray_init)
    monkeypatch.setattr(main_ppo, "get_ppo_ray_runtime_env", lambda config: {"env_vars": {"EXISTING": "kept"}})
    with pytest.raises(ReachedRayInit):
        main_ppo.run_ppo(config, task_runner_class=None)
    worker_env = received["runtime_env"]["env_vars"]
    assert worker_env["VERL_USE_EXTERNAL_MODULES"] == expected
    assert worker_env["EXISTING"] == "kept"
    if use_v1:
        assert worker_env["TRANSFER_QUEUE_ENABLE"] == "1"
    assert config.trainer.use_v1 is use_v1
    assert config.actor_rollout_ref.rollout.name == "dynamo"
    if use_v1:
        assert config.trainer.v1.trainer_mode == mode
        assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class is None
        assert config.transfer_queue.enable
    else:
        assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class.endswith("DynamoAgentLoopManager")


@pytest.mark.parametrize("script", sorted(RECIPE_ROOT.glob("smoke_dynamo_v1_*.sh")), ids=lambda p: p.stem)
@pytest.mark.parametrize("child_exit", [0, 7])
def test_v1_launchers_use_native_entrypoint_and_forward_arguments(script, child_exit, tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    # Replace only the child process. The real Bash launcher supplies all arguments.
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
        "with open(os.environ['DYNAMO_LAUNCH_CAPTURE'], 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], 'registry': os.environ['VERL_USE_EXTERNAL_MODULES']}, f)\n"
        "sys.exit(int(os.environ['DYNAMO_CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "VERL_USE_EXTERNAL_MODULES": "recipe.dynamo.register,other.module",
        "DYNAMO_LAUNCH_CAPTURE": str(capture),
        "DYNAMO_CHILD_EXIT": str(child_exit),
        "MODEL_PATH": "/models/model with spaces",
    }
    result = subprocess.run(
        ["bash", str(script), "trainer.total_training_steps=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == child_exit, result.stderr
    launched = json.loads(capture.read_text())
    argv = launched["argv"]
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    # Hydra cannot resume its positional override list after config options.
    parsed = get_args_parser().parse_args(argv[2:])
    assert parsed.config_name.startswith("dynamo_trainer_v1_")
    assert Path(argv[argv.index("--config-path") + 1]) == RECIPE_ROOT / "config"
    assert "actor_rollout_ref.model.path=/models/model with spaces" in argv
    assert argv[-1] == "trainer.total_training_steps=3"
    assert launched["registry"] == env["VERL_USE_EXTERNAL_MODULES"]
    assert (
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='" + env["VERL_USE_EXTERNAL_MODULES"] + "'"
    ) in argv
    if child_exit:
        assert "PASS:" not in result.stdout
