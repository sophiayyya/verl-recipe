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
"""Exercise recipe composition through verl's native entry point."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

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
