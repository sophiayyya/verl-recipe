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
"""Dynamo training entry point.

Thin wrapper around ``verl.trainer.main_ppo`` that only swaps the hydra config
name. Unlike verl's own ``main()`` it also runs ``migrate_legacy_reward_impl``
before validation, preserving compatibility with older recipe configs.

The V0/V1 dispatch follows ``trainer.use_v1`` exactly like upstream:
V1 (default) runs TaskRunnerV1 → ``trainer.v1.trainer_mode``
(sync | colocate_async | separate_async); V0 remains reachable with
``trainer.use_v1=false`` until upstream removes it (deprecated, v0.9.0).
"""

import os

import hydra
from omegaconf import open_dict

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def _propagate_registry_to_ray_workers(config) -> None:
    """Ship VERL_USE_EXTERNAL_MODULES to every Ray worker via the job's runtime_env.

    Multi-node clusters are started with `ray start` BEFORE this driver runs, so
    Ray workers inherit their environment from the raylet's shell, not from this
    process — an export here never reaches them, and workers then fail with
    "Rollout dynamo with mode async not found" because the registry module was
    never imported. main_ppo merges config.ray_kwargs.ray_init.runtime_env into
    ray.init(), which Ray forwards to all workers regardless of how the raylet
    was launched. The schema's runtime_env node is a struct without an env_vars
    field (see verl main_ppo's own ConfigKeyError note), hence open_dict.

    setdefault semantics: an explicit user-provided value (config or the
    ++ray_kwargs CLI override documented in the README) wins.
    """
    modules = os.environ.get("VERL_USE_EXTERNAL_MODULES", "recipe.dynamo.register")
    try:
        runtime_env = config.ray_kwargs.ray_init.runtime_env
    except (AttributeError, KeyError):
        # Trimmed configs (unit-test doubles, minimal launchers) may not carry
        # the ray_kwargs schema; single-node runs don't need the injection.
        return
    with open_dict(runtime_env):
        if runtime_env.get("env_vars") is None:
            runtime_env.env_vars = {}
        runtime_env.env_vars.setdefault("VERL_USE_EXTERNAL_MODULES", modules)


@hydra.main(config_path="config", config_name="dynamo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    _propagate_registry_to_ray_workers(config)
    config = migrate_legacy_reward_impl(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    manager_fqn = (config.actor_rollout_ref.rollout.get("agent", {}) or {}).get("agent_loop_manager_class") or ""
    if config.trainer.use_v1 and "DynamoAgentLoopManager" in str(manager_fqn):
        raise ValueError(
            "trainer.use_v1=true with agent_loop_manager_class=DynamoAgentLoopManager: the legacy "
            "manager does not write TransferQueue and violates the V1 contract. Use "
            "--config-name=dynamo_trainer_v1_colocate (agent_loop_manager_class=null) for V1, or "
            "set trainer.use_v1=false for the legacy path."
        )
    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
