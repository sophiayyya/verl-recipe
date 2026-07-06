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
"""Dynamo training entry point."""

import hydra

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

try:
    # Current verl keeps the legacy PPO runner in a dedicated module.
    from verl.trainer.main_ppo_v0 import TaskRunner
except ModuleNotFoundError as exc:
    if exc.name != "verl.trainer.main_ppo_v0":
        raise
    # The ThunderAgent-pinned verl revision still selects its TaskRunner when
    # run_ppo is called without an explicit runner.
    TaskRunner = None


@hydra.main(config_path="config", config_name="dynamo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    if TaskRunner is None:
        run_ppo(config)
    else:
        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
