"""Exercise the customer launcher without starting Ray or allocating GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra._internal.utils import get_args_parser
from hydra.core.override_parser.overrides_parser import OverridesParser

RECIPE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("child_exit", [0, 7])
def test_customer_launcher_forwards_workload_and_child_status(multi_turn, child_exit, tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_CAPTURE'], 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], 'registry': os.environ['VERL_USE_EXTERNAL_MODULES'],\n"
        "               'pythonpath': os.environ['PYTHONPATH'], 'ray_address': os.getenv('RAY_ADDRESS'),\n"
        "               'rocr_devices': os.getenv('ROCR_VISIBLE_DEVICES')}, f)\n"
        "sys.exit(int(os.environ['CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "LAUNCH_CAPTURE": str(capture),
        "CHILD_EXIT": str(child_exit),
        "VERL_SRC": str(tmp_path),
        "DYNAMO_SRC": "/source/dynamo",
        "MODEL_PATH": "/models/model with spaces",
        "TRAIN_FILE": "/data/train.parquet",
        "VERL_USE_EXTERNAL_MODULES": "custom.reward",
        "AUTO_FINALIZE": "false" if multi_turn else "true",
        "AGENT_CONFIG": "/data/agent config.yaml" if multi_turn else "",
        "UNIAGENT_ROOT": "/source/uni-agent",
        "RAY_ADDRESS": "ray://unrelated-cluster:10001",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
    result = subprocess.run(
        ["bash", str(RECIPE_ROOT / "run_thunderagent_vllm024.sh"), "trainer.total_training_steps=3"],
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
    parsed = get_args_parser().parse_args(argv[2:])
    assert parsed.config_name == "dynamo_trainer_v1_colocate"
    assert Path(parsed.config_path) == RECIPE_ROOT / "config"
    assert argv[-1] == "trainer.total_training_steps=3"
    assert "trainer.use_v1=true" in argv
    assert "trainer.v1.trainer_mode=colocate_async" in argv
    assert "actor_rollout_ref.model.path='/models/model with spaces'" in argv
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=true" in argv
    assert launched["registry"] == "recipe.dynamo.register,custom.reward"
    assert "/source/dynamo/components/src" in launched["pythonpath"]
    assert "/source/uni-agent" in launched["pythonpath"]
    assert launched["ray_address"] is None
    assert launched["rocr_devices"] is None
    if multi_turn:
        assert "actor_rollout_ref.rollout.agent.agent_loop_config_path='/data/agent config.yaml'" in argv
        assert "actor_rollout_ref.rollout.multi_turn.enable=true" in argv
        assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.auto_finalize=false" in argv
    else:
        assert "actor_rollout_ref.rollout.multi_turn.enable=false" in argv
        assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.auto_finalize=true" in argv


@pytest.mark.parametrize("custom_runner", [False, True])
@pytest.mark.parametrize("child_exit", [0, 7])
def test_current_uniagent_launcher_selects_gateway_and_upstream_fsdp_config(custom_runner, child_exit, tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "sys.exit(int(os.environ['CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "LAUNCH_CAPTURE": str(capture),
        "CHILD_EXIT": str(child_exit),
        "VERL_SRC": str(tmp_path),
        "DYNAMO_SRC": "/source/dynamo",
        "MODEL_PATH": "/models/qwen",
        "TRAIN_FILE": "/data/train.parquet",
        "UNIAGENT_ROOT": "/source/uni-agent",
        "TASK_CONFIG": "/data/task config.yaml",
        "AGENT_CONFIG": "/data/obsolete-agent.yaml",
        "AUTO_FINALIZE": "true",
        "AGENT_RUNNER_FQN": "custom.run" if custom_runner else "uni_agent.framework.task_runner.run_task",
    }
    result = subprocess.run(
        ["bash", str(RECIPE_ROOT / "run_uniagent_thunderagent_vllm024.sh"), "trainer.total_training_steps=3"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == child_exit, result.stderr
    argv = json.loads(capture.read_text())
    get_args_parser().parse_args(argv[2:])
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    assert argv[-1] == "trainer.total_training_steps=3"
    assert not any("defer_fsdp_grad_sync" in arg for arg in argv)
    assert "actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false" in argv
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.auto_finalize=false" in argv
    assert (
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class="
        "recipe.dynamo.uniagent_gateway.DynamoAgentFrameworkRolloutAdapter" in argv
    )
    assert not any("obsolete-agent" in arg for arg in argv)
    assert any("runner_kwargs.task_config_path=" in arg for arg in argv) != custom_runner


@pytest.mark.parametrize("child_exit", [0, 7])
def test_fully_async_launcher_overrides_nested_colocate_defaults(child_exit, tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "sys.exit(int(os.environ['CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "LAUNCH_CAPTURE": str(capture),
        "CHILD_EXIT": str(child_exit),
        "VERL_SRC": str(tmp_path),
        "DYNAMO_SRC": "/source/dynamo",
        "MODEL_PATH": "/models/qwen",
        "TRAIN_FILE": "/data/train.parquet",
        "UNIAGENT_ROOT": "/source/uni-agent",
        "TASK_CONFIG": "/data/task.yaml",
        "AGENT_RUNNER_FQN": "uni_agent.framework.task_runner.run_task",
        "RAY_CLUSTER_ADDRESS": "10.0.0.1:6379",
    }
    result = subprocess.run(
        [
            "bash",
            str(RECIPE_ROOT / "run_uniagent_thunderagent_vllm024_fully_async.sh"),
            "trainer.total_training_steps=3",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == child_exit, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    parsed = get_args_parser().parse_args(argv[2:])
    overrides = OverridesParser.create().parse_overrides(parsed.overrides)
    config = {item.key_or_group: item.value() for item in overrides}
    assert config["trainer.v1.trainer_mode"] == "separate_async"
    assert config["actor_rollout_ref.hybrid_engine"] is False
    assert config["trainer.n_gpus_per_node"] == config["actor_rollout_ref.rollout.n_gpus_per_node"] == 8
    assert config["trainer.nnodes"] == config["actor_rollout_ref.rollout.nnodes"] == 1
    assert config["actor_rollout_ref.rollout.tensor_model_parallel_size"] == 4
    assert config["actor_rollout_ref.rollout.checkpoint_engine.backend"] == "nccl"
    assert config["ray_kwargs.ray_init.address"] == "10.0.0.1:6379"
    assert config["ray_kwargs.ray_init.num_cpus"] is None
    modules = config["ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES"].split(",")
    assert modules.count("recipe.dynamo.register") == 1
    assert modules.count("recipe.dynamo.verl_logging_compat") == 1
    assert config["actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled"] is True
    assert config["actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.auto_finalize"] is False
    assert config["trainer.total_training_steps"] == 3
