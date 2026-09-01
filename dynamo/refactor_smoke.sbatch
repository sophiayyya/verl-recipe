#!/bin/bash
#SBATCH --job-name=refactor_smoke
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/sopyang/logs/refactor_smoke_%j.out
set -x
B=/lustre/fsw/portfolios/coreai/users/sopyang
V=$B/verl_dynamo/verl

cat > $B/logs/_smoke_vllm.py <<'PY'
import sys; sys.path.insert(0, "/lustre/fsw/portfolios/coreai/users/sopyang/verl_dynamo/verl")
import recipe.dynamo.register as reg
from verl.workers.rollout.base import _ROLLOUT_REGISTRY
assert _ROLLOUT_REGISTRY[("dynamo","async")] == "recipe.dynamo.dynamo_rollout.ServerAdapter"
print("A registry entry intact            : OK")
from recipe.dynamo.dynamo_rollout import _load_vllm_adapter, _dynamo_engine
A = _load_vllm_adapter()
assert A.__module__ == "recipe.dynamo.dynamo_vllm_rollout", A.__module__
assert A.__name__ == "VllmDynamoServerAdapter"
print("B vllm adapter from new module     : OK", A.__module__)
import verl.workers.rollout.vllm_rollout.vllm_rollout as vr
assert issubclass(A, vr.ServerAdapter)
print("C still subclasses verl ServerAdapter: OK")
for m in ("_get_control_actor_name","_get_server_name_prefix","update_weights","_execute_method"):
    assert m in A.__dict__, m
print("D methods survived the move        : OK")
import recipe.dynamo.dynamo_rollout as dr
assert dr.VllmDynamoServerAdapter is A
print("E PEP562 re-export identity        : OK")
from recipe.dynamo.dynamo_naming import control_actor_name
assert control_actor_name({"dynamo":{"shared_pool_replica_rank":0}},1)=="dynamo_server_0_1"
print("F control_actor_name               : OK")
import types
assert _dynamo_engine(types.SimpleNamespace(engine_kwargs={"dynamo":{}}))=="vllm"
print("G engine default = vllm            : OK")
print("SMOKE_VLLM_PASS")
PY

cat > $B/logs/_smoke_sgl.py <<'PY'
import sys; sys.path.insert(0, "/lustre/fsw/portfolios/coreai/users/sopyang/verl_dynamo/verl")
import recipe.dynamo.dynamo_rollout as dr
assert not any(m.startswith("verl.workers.rollout.vllm_rollout") for m in sys.modules), "facade dragged in vLLM"
print("A facade imports, no vLLM pulled    : OK")
import recipe.dynamo.register
from recipe.dynamo.dynamo_sglang_rollout import SGLangServerAdapter as S
assert S.__module__ == "recipe.dynamo.dynamo_sglang_rollout"
assert "_get_control_actor_name" in S.__dict__
print("B sglang adapter + control name     : OK")
from recipe.dynamo.dynamo_naming import control_actor_name
assert control_actor_name({"dynamo":{"shared_pool_replica_rank":0}},1)=="dynamo_server_0_1"
print("C control_actor_name                : OK")
import types
assert dr._dynamo_engine(types.SimpleNamespace(engine_kwargs={"dynamo":{"engine":"sglang"}}))=="sglang"
print("D engine=sglang dispatch key        : OK")
print("SMOKE_SGL_PASS")
PY

echo "########## vllm024 image ##########"
srun --overlap --nodes=1 --ntasks=1 \
  --container-image=$B/images/verl_vllm024.dev2.sqsh \
  --container-mounts="$B:$B" --no-container-mount-home \
  bash -c "cd $V && python3 $B/logs/_smoke_vllm.py" || echo "VLLM_SMOKE_FAILED"

echo "########## sgl0512 image ##########"
srun --overlap --nodes=1 --ntasks=1 \
  --container-image=$B/images/verl_sgl0512.dev4.sqsh \
  --container-mounts="$B:$B" --no-container-mount-home \
  bash -c "cd $V && python3 $B/logs/_smoke_sgl.py" || echo "SGL_SMOKE_FAILED"
echo "########## smoke done ##########"
