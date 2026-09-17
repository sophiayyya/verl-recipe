# Installation

## Source install

Use an activated **Python 3.12** environment on Linux with verl and either
**vLLM 0.28.0** or **SGLang 0.5.19** already installed. Keep separate environments
for the two engines. Install the Python build tools into that environment:

```bash
python3 -m pip install uv 'maturin[patchelf]'
```

Rust **1.96.1** is selected by the pinned checkout's `rust-toolchain.toml`.
For system libraries, the compiler and libclang, follow the
[official source-build guide](https://docs.dynamo.nvidia.com/dynamo/advanced-customizations/building-from-source).
Build in a Linux environment compatible with the training container.

```bash
export DYNAMO_SRC=/path/to/dynamo
cd "$DYNAMO_SRC"
git checkout 8c5a73723109058f96c15fff3fc912231d65ad6e
python3 -m uv build --wheel
cd "$DYNAMO_SRC/lib/bindings/python"
python3 -m maturin build --release --locked --features kv-indexer --out "$DYNAMO_SRC/dist"
python3 -m uv pip install --python "$(command -v python3)" \
    --reinstall-package ai-dynamo --reinstall-package ai-dynamo-runtime \
    "$DYNAMO_SRC"/dist/ai_dynamo*.whl 'protobuf>=6.33.5,<7'
export PYTHONPATH="$DYNAMO_SRC/components/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import dynamo._core; print(dynamo._core.__file__)'
python3 -m dynamo.frontend --help
```

`uv build` produces the `ai_dynamo` Python wheel; `maturin build` produces the
`ai_dynamo_runtime` wheel containing `dynamo._core`. Both go into `dist/` here;
keep only this build's two wheels there. The install command replaces those two
packages even if a previous build has the same version number, while resolving
Dynamo dependencies. `--python` selects the active interpreter explicitly.

`PYTHONPATH` loads frontend/backend Python code from the same checkout. Repeat
that export in the training job, with the checkout mounted at `DYNAMO_SRC`.
Keep `etcd` and `nats-server` on `PATH`, then use the [README launch commands](README.md#launch).

## Build features

`--features kv-indexer` keeps Cargo's default features and adds the standalone
indexer. This recipe's text-only frontend KV routing does not use the other
optional services below. Enable routing at runtime with `router_mode=kv` and
`enable_kv_events=true`; the build flag alone does not configure the router.

The reduced feature selection is based on the pinned source and recipe's launch
path. Published benchmark results used the fuller build below; the reduced build
has not been rerun through the H100 comparison.

## DFW benchmark build

The actual `build_dynamo.sh` used the same commit and wheel-build sequence. To
match its feature selection, replace the `maturin` command above with:

```bash
cd "$DYNAMO_SRC/lib/bindings/python"
python3 -m maturin build --release --locked \
    --features 'kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3' \
    --out "$DYNAMO_SRC/dist"
```

These add standalone slot tracking/selection, multimodal routing, the AIC
performance model and S3 request traces. The original build reused an existing
build environment, Cargo cache and Clang paths. Its two **1.5.0** wheels were
copied to the DFW experiment's `runtime/wheels/` before environment preparation.

| Engine | Existing base image | Installed engine |
| --- | --- | --- |
| vLLM | `verl_vllm024.dev2.sqsh` | 0.28.0 |
| SGLang | `verl_sgl0512.dev4.sqsh` | 0.5.19 |

`prepare.sbatch` mounted the experiment at `/experiment`, the pinned Dynamo
components at `/dynamo/components:ro`, and the model at `/model:ro`.
`setup/prepare_environment.py` created `/experiment/engines/<engine>/venv`
without inheriting image packages. It prepared the engine/training dependencies
with Torch **2.13.0**, Transformers **5.12.1**, Ray **2.56.1**, CuPy **14.0.1**
(`cupy-cuda13x`), NIXL **1.3.2** for vLLM / **1.4.0** for SGLang, and TransferQueue
commit `434f8c476b4be24bc087e6e95070e64efcc739f9`.

`setup/prepare_cpu.py` verified wheel hashes, installed a compatible local
FlashAttention **2.8.3** wheel, then installed Dynamo and the prepared verl snapshot:

```bash
BENCH_ENGINE=sglang  # or vllm
DYNAMO_PYTHON="/experiment/engines/$BENCH_ENGINE/venv/bin/python"
uv pip install --python "$DYNAMO_PYTHON" \
    /experiment/runtime/wheels/ai_dynamo_runtime-1.5.0-cp310-abi3-manylinux_2_39_x86_64.whl \
    /experiment/runtime/wheels/ai_dynamo-1.5.0-py3-none-any.whl \
    'aisimulate==0.12.0.dev2' 'protobuf>=6.33.5,<7'
uv pip install --python "$DYNAMO_PYTHON" --no-deps -e /experiment/src/verl
export PATH="/experiment/engines/$BENCH_ENGINE/venv/bin:/experiment/bin:$PATH"
export PYTHONPATH="/dynamo/components/src:/experiment/src/verl:/experiment/setup"
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
```

The four formal runs reused those prepared environments and launched
`verl.trainer.main_ppo`. The experiment scripts/artifacts are archived separately
from this recipe. See the [benchmark prerequisites](benchmarks/retool_h100_20260916.md#experiment-prerequisites-and-limitations)
for the ReTool data, tool service and training compatibility changes.

The repository's [SGLang Slurm launcher](train_qwen3_30b_sglang.sh) also installs
local wheels and loads source through `PYTHONPATH`, but uses
`pip install --force-reinstall --no-deps` in its prepared environment. Override
its historical `DYNAMO_WHEELHOUSE` default with your build's `dist/` directory
and set `DYNAMO_SRC` to the matching checkout, both visible inside the container.
