# Installation used by the DFW H100 runs

The [ReTool results](benchmarks/retool_h100_20260916.md) used this sequence:
**existing verl image → isolated engine environment → local Dynamo wheels →
mounted Python source**. This guide follows the experiment's `build_dynamo.sh`,
`prepare.sbatch`, `setup/prepare_environment.py`, `setup/prepare_cpu.py` and
training launchers. Those experiment scripts and artifacts are stored in the
experiment archive; they are not packaged in this recipe repository.

## 1. Build the two Dynamo wheels

The wheels were built once from commit
`8c5a73723109058f96c15fff3fc912231d65ad6e` and copied into the DFW experiment's
`runtime/wheels/` directory. These are the build commands, with paths normalized:

```bash
DYNAMO_SRC=/path/to/dynamo
DYNAMO_WHEELHOUSE=/path/to/wheels
mkdir -p "$DYNAMO_WHEELHOUSE"
rustup toolchain install 1.96.1 --profile minimal
cd "$DYNAMO_SRC"
git checkout 8c5a73723109058f96c15fff3fc912231d65ad6e
uv build --wheel --out-dir "$DYNAMO_WHEELHOUSE"
cd lib/bindings/python
maturin build --release --locked \
    --features 'kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3' \
    --out "$DYNAMO_WHEELHOUSE"
sha256sum "$DYNAMO_WHEELHOUSE"/*.whl > "$DYNAMO_WHEELHOUSE/SHA256SUMS"
```

The original build reused an existing build environment, Cargo cache, libclang
and Clang headers (`CARGO_TARGET_DIR`, `LIBCLANG_PATH`, `BINDGEN_EXTRA_CLANG_ARGS`).
For a fresh build host, prepare those tools using the
[official source-build guide](https://docs.dynamo.nvidia.com/dynamo/advanced-customizations/building-from-source).
Use a runtime wheel compatible with the target container's architecture and glibc.

The recorded artifacts were:

| Wheel | Provides |
| --- | --- |
| `ai_dynamo_runtime-1.5.0-cp310-abi3-manylinux_2_39_x86_64.whl` | Compiled Rust runtime, including `dynamo._core` |
| `ai_dynamo-1.5.0-py3-none-any.whl` | Dynamo Python package |

## 2. Use the existing images and mounts

The DFW jobs used site-local images with Python 3.12:

| Engine | Base image | Version installed in the new environment |
| --- | --- | --- |
| vLLM | `verl_vllm024.dev2.sqsh` | 0.28.0 |
| SGLang | `verl_sgl0512.dev4.sqsh` | 0.5.19 |

The image filenames describe their original contents; the experiment installed
its own engine versions into `/experiment/engines/<engine>/venv`.
The same prepared environments were reused by the four formal training runs.

From an allocated Slurm CPU job, the preparation command had this shape:

```bash
EXPERIMENT_ROOT=/path/to/staged/retool-h100-20260916
DYNAMO_SRC=/path/to/dynamo
MODEL_DIR=/path/to/Qwen3-30B-A3B-Base
BASE_IMAGE=/path/to/images/verl_sgl0512.dev4.sqsh
export BENCH_ENGINE=sglang  # use vllm and its image for the other environment
srun --container-image="$BASE_IMAGE" --no-container-mount-home \
    --container-mounts="$EXPERIMENT_ROOT:/experiment,$DYNAMO_SRC/components:/dynamo/components:ro,$MODEL_DIR:/model:ro" \
    --container-workdir=/experiment \
    python3 /experiment/setup/prepare_cpu.py
```

This command expects the original staged experiment, including its setup scripts,
source/data snapshots, wheels and model files. The container mounts were:

| Container path | Contents |
| --- | --- |
| `/experiment` | Experiment archive and writable per-engine environments |
| `/dynamo/components` | Read-only Python components from the pinned Dynamo checkout |
| `/model` | Read-only Qwen3-30B-A3B-Base model |
| `/experiment/bin` | Existing `etcd` and `nats-server` binaries copied during staging |

## 3. Prepare dependencies, then install the wheels

`prepare_environment.py` created each environment with
`uv venv --python /usr/bin/python3`, without inheriting the image's Python packages.
It generated `environment-requirements.txt` from the selected engine's package
metadata, excluded `flash-attn-4`, and added the training/tool dependencies.
Key pins were Torch **2.13.0**, Transformers **5.12.1**, Ray **2.56.1**,
CuPy **14.0.1** (`cupy-cuda13x`), and NIXL **1.3.2** for vLLM / **1.4.0** for SGLang.
The engine itself was then installed with `--no-deps`; TransferQueue came from
commit `434f8c476b4be24bc087e6e95070e64efcc739f9`.

After that environment was ready, `prepare_cpu.py` installed a matching local
FlashAttention **2.8.3** wheel, verified the Dynamo wheel hashes against
`wheel-manifest.json`, and ran the following installation sequence. The paths
below use the underlying archive directories rather than its per-engine symlinks:

```bash
BENCH_ENGINE=sglang  # or vllm
DYNAMO_PYTHON="/experiment/engines/$BENCH_ENGINE/venv/bin/python"
uv pip install --python "$DYNAMO_PYTHON" \
    /experiment/runtime/"$BENCH_ENGINE"/flash_attn-*.whl
uv pip install --python "$DYNAMO_PYTHON" \
    /experiment/runtime/wheels/ai_dynamo_runtime-1.5.0-cp310-abi3-manylinux_2_39_x86_64.whl \
    /experiment/runtime/wheels/ai_dynamo-1.5.0-py3-none-any.whl \
    'aisimulate==0.12.0.dev2' 'protobuf>=6.33.5,<7'
uv pip install --python "$DYNAMO_PYTHON" --no-deps -e /experiment/src/verl
```

The Dynamo install above resolves dependencies. The `--no-deps` editable install
applies to the prepared verl snapshot, preserving the selected engine stack and
its experiment-specific compatibility changes.

The repository's [SGLang Slurm launcher](train_qwen3_30b_sglang.sh) uses the same
local-wheel/source-overlay pattern, but installs its two wheels with
`pip install --force-reinstall --no-deps`. Its default wheelhouse is historical;
set `DYNAMO_WHEELHOUSE` to the wheels for the chosen commit and `DYNAMO_SRC` to
the matching checkout, using paths visible inside its container. That launcher's
bootstrap differs from the DFW comparison's `uv` preparation above.

## 4. Load the source overlay and verify

The DFW preparation and training processes used these paths:

```bash
export PATH="/experiment/engines/$BENCH_ENGINE/venv/bin:/experiment/bin:$PATH"
export PYTHONPATH="/dynamo/components/src:/experiment/src/verl:/experiment/setup"
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
cd /experiment/src/verl
python3 -c 'import torch, flash_attn_2_cuda, dynamo._core; print(torch.__version__, dynamo._core.__file__)'
python3 -m dynamo.frontend --help
etcd --version
nats-server --version
```

Keep the source overlay and runtime wheel at the same commit. Check the actual
Python import location inside the job, for example on SGLang:

```bash
python3 - <<'CHECK'
from pathlib import Path
from dynamo.sglang.request_handlers import handler_base
path = Path(handler_base.__file__).resolve()
print(path)
assert path.is_relative_to(Path("/dynamo/components/src/dynamo"))
CHECK
```

Preparation also resolved the real trainer configs and saved dependency/import
reports. The allocated H100 job checked the backend CLI, GPU availability and
SandboxFusion before starting `verl.trainer.main_ppo`. ReTool dataset, reward,
tool-service and training-side changes are described in the
[benchmark prerequisites](benchmarks/retool_h100_20260916.md#experiment-prerequisites-and-limitations).
