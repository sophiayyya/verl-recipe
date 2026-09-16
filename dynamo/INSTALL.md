# Installation guide

Use one environment per engine: **vLLM 0.28.0** or **SGLang 0.5.19**.
The commands below use the Dynamo and verl commits listed in the
[README](README.md#required-versions). The H100 benchmark also used the
[experiment-specific preparations](benchmarks/retool_h100_20260916.md#experiment-prerequisites-and-limitations).

## 1. Build and start a Dynamo container

Use a Linux GPU host with a compatible NVIDIA driver, Docker/BuildKit and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The pinned backend stack uses CUDA 13. Run the [README build commands](README.md#install-dynamo)
with `DYNAMO_ENGINE=sglang` or `vllm`. The `dev` image includes the selected engine,
Dynamo's compiled runtime, build tools, etcd and NATS.

Then, from the same host shell:

```bash
mkdir -p ../verl-workspace
docker run --rm -it --gpus all --network host --ipc host \
    -v "$PWD/../verl-workspace:/work" -w /work \
    "dynamo:8c5a737-$DYNAMO_ENGINE" bash
```

For Slurm/Pyxis, use the same image and mount a shared workspace at the same
path on every node. Use your cluster's container launcher instead of Docker.
For other image targets, see the [pinned container guide](https://github.com/ai-dynamo/dynamo/blob/8c5a73723109058f96c15fff3fc912231d65ad6e/container/README.md).

## 2. Install verl and the recipe

Run inside the container. Keep the engine's Torch, CUDA and Transformers versions
when adding training dependencies:

```bash
cd /work
python3 - <<'CHECK'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
pins = []
for name in ("torch", "torchvision", "torchaudio", "transformers", "vllm", "sglang", "nixl"):
    try:
        pins.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        pass
Path("engine-constraints.txt").write_text("\n".join(pins) + "\n")
CHECK
python3 -m pip install -c /work/engine-constraints.txt \
    accelerate codetiming datasets dill hydra-core 'numpy>=2' pandas peft \
    'pyarrow>=19' pybind11 pylatexenc 'ray[default]' torchdata \
    'tensordict==0.10.0' wandb tensorboard packaging cachetools \
    mathruler qwen-vl-utils nvtx liger-kernel ninja psutil setuptools wheel
python3 -m pip install -c /work/engine-constraints.txt \
    'TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@434f8c476b4be24bc087e6e95070e64efcc739f9' \
    'cupy-cuda13x==14.0.1'

git clone https://github.com/verl-project/verl.git
git -C verl checkout 6cbca9ce7208100d11b4d1b06eccf098cc9e76aa
git clone --branch sopy/dynamo_sglang https://github.com/sophiayyya/verl-recipe.git verl/recipe
cd verl
python3 -m pip install --no-deps -e .
```

The final `--no-deps` is intentional: this verl pin declares `transformers<5.11`,
while SGLang 0.5.19 requires 5.12.1. Its inference extras also target older engines.
The commands install training dependencies explicitly and preserve the engine
stack. A package-metadata check can therefore report the known verl/Transformers
mismatch; use the import, configuration and training checks below to validate
this combination. See [verl's installation guide](https://verl.readthedocs.io/en/latest/start/install.html)
for other training backends.

For the FSDP2 path, install FlashAttention 2 against the active Torch/CUDA build.
In this dedicated container, remove FlashAttention 4 first if present; both
packages use the `flash_attn` namespace. The SGLang recipe uses FlashInfer for
inference:

```bash
python3 -m pip uninstall -y flash-attn-4
MAX_JOBS=4 python3 -m pip install -c /work/engine-constraints.txt \
    --no-build-isolation 'flash-attn==2.8.3'
python3 -c 'import torch, flash_attn_2_cuda; print(torch.__version__, torch.version.cuda)'
```

A prebuilt FlashAttention wheel must match Python, Torch, CUDA and the C++ ABI.
For a separately prepared CUDA 12 environment, use `cupy-cuda12x` instead of
`cupy-cuda13x`; install only one CuPy package. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## 3. Existing environment: build Dynamo from source

If you already have a working verl environment with the selected engine,
activate it and install the system libraries and Rust toolchain from the
[official source-build guide](https://docs.dynamo.nvidia.com/dynamo/advanced-customizations/building-from-source).
Use the Rust version in the checkout's `rust-toolchain.toml`.

From a fresh workspace, build both the native runtime and Python components
from the same commit:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout 8c5a73723109058f96c15fff3fc912231d65ad6e
python3 -m pip install 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --release)
python3 -m pip install -e lib/gpu_memory_service
python3 -m pip install -e .
python3 -m dynamo.frontend --help
```

The base install above keeps engine selection in your prepared environment.
Dynamo also provides `.[vllm]` and `.[sglang]` extras for dependency resolution;
use them only in separate environments and recheck the training stack afterward.
A Python-only editable install does not rebuild `dynamo._core`.

Install the [etcd binary](https://etcd.io/docs/v3.5/install/) and
[nats-server binary](https://github.com/nats-io/nats-server/releases) for your
platform, and add their directory to `PATH` on every node. The recipe starts
its own services; an external Docker Compose deployment is not required.

## 4. Verify before training

Run from the verl checkout using the same interpreter as your training job:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
unset PYTORCH_CUDA_ALLOC_CONF  # required by the SGLang sleep/wake path
etcd --version
nats-server --version
python3 -c 'import torch, dynamo._core, verl, recipe.dynamo.register; assert torch.cuda.is_available()'
python3 -m dynamo.frontend --help
python3 -m verl.trainer.main_ppo \
    --config-path ../../recipe/dynamo/config \
    --config-name dynamo_trainer_v1_colocate_sglang --cfg job --resolve
```

For vLLM, use `dynamo_trainer_v1_colocate` in the configuration check.
On a GPU node, also run `python3 -m dynamo.sglang --help` or
`python3 -m dynamo.vllm --help`. Then set `MODEL_PATH`, `TRAIN_FILE` and `TEST_FILE`
and run the matching [two-step smoke](README.md#launch). CLI/configuration
checks do not exercise generation, sleep/wake or weight synchronization.

For multiple nodes, make the environment, source paths and binaries available
at identical paths before starting Ray. Mount model/data directories and use a
writable Hugging Face cache. ReTool additionally requires its prepared dataset,
reward function and a working SandboxFusion tool service; see the
[benchmark prerequisites](benchmarks/retool_h100_20260916.md#experiment-prerequisites-and-limitations).
