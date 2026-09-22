# vLLM 0.24 compatibility

The [current ThunderAgent async guide](../../THUNDERAGENT_VLLM024.md) uses
unmodified Uni-Agent `0825e1ea` and its unmodified verl submodule `3efe38c7`.
Both async modes need only the Dynamo source patch:

| Patch | Baseline | Purpose |
| --- | --- | --- |
| [dynamo-client-errors.patch](dynamo-client-errors.patch) | Dynamo `8c5a73723109058f96c15fff3fc912231d65ad6e` | Support vLLM 0.24 request-error classes; includes 9 compatibility tests. |

Apply the patch once to the matching Dynamo checkout. Use `git apply --check PATCH`
before applying, or `git apply --reverse --check PATCH` to check whether it is
already present. Load the patched Dynamo `components/src` on all Ray workers.
The fully async launcher loads a scoped [logging compatibility module](../../verl_logging_compat.py)
from the recipe to accept the pinned trainer's `logger.info(..., flush=True)`.
It preserves INFO records and leaves the verl source tree unchanged.

The former Uni-Agent lifecycle patch is replaced by the recipe's
[Gateway adapter](../../uniagent_gateway.py). The former verl Qwen/FSDP patch is
replaced by **Accelerate 1.15.0** and verl's native
`fsdp_config.use_no_sync_for_gradient_accumulation=false` setting. Do not apply
the old patches to the current checkouts.

[constraints.txt](constraints.txt) preserves the tested vLLM 0.24 training
stack while installing Dynamo dependencies. It is not a complete environment
lock or a replacement for compatible CUDA/PyTorch/FlashAttention binaries.
