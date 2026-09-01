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
"""Engine-agnostic facade for the dynamo rollout backend.

``register.py`` points ``rollout.name=dynamo`` at ``ServerAdapter`` below, which
dispatches to the per-engine adapter named by ``engine_kwargs.dynamo.engine``.
Both engine implementations live in their own modules and are imported lazily,
so this module stays importable on an image that ships only one of them.
"""


class ServerAdapter:
    """Dispatch to the per-engine adapter named by ``engine_kwargs.dynamo.engine``.

    Registered as the single ``dynamo`` rollout backend so that switching engines
    stays a config change — ``rollout.name=dynamo`` plus
    ``engine_kwargs.dynamo.engine=vllm|sglang`` — matching how verl selects every
    other backend. An earlier version registered a second ``dynamo_sglang`` name
    with its own config and entry point; that put the engine choice in three places
    that had to agree, and made ``dynamo_sglang`` miss verl's own
    ``rollout.name == "sglang"`` special-cases for no benefit.

    ``_ROLLOUT_REGISTRY`` stores an FQDN that ``get_rollout_class`` imports and then
    calls, so the registered object only has to be callable — it does not have to be
    the adapter class itself. ``__new__`` returns a fully constructed instance of the
    chosen implementation; Python skips ``ServerAdapter.__init__`` because that
    instance is not a ``ServerAdapter``.
    """

    def __new__(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        engine = str(_dynamo_engine(config)).lower()
        if engine == "sglang":
            from recipe.dynamo.dynamo_sglang_rollout import SGLangServerAdapter

            impl = SGLangServerAdapter
        elif engine == "vllm":
            impl = _load_vllm_adapter()
        else:
            raise ValueError(
                f"rollout.engine_kwargs.dynamo.engine must be 'vllm' or 'sglang', got {engine!r}"
            )
        return impl(*args, **kwargs)


def _dynamo_engine(config) -> str:
    """Read ``engine_kwargs.dynamo.engine`` off a DictConfig or a RolloutConfig."""
    if config is None:
        return "vllm"
    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    dynamo_cfg = engine_kwargs.get("dynamo", {}) or {}
    return dynamo_cfg.get("engine", "vllm")


def _load_vllm_adapter():
    """Import the vLLM adapter, failing loudly if this image has no vLLM.

    verl's ``workers/rollout/vllm_rollout/__init__`` raises PackageNotFoundError
    (not ImportError) on a vLLM-free image such as ``verlai/verl:sgl*.dev``,
    which used to surface as a TaskRunner crash long after bootstrap looked
    healthy (job 16510923). Keeping the import in here rather than at module
    scope is what lets the sglang path run on those images at all.
    """
    try:
        from recipe.dynamo.dynamo_vllm_rollout import VllmDynamoServerAdapter
    except Exception as exc:  # noqa: BLE001 - PackageNotFoundError, not ImportError
        raise RuntimeError(
            "engine_kwargs.dynamo.engine=vllm needs the vLLM package, which is not "
            "installed in this image. Use engine=sglang here, or run on an image "
            f"that ships vLLM. Original import error: {exc!r}"
        ) from exc
    return VllmDynamoServerAdapter


__all__ = ["ServerAdapter", "VllmDynamoServerAdapter"]


def __getattr__(name):
    # PEP 562 lazy re-export: keeps the old
    # ``from recipe.dynamo.dynamo_rollout import VllmDynamoServerAdapter`` working
    # without dragging vLLM into the import of this engine-agnostic facade.
    if name == "VllmDynamoServerAdapter":
        return _load_vllm_adapter()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
