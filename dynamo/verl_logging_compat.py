"""Keep verl 3efe38c7's fully async setup log working without a source patch.

That revision passes print's ``flush=True`` to Logger.info. StreamHandler
already flushes each record. Adapt only this logger instance, preserving its
level, handlers and all other arguments. The fully async launcher loads this
module through VERL_USE_EXTERNAL_MODULES in the driver and Ray workers.
"""

import logging
from functools import wraps


def install():
    logger = logging.getLogger("verl.trainer.ppo.v1.trainer_separate_async")
    if getattr(logger.info, "_dynamo_flush_compat", False):
        return
    original_info = logger.info

    @wraps(original_info)
    def info(message, *args, **kwargs):
        kwargs.pop("flush", None)
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        return original_info(message, *args, **kwargs)

    info._dynamo_flush_compat = True
    logger.info = info


install()
