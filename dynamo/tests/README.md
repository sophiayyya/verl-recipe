# Dynamo tests

Run the CPU regression suite from a verl checkout with this repository at
`recipe/`, using a prepared environment from the [installation guide](../INSTALL.md):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    -p pytest_asyncio.plugin --import-mode=importlib recipe/dynamo/tests
```

[smoke/](smoke/) contains GPU generation/training smokes, the NIXL refit smoke,
and the Slurm KV-metrics validation job. Configure model/data paths and resources
using the [README smoke examples](../README.md#launch).

The launcher argument checks in [test_smoke_launchers.py](test_smoke_launchers.py)
need only pytest and hydra-core; they run the Bash scripts with a captured Python
child process and do not start training.

[check_continuations.py](check_continuations.py) checks all recipe shell/Slurm
scripts recursively by default; optional arguments select file/glob paths:

```bash
python3 recipe/dynamo/tests/check_continuations.py
```

The NIXL bandwidth microbenchmark is in [benchmarks/](../benchmarks/nixl_bench.py).
