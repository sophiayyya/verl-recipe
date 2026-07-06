# Dynamo ThunderAgent rollout

This recipe extends the Dynamo rollout backend from
[verl-recipe PR #110](https://github.com/verl-project/verl-recipe/pull/110)
with program-aware scheduling for multi-turn agent trajectories. It does not
modify core `verl` or Dynamo.

## Required versions

- Core `verl`: see [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Dynamo: source commit `48632da9c77c5a7647b50cf1ba2a729dcdca7aea`.

ThunderAgent is experimental and is not included in a released Dynamo Python
package. Build or install Dynamo from the commit above; `etcd`, `nats-server`,
`dynamo.vllm`, `dynamo.thunderagent_router`, and `dynamo.frontend` must be
available in the runtime image.

## Request lifecycle

Each `AgentLoopBase.run()` is one ThunderAgent program:

1. The recipe creates one random session ID when a trajectory starts.
2. Every LLM turn sends the same `X-Dynamo-Session-ID`.
3. Concurrent trajectories use task-local, isolated session IDs.
4. On success, error, or cancellation, admitted requests drain before one
   `X-Dynamo-Session-Final: true` request releases the program.

The final request must return no model choices. A non-empty completion is
treated as a routing bypass and fails closed.

## Topology

With ThunderAgent enabled, PR #110 launches processes in this order:

```text
etcd -> NATS -> Dynamo vLLM workers -> ThunderAgent router -> frontend
```

The frontend uses round-robin only to reach the registered ThunderAgent
handler. ThunderAgent owns the internal KV router and forwards to the worker
endpoint `<namespace>.backend.generate`. Shutdown reverses the consumer side:
frontend, ThunderAgent, workers, NATS, then etcd.

## Configuration

The default recipe enables ThunderAgent:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
    engine_kwargs:
      dynamo:
        thunderagent:
          enabled: true
          router_block_size: 16
```

`router_block_size` is applied to both vLLM and ThunderAgent. Pass scheduler
CLI options without hard-coding them in the recipe:

```yaml
thunderagent:
  enabled: true
  router_block_size: 16
  extra_args:
    - --pause-threshold
    - "0.95"
```

Optional finalization controls are `finalize_max_attempts` (default `3`) and
`finalize_retry_delay_s` (default `0.1`). Set `thunderagent.enabled=false` for
the PR #110 KV-router baseline.

## Run

From a verl checkout containing this repository at `recipe/`:

```bash
VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register \
python -m recipe.dynamo.main_dynamo \
  actor_rollout_ref.model.path=/path/to/model \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

Supply the remaining dataset, trainer, and resource overrides required by the
standard verl PPO configuration.

## Tests

```bash
pytest -q recipe/dynamo/tests
ruff check recipe/dynamo/thunderagent.py recipe/dynamo/dynamo_thunderagent.py \
  recipe/dynamo/dynamo_agent_loop.py recipe/dynamo/main_dynamo.py \
  recipe/dynamo/register.py recipe/dynamo/tests
ruff format --check recipe/dynamo/thunderagent.py recipe/dynamo/dynamo_thunderagent.py \
  recipe/dynamo/dynamo_agent_loop.py recipe/dynamo/main_dynamo.py \
  recipe/dynamo/register.py recipe/dynamo/tests
```

GPU validation should run the same model, prompts, concurrency, warmup, and
measurement window twice, changing only `thunderagent.enabled`. Record
throughput, latency, KV-cache hit rate, errors, and router pause/resume logs.
