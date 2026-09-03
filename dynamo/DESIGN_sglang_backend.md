> **文件改名对照（2026-09-03）**：`smoke_dynamo_v1.sh`→`smoke_vllm_generate.sh`，`train_30b_dynamo_sglang_4n.sh`→`train_qwen3_30b_sglang.sh`，`train_30b_sglang_native_i100.sh`→`baseline_qwen3_30b_sglang_native.sh`，`retool_ab_bootstrap.sh`→`container_bootstrap.sh`。下文沿用旧名。

# 设计：把 SGLang 加进 verl × Dynamo rollout backend

> **历史设计文档（2026-08-21）。** 实现过程中若干细节已变：不再有 `dynamo_sglang` 这个 rollout name 和
> `dynamo_sglang_trainer.yaml`，引擎通过 `engine_kwargs.dynamo.engine=sglang` 切换；入口是
> `verl.trainer.main_ppo trainer.use_v1=False`。以 README.md 为准。

- 仓库：`/lustre/fsw/portfolios/coreai/users/sopyang/verl_dynamo/verl`
  （core verl @ `6cbca9ce`，`recipe/` 子模块 = `sophiayyya/verl-recipe` 分支 `sopy/dynamo_next` @ `ab641fe`）
- 参考：
  - [slime#2272](https://github.com/THUDM/slime/pull/2272)（streaming external SGLang rollout）
  - [dynamo#11640](https://github.com/ai-dynamo/dynamo/pull/11640)（SGLang engine-native `/generate`）
  - [verl-recipe#136](https://github.com/verl-project/verl-recipe/pull/136)（dynamo NIXL weight-sync + 多机，**open，未进本分支**）
- 本地 dynamo checkout：`/lustre/.../sopyang/dynamo` @ `94accc7389`（= ThunderAgent PR #11185）

---

## 0. 结论先说

三句话：

1. **控制面几乎是白送的。** `dynamo.sglang` 的 `BaseWorkerHandler.register_engine_routes()`
   已经原生注册了 `control/update_weights_from_{tensor,ipc,disk,distributed}`、
   `control/release_memory_occupation`、`control/resume_memory_occupation`、
   `control/update_weight_version`、`control/start_profile|stop_profile`、KV-cache flush，
   全部挂在 worker 的 system-status server 上（`/engine/{*path}`，端口 `DYN_SYSTEM_PORT`）。
   → **不需要**为 sglang 复刻 `_dynamo_vllm_with_control.py` 那套 ZMQ REQ/REP sidecar。
   这是整个改造里最省力的一块，也是和 vLLM 路径最大的结构差异。

2. **权重同步要换协议，不能沿用 vLLM 的 `worker_extension_cls`。**
   sglang 没有 vLLM 的 `worker_extension_cls` 钩子，verl 侧现成的路径是
   `sglang.srt.weight_sync.utils.update_weights` → `update_weights_from_tensor`
   （CUDA-IPC handle 经 `MultiprocessingSerializer` 序列化）。
   所以 `dynamo_worker_extension.py` + `BucketedWeightSender` + `VERL_DYNAMO_RANK_OFFSET`
   这一串在 sglang 路径上整体作废，换成"每个 DP shard 一个 HTTP 控制端点 + 按桶发 IPC handle"。

3. **生成面要分两步走。** 当前 pin 的 dynamo commit **没有** #11640 的
   `components/src/dynamo/sglang/engine_generate.py`（本地 checkout 里 sglang 目录下无
   `engine_generate.py` / `engine_routes.py`）。所以：
   - M1 先用现有的 frontend `/v1/completions`（和 vLLM 路径同一条），拿 KV router + 端到端跑通；
   - M3 升 dynamo 到含 #11640 的版本，切到 sglang-native streaming `/generate`（真 TITO），
     客户端按 slime#2272 的 `SGLangStreamAccumulator` 形状写增量累积 + request 级 abort。

4. **（追加）verl-recipe#136 是正交轴，不是替代方案。** 它给 dynamo recipe 加了
   `checkpoint_engine.backend=nixl`，解决的是 **trainer → CE worker 的跨机传输**；
   而 **CE worker → 推理引擎的最后一跳仍然是 `server_adapter.update_weights()`**
   （`verl/checkpoint_engine/base.py:355-361`：`weights = checkpoint_engine.receive_weights(...)`
   → `await self.server_adapter.update_weights(weights, wire_format=...)`）。
   两件事要分别做。（早先这里写的"b64 缺口不会被 #136 消掉"已作废——那个缺口本身就不存在，见风险 1。）
   反过来，#136 里为 vLLM 硬凑的 `release_kv_cache()/resume_kv_cache()`（sleep_level=1 语义）
   在 sglang 上是**原生**的 `release_memory_occupation(tags=["kv_cache"])` —— sglang 反而更贴。

---

## 1. 现状：vLLM 路径的 5 个耦合点

`recipe/dynamo/` 现在是 vLLM-only。把它拆成 5 个耦合点，逐个看 sglang 要怎么接：

| # | 耦合点 | 现有实现 | 关键文件:行 |
|---|--------|----------|-------------|
| C1 | 进程编排 | 每个 DP shard 一个 `python -m recipe.dynamo._dynamo_vllm_with_control` 子进程，切 `CUDA_VISIBLE_DEVICES` | `dynamo_async_server.py:533` `_start_vllm_workers` / `:795` `_build_vllm_cmd` |
| C2 | 控制面 | 自建 ZMQ REP sidecar，桥接 `engine.collective_rpc` 和 `getattr(engine, m)()` | `_dynamo_vllm_with_control.py` 全文；`dynamo_async_server.py:1679` `collective_rpc` / `:1810` `_engine_method_all` |
| C3 | 权重同步 | trainer rank → `BucketedWeightSender` → `/tmp/rl-colocate-zmq-*.sock` → vLLM worker extension `update_weights_from_ipc` | `dynamo_vllm_rollout.py` `update_weights`；`dynamo_worker_extension.py:47` `_get_zmq_handle` |
| C4 | 生成面 | frontend `POST /v1/completions`，`prompt=[token_ids]`，靠 `nvext.engine_data` / `nvext.completion_token_ids` / `logprobs=0` 取回 token id | `dynamo_async_server.py:1091` `generate` / `:1206` `_build_frontend_completion_payload` |
| C5 | 注册 & 生命周期 | `register.py` 把 `dynamo` 注册进两个 registry；`ServerAdapter` 是引擎无关门面，`VllmDynamoServerAdapter` 继承 **vLLM** 的 ServerAdapter | `register.py:17-18`；`dynamo_vllm_rollout.py` |

---

## 2. sglang 侧的对应物

| # | vLLM 现状 | SGLang 对应 | 改造量 |
|---|-----------|-------------|--------|
| C1 | `-m recipe.dynamo._dynamo_vllm_with_control` + vLLM CLI | 直接 `-m dynamo.sglang` + sglang `ServerArgs` CLI（`args.py:338` `ServerArgs.add_cli_args`，全量透传） | 中：只是换 cmd builder + 参数名映射 |
| C2 | 自建 ZMQ sidecar | **原生** `/engine/control/*`（`request_handlers/handler_base.py:958-993` `register_engine_routes`），走 worker 的 `DYN_SYSTEM_PORT` HTTP | **小：删代码**。sidecar 整个不需要 |
| C3 | `worker_extension_cls` + bucket ZMQ IPC | `update_weights_from_tensor`（`MultiprocessingSerializer` CUDA-IPC，verl 已有 `sgl_update_weights` 封装）或 `update_weights_from_ipc`（checkpoint-engine 协议） | **大**：这是主要工作量，见 §5 D2 |
| C4 | `/v1/completions` + nvext | 同一条 frontend 路可用（M1）；#11640 后有 sglang-native `POST /generate`（`input_ids` + `stream:true` + `n=1`，`DYN_SGLANG_ENABLE_GENERATE=1`） | 中：M1 复用，M3 新写 streaming 客户端 |
| C5 | 继承 vLLM ServerAdapter | 应继承 `verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter` | 小 |
| C6 | sleep/wake = vLLM `enable_sleep_mode` + `engine.sleep(level=1)` | `release_memory_occupation(tags=["kv_cache","weights"])` / `resume_memory_occupation`，**要求 sglang 起时带 `--enable-memory-saver`** | 小，但漏了会静默退化 |
| C7 | KV events 靠 `--kv-events-config` JSON 显式配 | dynamo sglang 有 `publisher.py::DynamoSglangPublisher`，直接从 sglang 的 `ZmqEventPublisher` 收再转发 NATS，**不需要** `--kv-events-config` | 小：删掉那段 |
| C8 | checkpoint-engine 第 1 跳：#136 加了 `naive`/`nixl` + CE actor 编排 + `release/resume_kv_cache` | 后端无关，直接继承；`release/resume_kv_cache` 在 sglang 上比 vLLM 更自然（原生 tags） | 小，但**必须先 rebase #136**，见 D6 |

---

## 3. 两个参考 PR 各自提供什么

**dynamo#11640 → 解决"生成面"**
- 新增 `POST/PUT /generate`（SGLang 引擎原生 envelope），契约：一条非空 `input_ids`、
  `stream: true`、`sampling_params.n == 1`；文本 / 多模态 / batch / 非流式 / `n>1` 都不在范围内。
- 开关：`DYN_SGLANG_ENABLE_GENERATE=1`（或 builder 的 `enable_engine_apis`）；
  路径可用 `DYN_HTTP_SVC_SGLANG_GENERATE_PATH` 覆盖。
- 非路由字段整包透传在 `extra_args.sglang_tito`，worker 端 `engine_generate.py` 按**当前装的 sglang 版本**
  重建 `GenerateReqInput`（不维护字段镜像）——这点对我们很重要：不用担心 dynamo 和 sglang 版本字段漂移。
- capability-scoped 挂载：worker 发布 `sglang_generate` capability，路由不会把 envelope 发错后端。
- 有 12 个内部字段的 denylist（`bootstrap_host/port/room`、`routed_dp_rank` 等），客户端注入会被拒。

对我们的意义：**这才是 RL 想要的 token-in/token-out**。现在 vLLM 路径靠
`return_tokens_as_token_ids` + `logprobs=0` 从 `choice.logprobs.tokens` 里抠 `token_id:<id>` 字符串
（`dynamo_async_server.py:1515` `_extract_completion_token_ids` 那一大坨），是绕路。
sglang 走 `/generate` 可以直接拿 `output_ids`。

**slime#2272 → 解决"客户端消费面"**
- `slime/rollout/sglang_streaming_rollout.py::generate_streaming`，入口靠
  `--custom-generate-function-path` 挂进去。
- `SGLangStreamAccumulator`：把两种 SSE 流形态（**cumulative**（默认，校验 reported length，只 apply 未见后缀）
  和 **incremental**（要 server 与 client 同时开 `--incremental-streaming-output` /
  `--sglang-incremental-streaming-output`））归一成同一种 sample 更新。格式不匹配直接拒。
- 增量累积的东西包括：tokens、logprobs、top-p 元数据、**routed-expert 数据**、response text。
- request 级 abort：generate 函数上打属性 `generate_streaming.abort_mode = "request"`，
  框架把"可 abort 的 request"和"server-wide abort"分开跟踪，只对**故意取消**的 task 转换
  `CancelledError`；partial rollout 恢复时跳过已完成的 sample，只让被 abort 的兄弟 sample 续跑。

对我们的意义：verl 这边要做**同构**的一层。而且我们比 slime 多一层
——中间隔着 dynamo frontend/router，abort 要能穿透到 dynamo 的 request lifecycle
（#11640 里 `sglang_generate.rs` 有 cancellation + SSE lifecycle，靠 HTTP 连接断开传播）。

**verl-recipe#136 → 解决"跨机权重传输 + 多机拓扑"**（状态：**open**，本分支 `sopy/dynamo_next` 未合）
- 给 dynamo recipe 接上 verl 的 CheckpointEngine 体系，`checkpoint_engine.backend=nixl`，
  在 2×8×H100 RDMA 上验过端到端多机 GRPO（PR #110 只验过单机）。
- `_DynamoCheckpointEngineWorker`：`num_gpus=0` 的 Ray actor，靠注入 `CUDA_VISIBLE_DEVICES`
  + 覆写 `_setup_env_cuda_visible_devices` 来和 trainer 的 placement-group bundle 同卡配对
  （CUDA-IPC 交接的硬前提）。`DynamoReplica.init_hybrid_worker_pool` 每个 rollout rank 起一个。
- CE gloo group 的 `MASTER_ADDR` 改为从 `ray.nodes()` 查 head trainer worker 的 node_id 解析
  （原来硬编码 `127.0.0.1`，多机必挂）。
- **只在 non-naive backend 时才起 CE actor**，避免 IPC socket 命名空间冲突。
- `DynamoHttpServer` 新增 `release_kv_cache()` / `resume_kv_cache()`；
  `ServerAdapter.update_weights` 入口处先把引擎唤醒（weights + kv_cache）；
  `_engine_method_all` 的 ZMQ recv 超时 120s → 600s。
- 新文件 `run_nixl_smoke.sh`（可配 NNODES / CE_BACKEND / MODEL_PATH 的 3-step GRPO smoke）、
  `nixl_bench.py`（跨机 NIXL 带宽诊断）。
- 性能：Qwen2.5-0.5B naive 3.08s vs NIXL 3.09s（持平）；Qwen3-8B 29.7s → 27.1s（−8.6%）。
  传输层 1 GiB 跨机读：**UCX 0.23 GB/s vs LIBFABRIC 48.5 GB/s** —— 差 200 倍，
  因为他们的 fabric 没有 native RDMA-read，UCX 退化成 send/recv 模拟。
  选 LIBFABRIC 需要 verl 侧一个小改动（`NIXLCheckpointEngine` 加 `backends` kwarg）。
  有 native RDMA-read 的 fabric 则用 UCX + `UCX_TLS=cuda_ipc,cuda_copy,rc,tcp`。
- 容器要求：fabric RDMA device 资源 + `IPC_LOCK` capability。

对我们的意义：见 §5-D6。一句话——**它和 sglang 是正交的两件事，但 M4 多机必须两者都有**，
而且 sglang 让 #136 里最别扭的那块（release/resume kv_cache）变干净。

---

## 4. 目标架构

```
 verl trainer (colocated, 占 GPU)
   │  ① 生成：HTTP → dynamo frontend（KV router / ThunderAgent）
   │  ② 控制：HTTP → 每个 worker 的 DYN_SYSTEM_PORT /engine/control/*
   │  ③ 权重：CUDA-IPC handle（同卡，走 ②的通道带过去）
   ▼
 ┌──────────── DynamoHttpServer (Ray actor, 1/node, 不占 GPU) ─────────────┐
 │  etcd + nats                                                            │
 │  dynamo.frontend ──► KV router / ThunderAgent ──► dynamo.sglang × N      │
 │                                                    (一个 DP shard 一个)  │
 │                                                    每个自带                │
 │                                                      /engine/control/*   │
 │                                                      /metrics            │
 └─────────────────────────────────────────────────────────────────────────┘
```

和 vLLM 版本的图相比：**少了一层 ZMQ 控制 sidecar**，多了"trainer 直接 HTTP 打 worker
system port"这条线。`DynamoHttpServer` 依然是编排者，但对 sglang 它只需要记住
`(host, DYN_SYSTEM_PORT)` 列表，不需要 `_control_endpoints` 的 ZMQ。

---

## 5. 关键设计决策

### D1 — 控制面：走 `/engine/control/*`（推荐）

依据：`components/src/dynamo/sglang/request_handlers/handler_base.py:958` 起，
`register_engine_routes(runtime)` 无条件注册全部 control 路由；
`lib/runtime/src/system_status_server.rs:189` 把 `/engine/{*path}` 挂到 system status server。

做法：
- 复用现有的 `_allocate_stable_node_port(_SYSTEM_METRICS_PORT_BASE, ...)` 和
  `self._worker_metrics_endpoints`（`dynamo_async_server.py:588-635`）——现在只用来做 metrics 抓取，
  sglang 路径下它同时是**控制端点**。把 `enable_worker_system_metrics` 对 sglang 强制为 True，
  否则控制面直接没有。
- `DynamoHttpServer` 上加 `_engine_control_all(route, body)`：对 node 内所有 shard
  `asyncio.gather` POST `http://{host}:{sys_port}/engine/{route}`。
  这就是 `_engine_method_all` 的 HTTP 版，语义一一对应。
- `DYN_SYSTEM_PORT` 的 i16 坑（`dynamo_async_server.py:620-632` 注释）照旧：必须用 <32768 的固定低端口。

被否的方案：复刻 `_dynamo_sglang_with_control.py`。只有一个场景需要它——
要调 tokenizer_manager 上**没有** engine route 的方法。但 dynamo 已经给了逃生口：
`--enable-rl` 时注册 `call_tokenizer_manager`（`handler_base.py:163`），
body 是 `{"method":..., "args":[...], "kwargs":{...}}`，参数支持
`{"io_struct.ClassName": {kwargs}}` 的 typed constructor。够用。

### D2 — 权重同步：先分清两跳，再选协议

**先把两跳分开，否则会和 #136 打架：**

```
 trainer (ME)  ──第 1 跳──►  CE worker  ──第 2 跳──►  推理引擎
               naive / nccl /            server_adapter.update_weights()
               nixl / mooncake /         （CUDA-IPC，同卡）
               kimi / delta_sharded
               ↑ #136 加的是这一跳         ↑ 本文 D2 说的是这一跳
```

依据 `verl/checkpoint_engine/base.py:355-361`：

```python
async def update_weights(self, global_steps=None):
    weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
    await self.server_adapter.update_weights(
        weights, global_steps=global_steps,
        wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"))
```

CheckpointEngine 是**后端无关**的：它只负责把权重搬到和推理引擎同卡的 CE worker 上，
最后一跳永远回到 rollout adapter。所以 **#136 的 NIXL 不能替我们解决 sglang 的第 2 跳**，
下面讲的全是第 2 跳。第 1 跳的选型见 D6。

**第 2 跳：两条路，建议先 tensor 后 ipc**

**方案 A（M2 首选）：`update_weights_from_tensor`**
- verl 侧几乎全是现成的：`sglang_rollout.py:371` 的
  `get_named_tensor_buckets(weights, bucket_bytes)` + `sgl_update_weights(engine, params_batch, "infer_tp", device_mesh)`。
  `sgl_update_weights` 自己在 `infer_tp` mesh 上做 gather，只在 TP rank 0 发 HTTP。
- 我们要做的是把 `self._engine`（verl 的 `AsyncHttpServerAdapter`）换成一个
  "打 dynamo `/engine/control/update_weights_from_tensor`" 的 adapter。
- **✅ 已实测可用（无需任何补丁）**：JSON 没有 bytes，所以 blob 走 base64 上线 ——
  而这正是 sglang 的契约，不是变通。`UpdateWeightsFromTensorReqInput.serialized_named_tensors`
  的类型是 `List[Union[str, bytes]]`，`MultiprocessingSerializer.deserialize` 见到 `str`
  就 `pybase64.b64decode(data, validate=True)`。dynamo 的 engine route 原样透传 JSON 是正确行为。
  未打补丁的 dynamo 1.3.0 + sglang 0.5.14 上实测 `{"success":true,"message":"Success"}`
  （M0c，job 16215105，真实 `model.embed_tokens.weight` / shape (151936,896) / bf16）。
- **⚠️ 反序列化失败 = 整个 worker 进程死掉**，不是返回错误。M0c 用一段随机字节做载荷，
  worker 直接 `RemoteDisconnected`，之后所有请求 `Connection refused`。
  含义：**权重同步的载荷格式错误是"引擎重启"级事故，不是可重试的失败**，
  而且调用方无法区分"载荷坏了"和"worker 因别的原因崩了"。所以线上格式必须严格照 sglang 的规定来。
  （副作用：M0c 里 badshape / badname 两臂因为 worker 已死没跑成，这两个仍是未知。）

**方案 B（M4 性能优化）：`update_weights_from_ipc`**
- sglang 的 `UpdateWeightsFromIPCReqInput` 是 checkpoint-engine 协议（每 TP rank 一个 zmq handle，
  按 bucket 收），和 verl 的 `BucketedWeightSender` 是同源设计。
  理论上能把现有 vLLM 路径的 bucket sender 直接复用过来，省掉 b64 和 JSON 开销。
- 代价：要自己算 zmq handle 的 rank 映射（sglang TP rank ↔ trainer rank），
  等价于重做一遍 `VERL_DYNAMO_RANK_OFFSET` 的活。**不要在 M2 做**。

**Rank 映射（两条路都要）**：dynamo 一个 DP shard = 一个 `sgl.Engine`（`init_llm.py:78`），
自带 TP 组。verl 的 `infer_tp` mesh 必须和 shard 的 TP 组对齐，而且
"哪个 trainer rank 负责哪个 shard 的控制端点"要显式建表。
vLLM 路径靠 `VERL_DYNAMO_RANK_OFFSET` 环境变量 + socket 路径隐式对齐；
sglang 路径要改成显式的 `shard_idx = local_rank // tp` → `self._worker_control_endpoints[shard_idx]`。
这是**最容易出 silent bug 的地方**（对错了不报错，只是权重串了）。
M2 必须带一个校验：更新完权重后对同一个 prompt 做 greedy 采样，比对 trainer forward 的 argmax。

### D3 — 生成面：M1 复用 completions，M3 切 native generate

M1：直接复用 `_build_frontend_completion_payload`。需要确认 sglang decode handler
在 `/v1/completions` 上认 token-id prompt 和 `return_tokens_as_token_ids`
（`decode_handler.py:317,349` 有 `return_tokens_as_token_ids`，看起来认）。
`nvext.engine_data` / `nvext.completion_token_ids` 是否在 sglang handler 上有实现要实测——
`dynamo_async_server.py:1119` 已有"400 + Unsupported parameter → 去掉扩展字段重试"的 fallback，
所以最坏情况是静默退化到 detokenize 再 re-encode（RL 不可接受，要在日志里显式告警）。

M3：升 dynamo 到含 #11640 的 commit，`DYN_SGLANG_ENABLE_GENERATE=1`，
新写 `dynamo_sglang_stream.py`，形状照抄 slime 的 `SGLangStreamAccumulator`：
- 默认 cumulative 模式（校验 server 报的 length，只 apply 未见后缀）；
- incremental 模式要 server `--incremental-streaming-output` 和 client 开关**同时**打开，
  不匹配就直接抛错，不要试图自动探测；
- 累积 tokens / logprobs / finish_reason，MoE 场景额外累积 routed-expert 数据；
- abort：断 HTTP 连接 → dynamo `sglang_generate.rs` 的 cancellation 传到 worker；
  verl 侧保留已观测前缀，partial rollout 时只续被 abort 的样本。

### D4 — sleep / wake

`free_engine_on_train=true` 时，`DynamoHttpServer.sleep/wake_up` 改成打
`/engine/control/release_memory_occupation` / `resume_memory_occupation`，
body 里带 `tags`。**前置条件：sglang 起时必须有 `--enable-memory-saver`**，
否则 torch_memory_saver 没装好，release 是 no-op 或直接报错。

另外 `verl/workers/engine_workers.py:774` 有一段
`is_sglang = self.config.rollout.get("name","") == "sglang"`，
决定 sleep_level=1 时要不要 `resume(tags=["weights"])`
（sglang 的 level-1 不释放 weights，vLLM 释放）。
我们的 rollout name 不叫 `sglang`，会走到 vLLM 分支 → 多余的 resume。
LoRA / sleep_level=1 场景下必须处理，见 D5。

### D5 — 注册名

推荐：**`rollout.name=dynamo` + `engine_kwargs.dynamo.engine=sglang|vllm`（默认 vllm）**，
同时在 registry 里额外注册一个 `dynamo_sglang` 别名指向同一套类，方便 A/B 脚本。

`register.py` 改成：

```python
RolloutReplicaRegistry.register("dynamo", _load_dynamo)            # 按 engine 分派
RolloutReplicaRegistry.register("dynamo_sglang", _load_dynamo_sglang)
_ROLLOUT_REGISTRY[("dynamo", "async")] = "recipe.dynamo.dynamo_rollout.ServerAdapter"
_ROLLOUT_REGISTRY[("dynamo_sglang", "async")] = "recipe.dynamo.dynamo_sglang_rollout.SGLangServerAdapter"
```

`_ROLLOUT_REGISTRY` 的 value 是类的 FQDN 字符串、由 `get_rollout_class` 直接 import
（`verl/workers/rollout/base.py:105`），**没有按配置分派的钩子**。所以
"一个 name 两个 engine"必须在 `ServerAdapter` 内部用组合（持有一个 backend 对象）而不是继承来实现，
否则只能靠第二个 name。**建议就用第二个 name `dynamo_sglang`**，简单、可回滚、不动 vLLM 路径。

代价：verl core 里几处 `== "sglang"` 的分支会 miss
（`engine_workers.py:774`、`checkpoint_engine/base.py:326` 的 `delta_sharded` 门禁、
`workers/config/rollout.py:318/339`）。现有的 `dynamo` name 已经有同样的问题，
所以不是新增风险，但 D4 提到的 sleep_level=1 那条要在 recipe 侧显式覆盖。

**⚠️ 新增代价（#136 带出来的）**：`verl/checkpoint_engine/base.py:326` 硬门禁
`if backend == "delta_sharded" and self.rollout_config.name != "sglang": raise NotImplementedError`。
用 `dynamo_sglang` 这个名字会被这条拒掉，而 `delta_sharded` 恰恰是 **sglang-only 的独门能力**
（走 sglang custom-weight-loader hook，`wire_format="delta_flush"`，verl 的 sglang ServerAdapter
`sglang_rollout.py:384` 已实现 `_update_weights_delta`）。也就是说：加了 sglang 却拿不到 delta 同步，
有点亏。三个处理方式，建议 (a)：
- (a) M2 阶段先不管，M4 给 verl 上游提 PR：把 name 判断换成能力判断
  （例如 `getattr(server_adapter, "supports_delta_flush", False)`）；
- (b) `register.py` 里 monkeypatch 掉这个检查（快，但脏，升级 verl 会静默失效）；
- (c) rollout name 就叫 `sglang`，靠 `engine_kwargs.dynamo.*` 区分 —— 会和原生 sglang backend 撞名，**不推荐**。

### D6 — checkpoint-engine backend（第 1 跳）：单机 naive，多机直接上 nixl

`checkpoint_engine.backend` 和 `engine=sglang|vllm` 是**正交**的两个轴，组合矩阵：

| 第1跳 backend | vLLM（#110 + #136 现状） | SGLang（本设计） | 备注 |
|---|---|---|---|
| `naive` | ✅ 单机 / 多机 | M2 目标 | 不起 CE actor，trainer 直接推 |
| `nixl` | ✅ #136 验过 2×8×H100 | M4 目标 | 跨机 RDMA；**LIBFABRIC 比 UCX 快 200×** |
| `nccl` | 未验 | 未验 | — |
| `delta_sharded` | ❌ 被 `base.py:326` 拒 | ⚠️ 被名字拒，见 D5 | sglang-only 能力 |

结论与顺序：
1. **M2 用 `naive`**：不起 CE actor，路径最短，先把第 2 跳（sglang 权重协议 + rank 映射）调通。
   #136 里"只在 non-naive 时才起 CE actor（避免 IPC socket 命名空间冲突）"这条对我们更宽松——
   sglang 路径没有 `/tmp/rl-colocate-zmq-*.sock`（走 HTTP），那类冲突原则上不存在，
   但**先按 #136 的保守做法来**，别在 M2 引入第二个变量。
2. **M4 多机直接上 `nixl`**，不要先做 naive 多机再迁。同时把 #136 的 `nixl_bench.py`
   在 dfw 的 fabric 上先跑一遍 —— 他们测出 UCX 0.23 GB/s 是因为 fabric 无 native RDMA-read，
   **dfw 是什么情况必须实测**，这个数直接决定要不要打 LIBFABRIC 的 verl 补丁。
3. **合并顺序建议：先 rebase #136，再做 sglang**。理由是 #136 动了
   `dynamo_async_server.py`（+175/−3）和 `dynamo_rollout.py`（+8），而 §6 的重构
   （抽 `_EngineBackend`、`_start_vllm_workers` → `_start_engine_workers`、
   `_vllm_processes` → `_engine_processes`）会和它**大面积冲突**。
   反过来先重构再 rebase #136 会更痛。

`release_kv_cache()` / `resume_kv_cache()` 是 #136 引入的必需接口
（`verl/workers/rollout/replica.py:285-291` 已在 `RolloutReplica` 基类里 fan-out 到
`server.release_kv_cache.remote()`，当前 recipe 里**没有实现**）。sglang 侧映射非常干净：

| verl 接口 | vLLM（#136 的做法） | SGLang（本设计） |
|---|---|---|
| `release_kv_cache()` | sleep(level=1) 凑 | `release_memory_occupation(tags=["kv_cache"])` |
| `resume_kv_cache()` | wake_up 凑 | `resume_memory_occupation(tags=["kv_cache"])` |
| `release()`（全量） | sleep(level=2) | `release_memory_occupation(tags=["kv_cache","weights"])` |

---

## 6. 文件级改动清单

### 新增

| 文件 | 内容 | 规模 |
|------|------|------|
| `recipe/dynamo/dynamo_sglang_rollout.py` | `SGLangServerAdapter(verl sglang ServerAdapter)`：`update_weights` 走桶 + `/engine/control/update_weights_from_tensor`；`resume`/`release` 走 memory_occupation；控制 RPC 打到共享 Ray actor | ~250 行 |
| `recipe/dynamo/dynamo_sglang_engine.py` | 把 `/engine/control/*` 包装成 verl `EngineBase` 形状的 adapter（对齐 `AsyncHttpServerAdapter` 的方法名），这样 `sgl_update_weights` 能直接吃 | ~200 行 |
| `recipe/dynamo/config/dynamo_sglang_trainer.yaml` | `rollout.name=dynamo_sglang`, `mode=async` | ~45 行 |
| `recipe/dynamo/smoke_dynamo_sglang.sh` | 生成-only smoke（对齐 `smoke_dynamo_v1.sh`） | ~55 行 |
| `recipe/dynamo/dynamo_sglang_stream.py`（M3） | `SGLangStreamAccumulator` 等价物 + request 级 abort | ~300 行 |

### 修改

| 文件 | 改动 |
|------|------|
| `dynamo_async_server.py` | 抽出 `_EngineBackend` 抽象：`build_cmd()` / `env_overrides()` / `control_call(route, body)` / `sleep()` / `wake()` / `clear_kv_cache()` / `release_kv_cache()` / `resume_kv_cache()`。`VllmEngineBackend` 保留现状（ZMQ sidecar），`SglangEngineBackend` 走 HTTP。`_start_vllm_workers` 更名 `_start_engine_workers`，`_vllm_processes` → `_engine_processes`（注意 `_WATCHDOG_SPECS`、`__getstate__` 里的名字，`:55-58`、`:2026`）。**与 #136 的 +175 行重度冲突，先 rebase #136** |
| `register.py` | 加 `dynamo_sglang` 两处注册；确认 `VERL_USE_EXTERNAL_MODULES` 能传到 `_DynamoCheckpointEngineWorker` 的 actor 里（CE worker 自己会 `get_rollout_class(rollout_config.name, mode)`，registry 没注册就直接 assert 失败） |
| `dynamo_agent_loop.py` | 大概率无改动（它只跟 frontend 说话）；M3 切 streaming 时才动 |
| `dynamo_thunderagent.py` | ThunderAgent router 是后端无关的（转发到 `<ns>.backend.generate`），预期只需确认 sglang worker 注册的 endpoint 名一致 |
| `README.md` | 加 sglang 段落 + 参数映射表 |

---

## 7. CLI 参数映射（`_build_sglang_cmd` 要写的表）

| verl `rollout.*` | vLLM CLI（现有） | sglang CLI |
|---|---|---|
| `tensor_model_parallel_size` | `--tensor-parallel-size` | `--tp-size` |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` |
| `max_model_len` | `--max-model-len` | `--context-length` |
| `max_num_seqs` | `--max-num-seqs` | `--max-running-requests` |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | 默认开；关用 `--chunked-prefill-size -1` |
| `enable_prefix_caching` | `--enable-prefix-caching` | 默认开；关用 `--disable-radix-cache` |
| `enforce_eager` | `--enforce-eager` | `--disable-cuda-graph` |
| `enable_sleep_mode` | `--enable-sleep-mode` | `--enable-memory-saver` |
| `dtype` | `--dtype` | `--dtype` |
| `trust_remote_code` | `--trust-remote-code` | `--trust-remote-code` |
| KV router block size | `--block-size` | `--page-size`（ThunderAgent 的 `router_block_size` 要跟着改这个） |
| KV events | `--kv-events-config <json>` | **不需要**，`publisher.py` 原生转发 |
| executor backend | `--distributed-executor-backend uni/mp` | 无对应，sglang 自己管 TP 进程 |

`dynamo.sglang` 的 `args.py:338` 是 `ServerArgs.add_cli_args(...)` 全量透传，
所以 `engine_kwargs.dynamo.extra_args` 依然是万能逃生口。

---

## 8. 分阶段落地

| 阶段 | 目标 | 验收 |
|---|---|---|
| **M-1** rebase #136（0.5-1d） | 把 verl-recipe#136 合进 `sopy/dynamo_next`，vLLM 路径回归 | vLLM 单机 naive + 单机 nixl 各跑一遍 `run_nixl_smoke.sh`；**顺手在 dfw fabric 上跑 `nixl_bench.py`**，记下 UCX vs LIBFABRIC 带宽 |
| ~~**M0a** 环境~~ **✅ DONE** | 在现有 `verl_vllm024.dev2.sqsh` 的 SETUP_COMMAND 里条件安装本地 wheel 的 `[sglang]` extra（见风险 10；不需要新建镜像） | ✅ 2026-08-21 job 16212011：`sglang`/`dynamo.sglang`/`dynamo.vllm`/`vllm`/`verl` 五方 import 全过，26 单测全绿 |
| ~~**M0b** 探路~~ **✅ DONE** | 手工拉 etcd+nats+`python -m dynamo.sglang`+frontend，逐个 curl 控制面路由 | ✅ 2026-08-21 jobs 16214273 / 16215105（H100）：13 个 CLI flag 全部 ACCEPTS；`call_tokenizer_manager` / `release_memory_occupation` / `resume_memory_occupation` / `update_weight_version` 全 200；`control/flush_cache` 404 "Route not found"（反向验证成立）；frontend token-in 生成通过；`update_weights_from_tensor` 用真实参数 + base64 在**未打补丁**的 dynamo 上 `success:true` |
| ~~**M1** 生成打通~~ **✅ DONE** | `_build_sglang_cmd` + `_start_vllm_workers` 的 `is_sglang` 分派；`smoke_dynamo_sglang.sh STAGE=gen` | ✅ 2026-08-21 job 16222989：`PASS: Dynamo x SGLang generation smoke completed`，`SMOKE_RC=0`。verl 全程编排 etcd/nats/dynamo.sglang/frontend，`frontend healthy: 1/1 workers registered`，`sglang control plane OK on 1 shard(s)`，`weight sync done: rank=0 shard=0 buckets=2 step=0`，产出 validation metrics |
| ~~**M2** 权重同步~~ **✅ 流程 DONE / ⚠️ 正确性未验** | `SGLangServerAdapter.update_weights` + sleep/wake | ✅ 2026-08-21 job 16225185：`PASS: Dynamo x SGLang 2-step training smoke completed`。3 次权重同步（step 0/1/2，各 2 buckets），3 组 `released/resumed ['kv_cache']` 正确配对，`resume['weights'] skipped` 过滤生效，产出真实训练指标（grad_norm 14.14、update_weights 1.53s）。❌ **权重一致性仍未验证**：`get_weights_by_name` 是模型相关 API，`Qwen2ForCausalLM` 未实现，探针每次都 `verification inconclusive`。设计原定的 greedy 采样 vs trainer forward argmax 比对仍待做 |
| **M3** TITO streaming（3-5d） | 升 dynamo 到含 #11640；`DYN_SGLANG_ENABLE_GENERATE=1`；写 accumulator + abort | 与 M1 的 completions 路径对比：同 seed 下 token 序列一致；abort 后 partial prefix 保留 |
| **M4** 多机 + 打平 vLLM（1w+） | `backend=nixl` 多机（CE actor 路径）、ThunderAgent 兼容、KV offload、metrics sidecar、30B | 2×8 H100 上 sglang×nixl 端到端跑通；复现 README 里 vLLM 的对照表格式：ms/token + KV hit rate + 权重同步耗时，sglang×{naive,nixl} vs vLLM×{naive,nixl} vs 原生 sglang baseline |

---

## 9. 风险与未决

1. ~~**b64 缺口（D2）**~~ **不存在 —— 此前的判断是错的**（2026-08-21，GPU 实测 M0b/M0c 推翻）。
   `sglang.srt.utils.MultiprocessingSerializer.deserialize` 本身就吃 base64：
   ```python
   def deserialize(data):
       """data (bytes or str): The serialized data, optionally base64-encoded."""
       if isinstance(data, str):
           data = pybase64.b64decode(data, validate=True)
       return SafeUnpickler(io.BytesIO(data)).load()
   ```
   配合 `UpdateWeightsFromTensorReqInput.serialized_named_tensors: List[Union[str, bytes]]`
   的字段类型，以及 sglang `http_server.py:1259` 的 "Any binary data in the named tensors
   should be base64 encoded" —— **base64 就是 sglang 的契约**，dynamo 原样透传 JSON 是对的。
   未打补丁的 dynamo 1.3.0 + sglang 0.5.14 上 `{"success":true,"message":"Success"}`。
   `patches/dynamo_sglang_b64.patch` 已删除。

   **错在哪**：读了 dynamo handler（无 base64）+ verl 客户端（做了 base64 编码），就推断中间
   有缺口，唯独没读真正消费这个字段的 `deserialize`。教训是——**跨仓库的"接口不匹配"结论，
   必须读到真正消费数据的那一层为止**，只看两端会得出对称但错误的结论。
2. **rank 映射 silent bug（D2）** — 错了不报错，只是模型变差。M2 的一致性校验是硬性门禁，不是 nice-to-have。
3. **`nvext` 在 sglang handler 上的支持度（D3）—— 已发生，代价是一次 30B 双节点运行。**
   原文写的是"必须 fail-loud，不要让 fallback 静默吞掉"。这条预判是对的，但代码里
   `_fallback_token_ids()` 从来就没响过，于是我自己踩了自己写下的坑。

   实际形态：frontend 正常返回**文本**，只是没带 token id；`_extract_completion_token_ids`
   取不到，就静默替换成 **1 个 eos**。后果不是崩溃，而是——
   `response_length/mean` 恒为 1.0、`grad_norm` 为 0、reward 贴地板，**作业退出码 0、指标齐全**。
   我为此提了三个错误假设（Base-vs-Instruct、采样参数、agent_loop 截断），
   因为聚合指标里没有任何东西能区分"没生成"和"生成了但没记录"。

   决定性证据只能来自原始 dump（`trainer.rollout_data_dir`）：`output` 字段有
   **2384 字符的连贯推理**。dump 里的 output 是拿 response_ids 解码出来的，
   所以它一长，就说明 token id 这条路是通的，问题在长度记账。
   > 教训：指标是模型行为经过一整条记账链之后的投影。链子断了，指标不会变成 NaN，
   > 会变成一个**很整齐的常数**。"整齐得不可能"本身就是信号——1.0000 不是采样能采出来的数。
   > 排查顺序应该是先看一条原始样本，再看聚合指标，我反了。

   触发条件与修复：打开 `request_engine_data=true` + `request_completion_token_ids=true`
   （已进 `retool_ab.sbatch`）。同时把 fallback 改成响亮的：首 3 次 + 每 100 次打一条
   ERROR，并在日志里直接点名要设的 flag。回归测试
   `test_missing_token_ids_is_loud_not_silent` / `test_fallback_keeps_logging_on_a_long_run`。
   注意这个 fallback 是 vLLM 路径就有的既有行为，**vLLM 侧同样会静默中招**。
4. **dynamo 版本跳跃** — 当前 pin 的 `94accc7389` 是为 ThunderAgent PR #11185 选的。
   升到含 #11640 的版本可能带崩 ThunderAgent 路径。M3 之前先在 vLLM 路径上验一遍新 commit。
5. **`--enable-memory-saver` 的显存代价** — torch_memory_saver 会改变分配器行为，
   colocate 下和 trainer 抢显存的表现可能和 vLLM sleep mode 不同，M2 要单独量一次峰值。
6. **PD 分离** — verl 有 `SGLangPDReplica`，dynamo 也有 `_disagg.py` + prefill_router。
   两套 PD 编排在 colocate RL 下怎么共存，本设计**不覆盖**，留到 M4 之后单独设计。
7. **#136 是 open PR，可能还会改** — 我们要在它上面做重构（§6 与它 +175 行重叠）。
   如果它在 review 中被大改，我们的 rebase 成本会重复付。M-1 之前先看一眼 PR 上的 review 状态；
   如果分歧大，考虑只 cherry-pick `release_kv_cache/resume_kv_cache` 这两个方法（sglang 必需），
   把 NIXL 那部分推迟到 M4。
8. **`delta_sharded` 被 rollout name 拒**（D5）— 加了 sglang 却拿不到 sglang 独有的 delta 同步。
   不阻塞 M2，但要在 M4 前给 verl 上游提能力判断的 PR，否则这块能力永久拿不到。
9. **CE worker 里的 registry 注册** — `_DynamoCheckpointEngineWorker` 是独立 Ray actor，
   自己调 `get_rollout_class(rollout_config.name, mode)`。`VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register`
   必须在那个 actor 的 runtime env 里生效，否则 `dynamo_sglang` 未注册直接 assert。
   M4 上 nixl 时第一个会踩的坑。
10. **容器：不用新建镜像；在现有 `verl_vllm024.dev2.sqsh` 的 SETUP_COMMAND 里条件安装即可。
    但绝不能把 sglang 烘进镜像。**（2026-08-21 实测，jobs 16196357 / 16210555 / 16212011）

    **已验证可行**（job 16212011，真装非 dry-run）：
    ```
    sglang OK   dynamo.sglang OK   dynamo.vllm OK   vllm STILL OK 0.24.0   verl OK
    torch 2.11.0+cu130 未变；transformers 5.5.3→5.8.1；numpy 1.26.4→2.3.5
    torch_memory_saver 0.0.9.post1 随 [sglang] extra 自动带入（--enable-memory-saver 不需额外处理）
    python -m dynamo.sglang --help 正常；register_engine_routes 存在于装出的 wheel 中
    recipe 的 26 个单测在该环境下仍全绿
    ```
    安装命令（`$W = $B/dynamo_wheels_1.3.0_94accc7389d4`）：
    ```bash
    pip install "$W/ai_dynamo-1.3.0-py3-none-any.whl[sglang]" \
                "$W/ai_dynamo_runtime-1.3.0-cp310-abi3-manylinux_2_39_x86_64.whl"
    ```
    `ai_dynamo==1.3.0` **不在 PyPI 上**（只有 `1.3.0.post1`/`1.3.1`/dev），必须走本地 wheel，
    否则 pip 直接 "No matching distribution"。镜像里 `dynamo` 本来就没装 —— 现有 vLLM 路径
    也是作业启动时装本地 wheel 的，sglang 只是同一处多一个 extra。

    **⚠️ 代价：装 sglang 会破坏同容器内 vLLM 的依赖**（pip 明确报告）：
    | 包 | 装后 | vllm 0.24.0 要求 |
    |---|---|---|
    | `llguidance` | 0.7.30 | `>=1.7.0,<1.8.0` |
    | `outlines_core` | 0.1.26 | `==0.2.14` |
    | `opencv-python-headless` | 4.10.0.84 | `>=4.13.0` |
    | `tilelang` | 0.1.8 | `==0.1.9` |
    | `tokenspeed-mla` | 0.1.7 | `==0.1.2` |

    `import vllm` 仍然成功，所以**这类破坏不会在 import 期暴露**：`llguidance`/`outlines_core`/
    `xgrammar` 是 vLLM 的 guided-decoding 栈，`tilelang`/`tokenspeed-mla` 是 kernel 包，
    只会在跑到那条路径时才炸。
    **解法**：容器是每个 slurm 作业从 `.sqsh` 全新拉起的，装包在 SETUP_COMMAND 里发生，
    所以只要**按 `engine` 条件安装**（sglang 作业装、vLLM 作业不装），两边天然隔离。
    反过来说：**不要把 sglang 预装进 .sqsh**，那会让所有 vLLM 作业带着坏掉的 guided-decoding 栈跑。
    做 vLLM↔sglang A/B 对比时，两侧必须是各自独立的作业，不能同容器切换。

    顺带一个副作用：sglang 强推的 `transformers 5.8.1` 和 `numpy 2.3.5` 恰好修掉了镜像里两个
    **既存**的未满足约束（`megatron-bridge` 要 `transformers>=5.8.1`、`opencv` 要 `numpy>=2`）。
11. **dfw 部分 CPU 节点没有 `/raid`**，而 enroot 默认数据路径是 `/raid/enroot`，pyxis 会以
    `failed to create container filesystem` 直接失败（实测 `cpu1-00012` 挂、`cpu1-00129` 正常）。
    在 sbatch 里 export `ENROOT_DATA_PATH` 等**无效**（pyxis 不采纳作业环境里的这些变量）。
    可行的绕法是 `--exclude=<坏节点>` 或指定已知可用节点。GPU 分区上是否同样有这个问题未测。
12. **core verl HEAD 的 V1 trainer 与 agent_loop 不兼容 —— 影响整个 recipe，不只 sglang**
    （2026-08-21 实测，M1 job 16221936）。当前 `verl_dynamo/verl` 在 `6cbca9ce`，而
    `recipe/dynamo/REQUIRED_VERL.txt` pin 的是 `d82d2777`（HEAD 更新）。在 HEAD 上：
    ```
    verl/trainer/ppo/v1/trainer_base.py:988   batch = tu.get_tensordict(...)   → TensorDict
      → verl/experimental/agent_loop/agent_loop.py:1210
            if "priority" not in prompts.non_tensor_batch:   ← DataProto 的 API
    AttributeError: 'TensorDict' object has no attribute 'non_tensor_batch'
    ```
    那行是 verl PR #6572（vLLM rollout determinism）加的，**晚于** recipe 的 pin。
    v0 trainer（`ppo/ray_trainer.py:609` 用 `DataProto.from_single_dict`）不受影响。
    → **必须走 `recipe/dynamo/main_dynamo*.py`**，它 import `verl.trainer.main_ppo_v0.TaskRunner`
    并强制传给 `run_ppo` —— 那个看起来多余的 import 就是干这个的。
    → 直接用 `python -m verl.trainer.main_ppo` 会选到 TaskRunnerV1 而踩雷；
    `$B/verl/recipe/dynamo/` 里那些跑通的脚本用 `main_ppo` 只是因为它们跑在更老的 core verl 上。
    → **vLLM dynamo 路径在这个 checkout 上同样会挂**，与 sglang 无关。

13. **权重同步的正确性仍无运行时防护**（M2 实测后仍未解决）。
    读回比对走不通：`get_weights_by_name` 由模型类实现，`Qwen2ForCausalLM` 没有它
    （`ERROR model_runner.get_weights_by_name: 'Qwen2ForCausalLM' object has no attribute ...`），
    探针只能降级为 warning。所以「rank→shard 映射错 → 训练与实际服务的策略静默发散」这条
    **头号风险目前只有 CPU 侧不变量测试兜底**。
    仍需实现设计原定的方案：同步后对固定 prompt 做 greedy 采样，与 trainer 侧 forward 的 argmax 逐 token 比对。
    经验教训：验证探针本身不能成为故障源 —— 第一版探 `get_internal_state` 返回 `CudaGraphConfig`
    导致 HTTP 500 直接拖垮整个 run；加了 try/except 降级后，后两次 API 不兼容才只留下 warning。

14. **`release_memory_occupation` 会把 worker 从 discovery 注销**，因此 release/resume 的状态
    必须由引擎的**唯一所有者**（`DynamoHttpServer` 节点 actor）持有。曾把它记在 per-rank adapter 上，
    结果 actor 释放并注销、adapter 不知情而跳过 resume，frontend 永久返回
    `503 Model ... is not ready to serve requests yet`。现由 `sglang_release`/`sglang_resume` 统一owner。

15. **dfw fabric 的 NIXL 传输选型未知** — #136 在他们的 fabric 上测出
    UCX 0.23 GB/s vs LIBFABRIC 48.5 GB/s（因为无 native RDMA-read）。dfw 是哪种未测。
    M-1 跑 `nixl_bench.py` 拿到数之前，不要假设 UCX 够用。
