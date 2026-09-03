> **文件改名对照（2026-09-03）**：`smoke_dynamo_v1.sh`→`smoke_vllm_generate.sh`，`train_30b_dynamo_sglang_4n.sh`→`train_qwen3_30b_sglang.sh`，`train_30b_sglang_native_i100.sh`→`baseline_qwen3_30b_sglang_native.sh`，`retool_ab_bootstrap.sh`→`container_bootstrap.sh`。下文沿用旧名。

# 交接：verl × Dynamo × SGLang（retool 30B RL）

最后更新：2026-08-24。上一轮会话的全部上下文压缩在此。

---

## 1. 目标

把 SGLang 作为引擎接进 verl 的 Dynamo rollout recipe，然后跑
**retool 任务 100 步，与 sglang 基线对比**（吞吐、step 时间、KV 命中率）。

当前卡在"端到端跑通 2 步"这一关，**100 步对比尚未开始**。

---

## 2. 环境与路径

```
登录        ssh <user>@<dfw login FQDN>（见本地 server.txt，不入库）
            密码登录，用 cc_project/rany.sh（expect 封装）；短主机名会假死，必须用 FQDN
$B          /lustre/fsw/portfolios/coreai/users/sopyang     （= fs1 路径的符号链接）
$W          /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/sopyang
```

| 用途 | 路径 |
|---|---|
| 本次开发的 verl | `$B/verl_dynamo/verl`，recipe 子模块分支 `sopy/dynamo_sglang` |
| **已跑通的参照脚本** | `$B/verl/recipe/dynamo/train_30b_rl_dynamo_kv_i100_metrics.sh` |
| 另一份参照（flexkv） | `$B/verl/recipe/dynamo/train_30b_rl_dynamo_kv_offload_common.sh` |
| dynamo 源码 overlay | `$B/dynamo` @ `94accc7389` |
| dynamo wheels | `$B/dynamo_wheels_1.3.0_94accc7389d4/`（py3-none-any + cp310-abi3，可移植） |
| 模型 | `$B/hf_models/Qwen3-30B-A3B-Base` |

推文件：`rput.sh` 超过 ~10KB 会失败，用 `split -b 6000` 分块 base64 append 再解码（见历史命令）。

---

## 3. 两个主脚本

### 3.1 `recipe/dynamo/train_30b_rl_dynamo_sglang_i100.sh`（**当前主线**）

从已验证脚本派生的 Dynamo+SGLang 脚本，2 节点 ×8 H100。

```bash
sbatch --job-name=sgl_i100 --output=$B/logs/sglang_i100_%j.out \
  --export=ALL,TOTAL_STEPS=2,TEST_FREQ=0 train_30b_rl_dynamo_sglang_i100.sh
```

镜像 `$B/images/verl_sgl0512.dev4.sqsh`（35 GB，官方 `verlai/verl:sgl0512.dev4`，已导入）。
镜像自带：**sglang 0.5.12 / torch 2.11.0+cu130 / transformers 5.3.0**，不带 vLLM。

### 3.2 `recipe/dynamo/retool_ab.sbatch`（三 arm A/B，旧主线）

`ARM=dynamo_sglang | dynamo_vllm | native_sglang`，镜像 `verl_vllm024.dev2.sqsh`。
用于分离"是 sglang 特有问题还是共有问题"。**dynamo_vllm arm 是关键对照组。**

---

## 4. 已修复的真 bug（都有回归测试，共 50 个单测）

| # | 问题 | 根因 | 位置 |
|---|---|---|---|
| 1 | `response_length` 恒为 1，指标全绿但没学到东西 | frontend 不回 token id 时 `_fallback_token_ids()` **静默**替换成 1 个 EOS | `dynamo_async_server.py`，已改为响亮报警 |
| 2 | 训练期引擎不交还 31 GB 权重 → OOM | 把 vLLM 的 `sleep(level=1)` 按字面映射成 sglang 的 `tags=["kv_cache"]`，但 vLLM level 1 本来就把权重挪到 CPU | `sleep()` 改为释放两个 tag |
| 3 | 引擎重复 release → 池损坏 | `sglang_release/resume` 的"检查—执行"跨了 `await`，4 个 shard adapter 并发全部通过检查 | 加 `asyncio.Lock` |
| 4 | `Failed to fold completions stream` ×257 | dynamo 在**第一次** resume 时就把 worker 注册回路由池，而 verl 首次只 resume `weights`，KV 池仍是释放态 | `sglang_resume` 改为恢复全部已释放 tag |
| 5 | 无 vLLM 镜像上整个 dynamo 路径无法 import | 门面 `dynamo_rollout.py` 顶层硬导入 vLLM 实现 | 改为容错导入 + 实例化时才响亮失败 |

---

## 5. 环境层面的硬约束（踩过才知道的）

1. **sglang 硬钉 transformers 5.x**（0.5.14 要 `==5.8.1`，官方镜像是 5.3.0），
   而两份已跑通的参照脚本都显式钉 `transformers>=4.56,<5`（实测 4.57.6）。
   **这是整条线的根因，详见 §6。**
   ⚠️ 上一版这里写的是"必须加 `_experts_implementation=eager`" —— **已证伪，不要照做**。
   eager 会让 `update_actor` 从 71.9s 变成 2810s（39×），而它想省的显存用
   `use_fused_kernels` 省得更彻底。默认 `EAGER_EXPERTS=0`。

2. **`expandable_segments:True` 与 sglang 的 `torch_memory_saver` 不兼容**
   （`TorchMemorySaver is disabled ... expandable_segments is not supported yet`）。
   已验证的 vLLM 脚本设了它；sglang 路径**必须 unset**，否则引擎起不来。

3. **官方 verl 镜像是 PEP 668 externally-managed**，需 `PIP_BREAK_SYSTEM_PACKAGES=1`。

4. **`--no-deps` 装 dynamo wheel 会缺 `blake3`**（sglang handler 的 multimodal 分支要）。

5. **`ppo_max_token_len_per_gpu` 会被 `ulysses_sequence_parallel_size` 乘**
   （`verl/workers/engine/utils.py:111`）。已验证脚本是 `18432 × sp4 = 73728`。
   而 **18432 是下界**（= `max_prompt_length 2048 + max_response_length 16384`，一条完整序列），
   调到下界以下会触发 `AssertionError: max_token_len must be greater than the sequence length`。
   ref 侧 `log_prob_max_token_len_per_gpu=73728` **不被乘**（ref 的 sp=1）。

6. `pkill -f` 的模式里 `.` 是通配符；脚本路径若含该子串会**自杀**（退出码 15）。
   已验证脚本对此有注释，我起了 `run_dynamo_sglang_*` 目录名后又踩了一次。

7. enroot 导入镜像：**temp 必须放本地盘**（lustre 不允许 mknod 建 whiteout），
   且**要在计算节点做**（登录节点 `ulimit -u=300`、内存不足，mksquashfs 会 OOM）。
   参考 `$B/logs/import_sgl.sbatch`。

8. dfw 通用：`unset ROCR_VISIBLE_DEVICES`（否则 torch 初始化失败，只以 ActorDiedError 出现）。

---

## 6. 根因：transformers 4.57.6 → 5.x 的专家融合布局

**这是整条线的总开关，之前几版交接把它当成了"约束"，其实它是"根因"。**

transformers 在 **4.58.0** 把 MoE 专家从 `nn.ModuleList` of `nn.Linear` 换成了单个 3D 融合张量
`[num_experts, in_features, out_features]`（见 `qwen3_moe.py` 里的 `self.gate_up_proj[expert_idx]`）。
**已跑通的 vLLM 参照脚本显式 pin `transformers>=4.56,<5`，实测装的是 4.57.6 —— 融合布局前的最后一个版本。**
而 sglang 0.5.x 硬钉 transformers 5.x，装不回去。

### 6.1 实测对照（token 量、response 长度都在同一量级：~300k / ~900）

| 跑 | transformers | 专家实现 | update_actor | step | 峰值显存 |
|---|---|---|---|---|---|
| 参照 vLLM 13560953/1417xxxx | **4.57.6** | `nn.ModuleList` | **22–42s** | 339–644s | **35.6 GB** |
| sg6 16442420 | 5.8.1 | 融合 grouped_mm | 52s | 819s | 72.1 GB |
| tok 16517408 | 5.3.0 | 融合 + **eager** | **2810s** | 3527s | 71.7 GB |
| fast 16545952 | 5.3.0 | 融合 grouped_mm | **71.9s** | **403s** | 71.6 GB |

两个独立的代价，之前混为一谈了：

1. **显存 2×（35.6 → 72 GB）**，来自融合布局本身。**无解**，除非放弃 sglang。
2. **eager 的 39× 速度税**。显存翻倍逼出了 `_experts_implementation=eager` 这个 mitigation，
   而它强制走 128 专家的 Python 循环。同一脚本同一镜像直接对照：**2810s → 71.9s**。
   → **默认必须关掉 eager**；省显存改用 `use_fused_kernels`（见 6.2）。

⚠️ 36 GB 的差距归给 transformers **没有完全隔离** —— 参照跑同时用的是旧的 `$B/verl` 树。
sg6（5.8.1 + 新 verl = 72 GB）把主因指向 transformers，但严格对照需要"新 verl + 4.57.6"，
而那个组合装不了 sglang。只能在 vLLM backend 上做这个对照。

### 6.2 有效的三个杠杆

| 杠杆 | 作用 | 证据 |
|---|---|---|
| **关掉 eager**（`EAGER_EXPERTS=0`，已是默认） | update_actor 2810s→71.9s | 16545952 |
| **`use_fused_kernels=True` + `impl_backend=torch`** | 干掉 `[T,151936]` logits 及其 fp32 upcast；OOM 从 forward 推到 backward | 16513027 vs 16515269 |
| **`PPO_MAX_TOKEN_LEN=4608`** | 唯一跑完 2 步的设置 | 16517408 |

### 6.3 已被证伪 / 需修正的结论

1. **「峰值不是激活主导」——错。** 那是拿 vLLM(4608) 和 sglang(18432) 两个**引擎和 transformers 版本都不同**的跑
   比出来的，混淆变量。降 token 预算确实有效（16517408 靠它跑完 2 步）。
2. **「step-2 OOM 是 Adam 挤进 fwd/bwd」——不准确。** 16545952 的探针证明 step 2 死在 forward，
   **根本没走到 `optimizer_step()`**，那时 Adam 已在 CPU。真实情况是配置卡在天花板：
   step1 峰值 71.62 GB，step2 只多要 480 MiB 就没了，两步之间**没有累积**。
   但补丁并非无用：Adam fp32 ≈ 15 GB（2×4B×30.5B/16），默认常驻的话 step2 峰值 ≈ 86 GB 必死。
   **补丁把"必然 OOM"压成"边缘 OOM"，必要但不充分。**
3. **`optimizer_offload` 调用链本来就是通的**，不是 bug（上一版怀疑错了）。
4. **「跑到 step:N」不等于跑通**：`retool_rt_dyn2_16255152` 是 `response_length=1.0` 的退化跑。

### 6.4 GitHub 证据状态（2026-08-25 查）

**没有任何上游 issue 报告过"transformers 5.x 让 MoE bf16 训练显存翻倍"。** 找到的是：
- 布局变更本身：[transformers#43472](https://github.com/huggingface/transformers/issues/43472)
  （原话："While this improve speed of Moe, it is making downstream library difficult to adapt"）
- 版本边界 ≤4.57.x / ≥4.58.x：[SkyRL#1297](https://github.com/NovaSky-AI/SkyRL/issues/1297)
- 下游破坏都在**量化和 PEFT**，不是显存：[bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)、
  [transformers#42491](https://github.com/huggingface/transformers/issues/42491)、[peft#3009](https://github.com/huggingface/peft/issues/3009)

⚠️ **bnb#1849 里的 55.60 GB vs 15.09 GB 不能当证据引用** —— 那是量化没生效（4-bit vs 16-bit），
不是显存回归。我们的 2× 是**自己的实测**，上游未知。

⚠️ **eager 路径还有一个已知正确性隐患**：SkyRL#1297 指出选择性专家循环（`expert_hit = ...nonzero()`）
在 non-reentrant gradient checkpointing 下会触发 `check_recomputed_tensors_match` 断言失败，
而 verl 正是用 `use_reentrant: False`。我们没撞上是因为先 OOM 了。又一个关掉 eager 的理由。

---

## 7. 脚本的 env 开关 + 当前状态

`train_30b_rl_dynamo_sglang_i100.sh` 现在全部通过环境变量控制，不用改脚本：

| 变量 | 默认 | 说明 |
|---|---|---|
| `EAGER_EXPERTS` | `0` | `1` 才加 `_experts_implementation=eager`。**保持 0**，见 6.1 |
| `USE_FUSED_KERNELS` / `FUSED_KERNELS_BACKEND` | `0` / `torch` | 注意 `impl_backend` 必须用 `++` 前缀（`+` 会 ConfigCompositionException） |
| `PPO_MAX_TOKEN_LEN` | `18432` | 下界受 `max_token_len = per_gpu × sp ≥ 18432` 约束 |
| `VERL_DEFER_OPTIMIZER_LOAD` | `0` | Adam 只在 `optimizer_step()` 期间上卡 |
| `VERL_MEM_DEBUG` | `0` | OOM 时 dump `memory_summary` + 分配栈 snapshot |

已改的 verl 代码（`$B/verl_dynamo/verl`，均为 env 门控，默认不改变行为）：
- `verl/workers/engine/fsdp/transformer_impl.py`：`EngineTrainModeCtx._context_switch` 进入 train 模式时
  不搬 Adam；`FSDPEngine.optimizer_step()` 前后 load/offload + 3 条探针。
  **探针必须用 `logger=None`** —— `logger=logger` 的那些 `log_gpu_memory_usage` 根本不进 stdout（踩过）。
- `verl/workers/engine/base.py`：`train_batch` 外包 try/except，`VERL_MEM_DEBUG=1` 时 dump 分配栈。

### 7.1 ✅ 已验证可用的配置（16547388，COMPLETED，3/3 步）

```bash
sbatch --job-name=sgl_f46 --output=$B/logs/sglang_f46_%j.out \
  --export=ALL,TOTAL_STEPS=3,TEST_FREQ=0,\
VERL_DEFER_OPTIMIZER_LOAD=1,USE_FUSED_KERNELS=1,FUSED_KERNELS_BACKEND=torch,\
EAGER_EXPERTS=0,PPO_MAX_TOKEN_LEN=4608 \
  train_30b_rl_dynamo_sglang_i100.sh
```

| step | update_actor | step | 峰值 | score | resp_len | grad_norm | entropy |
|---|---|---|---|---|---|---|---|
| 1 | 68.2s | 791.5s | 71.255 GB | −0.9215 | 1135.1 | 0.096 | 1.327 |
| 2 | 82.6s | 773.0s | 71.288 GB | −0.7266 | 891.2 | 0.207 | 1.334 |
| 3 | 76.3s | 764.3s | 71.293 GB | −0.8750 | 848.6 | 0.111 | 1.123 |

`OOM=0`，`NO TOKEN IDS=0`，**峰值三步几乎不动**（71.255→71.293）——步间无累积，
坐实了 §6.3 第 2 条：8192 那次 step-2 OOM 是卡天花板（差 480 MiB），不是有东西在涨。

**Adam 状态实测 = 14.22 GB**（估算 15.25 = 2×4B×30.5B/16，接近）：
```
step1: Before 14.28 → After 14.28 → offload 14.28   (Adam 未分配，load 是 no-op)
step2: Before 14.28 → After 28.50 → offload 14.28   (+14.22 GB，用完卸回)
step3: Before 14.28 → After 28.50 → offload 14.28   (复现)
```
`VERL_DEFER_OPTIMIZER_LOAD` 的 load→step→offload 循环每步都干净归位，无泄漏。
⚠️ **但"4608 下补丁是否必需"没做 ablation** —— 它确实省下 14.22 GB 常驻，
8192 那轮 forward 在 71.72 GB 差 480 MiB 而死、再加 14.22 GB 必死；4608 不带补丁能否活，未测。

### 7.2 ⚠️ 优先级已经变了：瓶颈是 rollout，不是训练

**`gen` 占 step 时间的 88%**（677–704s / 764–791s），`update_actor` 只占 ~10%。
`update_actor` 从 2810s 压到 ~76s（39×），但 step 只从 3527s 降到 ~775s。
**再优化训练侧收益很小。** 而 `gen` 正是 dynamo+sglang 要对比的对象：
参照 vLLM 的 `gen` 是 276–580s，本配置 677–704s —— 但两边 response 长度不同
（1135/891/849 vs 882–918），不能直接下结论，这恰恰是 100 步对比该测的东西。

### 7.3 离 100 步还差什么

- `~775s × 100 ≈ 21.5h`，超 4h 作业上限 → 需拆 ~6 段，**`save_freq` + resume 还没配**
  （当前 `save_freq=-1`，根本不存 checkpoint）。这是下一个工程活。
- **wandb 一直没开**：所有跑都是 `trainer.logger=["console"]`，因为没 export `WANDB_API_KEY`。
- **`experiment_name` 是写死常量** `e2_30b_dynamo_sglang_i100`，多 arm 会画进同一条曲线，
  需带上 `${SLURM_JOB_ID}` 或 arm 名（`default_local_dir` 已经带了，`experiment_name` 没带）。
- 对比结论必须写明：sglang arm 因被迫用 transformers 5.x，峰值显存是 vLLM baseline 的 2×
  （71.3 vs 35.6 GB），`ppo_max_token_len` 也不同（4608 vs 8192/18432）——
  这是栈的性质，不能悄悄用不同配置凑成"两边都能跑"。

---


## 8. A/B 结论：dynamo+sglang vs 原生 sglang（2026-08-27）

### 8.1 实验设计

两个脚本除 rollout 路径外**逐行相同**（同 `verl_sgl0512.dev4` 镜像、同 transformers 5.3.0、
4 节点 32 卡、`ULYSSES_SP=4` / `ROLLOUT_TP=2` / `PPO_MAX_TOKEN_LEN=4608` /
`max_response_length=16384`、同一套显存修复）：

| arm | 脚本 | rollout |
|---|---|---|
| dynamo + sglang | `train_30b_dynamo_sglang_4n.sh` | `rollout.name=dynamo` + `engine_kwargs.dynamo.*` |
| 原生 sglang | `train_30b_sglang_native_i100.sh` | `rollout.name=sglang`，`~agent_loop_manager_class` |

后者是从前者反向派生的（只改 rollout，其余不动），diff 只有 rollout 那一块。

### 8.2 rollout 吞吐：无可测量差异

**必须分层。** `timing_per_token_ms/gen` 与 `response_length/max` 的相关性 **r=0.88~0.95**
（复现了 `$B/verl/recipe/dynamo/dynamo_vllm_timing_20260620.md` 报告里的 r=0.93），
尾部长度是主导变量。而各 arm 的尾部饱和比例在 55%~71% 之间随机波动，
**直接比全体均值会被这个比例差污染**。

尾部饱和层（`max=16384`，条件严格对齐）：

| 配置 | n | mean | sd | se |
|---|---|---|---|---|
| dynamo, interval=1 | 12 | 2.6981 | 0.330 | 0.095 |
| native, interval=1 | 15 | 2.6383 | 0.305 | 0.079 |
| dynamo, interval=100 | 28 | 2.6982 | 0.255 | 0.048 |
| native, interval=100 | 22 | 2.7556 | 0.221 | 0.047 |

| 对比 | 差异 | t |
|---|---|---|
| dynamo vs native @ i=1 | dynamo 慢 2.3% | 0.48 |
| dynamo vs native @ i=100 | dynamo 快 2.1% | 0.85 |

**两个 interval 下 \|t\| 均 < 1，方向还相反 → 无显著差异。**

### 8.3 训练侧完全一致（符合实验设计）

| 配置 | n | update_actor | MFU | 峰值 GB |
|---|---|---|---|---|
| dyn_i1 | 21 | 21.37s | 0.01326 | 67.73 |
| nat_i1 | 21 | 22.39s | 0.01321 | 67.78 |
| dyn_i100 | 40 | 21.42s | 0.01330 | 67.73 |
| nat_i100 | 40 | 21.96s | 0.01248 | 66.02 |

⚠️ 提取 MFU 时 `perf/mfu/actor` 会误匹配 `perf/mfu/actor_infer`，必须用
`grep -o "perf/mfu/actor:[0-9.e+-]*"`（带冒号）。第一版统计出过 MFU=1.705 的脏数据。

### 8.4 `stream_interval` 实测无效

dynamo 侧尾部饱和层：interval **1 → 100，均值 2.6981 → 2.6982**（变化在小数点后四位）。
这是这批数据里最干净的一条。

两侧的注入方式（都已验证真的到达引擎，非"设了没生效"）：

```
dynamo: ++...engine_kwargs.dynamo.extra_args='["--stream-interval=100"]'
        → python3 -m dynamo.sglang ... --stream-interval=100      （命令行确认）
native: ++...engine_kwargs.sglang.stream_interval=100
        → ServerArgs stream_interval=100                          （日志确认）
```

⚠️ **`DYN_SGLANG_STREAM_INTERVAL` 在当前配置下完全无效**：它由 `SglangProcessor` 读取，
而那个 processor 需要 `--dyn-chat-processor sglang` 才实例化；
`frontend_args.py:444` 的默认值是 `dynamo`，我们的 frontend 命令行也没传这个参数。

### 8.5 代码分析：dynamo 的流式路径

- **流式是强制的**：`args.py:544` 无条件 `enable_disjoint_streaming_output()` →
  `_compat.py:118` 设 `incremental_streaming_output=True`；
  `decode_handler.py:390/460`、`prefill_handler.py:162` 全部硬编码 `stream=True`。
  整个 sglang 目录里 `stream=False` 只出现在一个单测里。
- **客户端的 `"stream": False` 只管 frontend→客户端那一段**
  （`dynamo_async_server.py:1496`，在 `_build_frontend_completion_payload()` 里）。
  它是**承重的**：响应侧用 `await resp.text()` 一次性读整个 body 再按单个 JSON 解析，
  没有任何 SSE 逐行解析路径，改成 `True` 会直接解析失败。
- **原生 sglang 不流式**：`http_server_engine.py:471/496` 的 payload 里没有 `stream` 字段
  （且 `None` 值会被过滤），走 sglang `/generate` 的非流式默认。

所以 dynamo-vs-native 的对照**同时也是流式-vs-非流式的对照**，而结论是无可测量差异
（|t|<1）。按噪声尺度反推，任何系统性流式开销的上界约 **< 0.14 ms/token**（相对 2.4 ms/token 的解码成本）。

### 8.6 ⚠️ 潜在的 O(N²)：logprob 累积重传

`common/backend/logprobs.py::extract_from_sglang_meta` 的注释和实现揭示了一个不对称：

> SGLang's `output_token_logprobs` / `output_top_logprobs` are **cumulative** across
> stream chunks even though `output_ids` is **disjoint**

每个 chunk 的 `meta_info` 携带**完整的累积 logprob 列表**，dynamo 切掉前面只留尾部。
interval=I 时总传输量 ≈ **N²/(2I)**：

| N | I=1 | I=100 | I=16384 |
|---|---|---|---|
| 980（平均） | 48 万条 | 4.8k | ~980（线性） |
| 16384（尾部） | **1.34 亿条** | 134 万 | 16384 |

**但实测 interval 1→100 效应为 0.0001**，所以这条路径在当前负载下**不构成瓶颈**。
可能的原因：`logprobs=0` 时 sglang 是否真的填充 `output_token_logprobs` **未经确认**。

verl 对 rollout logprob 有三个层次的用法：

| 用法 | 位置 | 必需性 |
|---|---|---|
| 诊断（`rollout_probs_diff_*`、`rollout_corr/*`） | `debug/metrics.py:83`、`rollout_corr_helper.py:1134` | 可关 |
| Bypass：`old_log_probs = rollout_log_probs` | `ray_trainer.py:1571`，开关 `actor.use_rollout_log_probs` | 可选 |
| **FullyAsync** | `fully_async_rollouter.py:733` 是 `assert` | **硬性要求** |

→ 若将来上 FullyAsync 或 Bypass 模式，`stream_interval` 需要重新评估
（那时它直接除掉二次项，参照脚本 `train_30b_rl_dynamo_kv_interval16384.sh` 取的就是
`interval = max_response_length`）。

### 8.7 ⚠️ 原生 sglang 的 NCCL bug（未解决）

`train_30b_sglang_native_i100.sh` 在**走 resume 路径时**稳定复现：

```
Load from checkpoint folder: .../global_step_20    ← resume 本身成功
Setting global step to 20
Training Progress: 40% | 20/50
→ ncclUnhandledCudaError (ProcessGroupNCCL.cpp:3780)
```

崩在 `SGLangHttpServer` 内部，崩前有 `Guessing device ID based on global rank` 警告。
50 步链的 seg2/seg3 都死在这里（约 26 分钟处），所以 **native arm 最多只能跑单个 4h job（~21 步）**。

⚠️ 我曾判断"换到 sgl0512 镜像绕开了这个 bug"——**那是基于 3 步冒烟没复现，说早了**。
短跑不触发，一进 resume 就必现。

### 8.8 dynamo 50 步完整跑通

`sgl_dynamo_50s`（16589433/34/35，三段 resume 接力）跑满 50 步，checkpoint 到 `global_step_51`。
`critic/score/mean` 轨迹：

```
step  1: -0.875    step 25: -0.461    step 48: -0.422
step 10: -0.836    step 30: -0.351    step 51: +0.008   ← 首次转正
step 20: -0.727    step 40: -0.445
```

准确率从 6.3% 涨到约 50%，`update_actor` 全程 20.6–21.8s，`OOM=0`。

⚠️ wandb 三段**没有合并成一条曲线**（三个 run id：`476soir2` / `mrdf5efm` / `4azwvm2b`）。
加的 `WANDB_RUN_ID` + `WANDB_RESUME=allow` 没生效，
大概率是 verl 的 wandb logger 自己调 `wandb.init()` 时显式传参覆盖了环境变量。

---

## 9. 排查方法论（血的教训）

- **先看原始样本，再看聚合指标。** `trainer.rollout_data_dir` dump 出的 `output` 是从
  `batch.batch["responses"]` 解码的，它一长就说明 token id 链路通、问题在记账。
  `response_length=1.0` 那次我从聚合指标猜了三个错误假设。
- **"整齐得不可能"的常数指标（1.0/0.0）是记账链断裂的信号**，不是采样结果。
- **跑通的邻近脚本是证据，不是模板。** 对齐参数集合不够，要问每个参数为什么在那里——
  `request_completion_token_ids` 和 `transformers<5` 都躺在我逐行比对过的脚本里。
- **跨仓库"接口不匹配"的结论，必须读到真正消费数据的那一层**（b64 那次的教训）。
- **交错日志不能推时序**：多节点 actor 的日志混在一起，判断"重复调用"前先带上 pid/ip。
- **`#SBATCH --error` 分流会让 stdout 看起来"什么都没输出"**，实际错误在 .err 里。
- **"某次运行跑到了 step:N" 本身不是证据**：先确认那一步的 `response_length/mean` 和
  `perf/total_num_tokens` 是真实量级。退化成 1-token 的运行能轻松跑完 N 步，
  用它当"能跑通"的反例会把整条排查引偏（`retool_rt_dyn2_16255152` 就是这样被我误读过一次）。
- **任何 mitigation 都要当场量它的代价。** `_experts_implementation=eager` 是从上一版交接直接沿用的
  "省显存"建议，我没量过它的速度代价就一路带着跑了四轮；实际它把 `update_actor` 从 71.9s 变成 2810s。
  第一次跑完就该看 `timing_s/update_actor` 和邻近跑对一下，而不是只盯着 OOM 有没有消失。
- **对照组必须只差一个变量。** 我曾用 vLLM(per_gpu=4608) 和 sglang(18432) 两个跑得出"峰值不是激活主导"，
  但那两个跑连 transformers 版本都不同 —— 结论是错的，白费了两轮。
- **性能对照必须先找出主导变量并分层。** `timing_per_token_ms/gen` 看似已按 token 归一化，
  但它与 `response_length/max` 的相关性高达 0.88~0.95 —— rollout 要等最慢的样本。
  我第一次只比全体均值就下了"无差异"的结论，方法上站不住；分层后才发现全体均值那一层
  恰恰是信号最弱的（比例混淆）。
- **小样本的"接近显著"多半会回归。** 尾部饱和层 n=14/10 时测出 dynamo 快 6.9%、t=1.95，
  我据此预测加倍样本后 t 会升到 2.8。实际加到 n=28/22 后，效应量掉到 2.1%、t=0.85。
  这是小样本偶然高估的典型形态 —— 效应量腰斩而非 t 值上升。**不要用 t≈2 的结果做外推。**
- **读到代码不等于那段代码在跑。** 我两次栽在这上面：先据一行 `"stream": False` 断言整条路非流式
  （实际它只管 frontend→客户端）；又据 `sglang_processor.py` 分析 frontend 聚合开销
  （实际那个 processor 需要 `--dyn-chat-processor sglang`，默认根本没实例化）。
  **确认实例化/调用，再谈它的成本。**
- **判断某类显存杠杆有没有用，先看峰值对该杠杆敏不敏感**：两次运行的 `ppo_max_token_len_per_gpu`
  差 4 倍而 `max_memory_allocated_gb` 只差 0.4 GB，就说明峰值根本不由激活主导，
  再调激活侧的任何参数都是浪费一轮 1.5 小时。

---

## 10. 续行链被注释截断，静默丢掉 17 个 hydra 参数（2026-08-31）

**这是 §9「跑到 step:N 不等于跑通」的对偶：一个跑**没到** step 1 就死，但死因和它报的错完全无关。**

### 10.1 症状

`train_30b_dynamo_sglang_4n.sh` 4 节点跑，在**第一次 `update_weights`** 就 OOM，
栈在 `get_per_tensor_param` → `FSDP.state_dict()` → `_create_chunk_dtensor` → `tensor.detach().clone()`：

```
torch.OutOfMemoryError: Tried to allocate 1.50 GiB.
  this process has 28.50 GiB in use.  Process <engine> has 49.09 GiB in use.
  79.11 GiB total, 1.46 GiB free.
```

三个作业（16694882 / 17014038 / 17015852）数字**逐字节相同** → 确定性，不是显存抖动。

### 10.2 一开始的三个错误假设（都被证伪）

1. **「引擎没交还显存」** —— 日志确实显示 `released ['kv_cache','weights']` 之后
   `resume['weights'] widened to ['kv_cache','weights']`，引擎在 update_weights 前就占回了 49 GB。
   看起来像是 §4 bug #4 那个 widening 修复的副作用。**但成功的跑时序完全一样**（16658934、16704764
   都是 released → widened → resumed），所以 widening 不是原因。
2. **「4 节点特有的显存布局问题」** —— 方向对了一半，但归因错了：不是布局，是 world size 配错。
3. **「参数没传进去」** —— 查 `FUSED_ARGS:` 的 echo，显示数组构造完全正确，
   于是排除了这条。**这一步排早了**：echo 对不代表它进了命令行。

### 10.3 真因

**把成功跑和失败跑实际执行的 `python3 -m recipe.dynamo.main_dynamo` 命令行做 token 级 diff**
（不是比脚本，是比日志里 `set -x` 打出来的真实命令行）：

```
success 16658934: 84 tokens
fail    17014038: 67 tokens        ← 少了 17 个，全部是 prometheus.file 之后的
```

丢掉的包括 `trainer.nnodes=4`、`trainer.total_training_steps`、`trainer.resume_mode`、
`trainer.logger`、`trainer.default_local_dir`、`use_fused_kernels`、`fused_kernel_options.impl_backend`。

脚本里：

```bash
    actor_rollout_ref.rollout.prometheus.file="$PROM_FILE" \
# stream_interval 实测无效，已撤除（2026-08-27）：dynamo 侧尾部饱和层     ← 注释插进了续行链
# ...三行
    trainer.logger="$TRAINER_LOGGER" \
    ...
    "${FUSED_ARGS[@]}"
```

行尾 `\` 把下一行拼上来，拼进来的是 `#`，于是**命令在这里结束**，其余 17 个参数变成
一条独立的、注定 command-not-found 的语句。`set -e` 没开，脚本继续往下跑，没有任何报错。

**`trainer.nnodes` 丢失 → verl 用默认值（1）→ FSDP 按 8 卡而不是 32 卡分片
→ 每卡分片大 4 倍 → state_dict() 的 unshard 撞上引擎的 49 GB。**
OOM 只是最后那 40 MiB 的表象。

实测坐实（同一行日志 `After FSDP, memory allocated`）：

| 跑 | nnodes | FSDP allocated | reserved |
|---|---|---|---|
| 16658934（成功） | 4（=32 卡） | **3.55 GB** | 11.71 GB |
| 17014038（OOM） | 丢失→1（=8 卡） | **14.25 GB** | 22.22 GB |

14.25 / 3.55 = **4.01** —— 与 world size 32→8 的预测严丝合缝。
这一个数字比任何显存推理都直接：**怀疑 world size 配错时，先看 `After FSDP` 那行，
不要去看峰值。** 峰值受激活、Adam、引擎多方影响，分片大小只受 world size 影响。

### 10.4 时间线（每条证据都对得上）

| 时间 (PDT) | 事件 |
|---|---|
| 08-26 09:33 | 16658934 / 16658935（A/B 双 arm，4 节点）启动 → **COMPLETED** |
| 08-26 17:30 | 注释加进 `train_30b_dynamo_sglang_4n.sh` 和 `train_30b_sglang_native_i100.sh` |
| 08-26 21:19 | 16694882（4 节点）→ **FAILED**，OOM |
| 08-26 22:36 | 16704764（2 节点，走 `_i100.sh`，该脚本无此注释）→ COMPLETED |
| 08-31 00:47 | 17014038 → **FAILED**，与 16694882 逐字节相同的 OOM |

✅ **A/B 结论（§8）不受影响**：两个 arm 都在 08-26 09:33 启动，早于注释引入。

### 10.5 已修

两个脚本的注释都移到 `python3` 调用之前（`.bak_contfix_0831` 留底）。
写了个扫描器，recipe/dynamo 下 12 个脚本现在 `TOTAL_BROKEN=0`；
verl 主仓 examples 里还有 5 处同类写法（多为「注释掉一行参数」，未修，非本次路径）。

### 10.6 方法论

- **echo 出来对 ≠ 进了命令行。** 验证参数要看 `set -x` 打出的**那一条真实命令行**，
  不是看构造它的中间变量。我因为 `FUSED_ARGS:` 的 echo 正确，一度把「参数丢失」排除掉了。
- **对照组要比「实际执行的东西」，不是「应该执行的东西」。** 决定性的一步是把两条命令行
  做 token 级 diff，而不是继续比脚本、比配置 dump、比显存数字。
- **确定性的失败（逐字节相同的 OOM）指向配置/代码，不指向资源。** 显存不足的抖动不会连
  小数点后两位都一样。三个作业相同数字时就该停止调显存参数，去查配置链路。
- **`bash -n` 查不出这个 bug** —— 语法完全合法。要靠「续行后接注释」的模式扫描。

---

## 11. vLLM 路径：`VERL_DEFER_OPTIMIZER_LOAD` 从来没生效过（2026-08-31）

### 11.1 「已跑通的基线」16718419 其实没跑通

`validate_vllm_kv_metrics.sbatch` 的注释里写着「The proven COMPLETED run 16718419」——
**这是错的，我据此写了注释又据此推理，绕了一圈。**

`sacct` 的 `COMPLETED` 只反映 sbatch 外壳脚本的退出码，而外壳最后一行是 `echo`，永远成功。
真实情况：

```
16718419:  step:1 只有一个   OutOfMemoryError × 15   KVVAL_DONE rc=1
17015763:  step:1 只有一个   OutOfMemoryError × 16   KVVAL_DONE rc=1
```

**两者行为完全一致** —— vLLM 这条路从来没跑完过 3 步。
§9 那条「跑到 step:N 不等于跑通」要再加一句：**`sacct COMPLETED` 更不等于跑通**，
它连 step 数都不看。判据是脚本自己埋的 `KVVAL_DONE rc=`，不是 slurm 状态。

### 11.2 真因：`ray start` 早于 `export`

`sbatch` 生成 head.sh 的顺序曾经是：

```bash
ray start --head ...        # ← raylet 在这里启动
export RAY_ADDRESS=...
export VERL_DEFER_OPTIMIZER_LOAD=1   # ← 太晚了
bash recipe/dynamo/train_30b_rl_dynamo_kv_metrics.sh ...
```

**Ray worker 由 raylet fork，继承的是 raylet 的环境**，不是之后那行 export 的。
而 worker.sh 里压根没有这个 export。所以补丁在两类节点上都没装载。

`transformer_impl.py` 的三条探针（`Before/After deferred load_fsdp_optimizer`）
在 17015763 的日志里 **一条都没有** —— 这就是最干净的证据。

OOM 栈也对得上：`forward_backward_batch → backward → _engine_run_backward`，
step 2 的 backward，正是 Adam 常驻卡上时会炸的地方（§6.3 第 2 条）。

**已修**：`COMMON` 块移到 `ray start` 之前，且同时写进 worker.sh 和 head.sh。

### 11.3 验收判据（下次别再靠日志"看起来正常"）

| 想确认的事 | grep 什么 | 不是什么 |
|---|---|---|
| 作业真的跑完 | `KVVAL_DONE rc=0` | `sacct COMPLETED` |
| 跑了几步 | `step:N` 的最大值 | 日志长度 / 耗时 |
| 生成没退化 | `response_length/mean` 是百量级 | `step:N` 出现过 |
| 权重同步正确 | `rollout_actor_probs_pearson_corr` ≈ 0.999 | 没报错 |
| defer 补丁生效 | `Before deferred load_fsdp_optimizer` 出现 | env 里 export 了 |

17015763 的 step 1 按这套判据是**真实且健康的**：
`response_length/mean=1081.5`、`global_seqlen/mean=89791`、
`rollout_actor_probs_pearson_corr=0.9995`、`rollout_probs_diff_mean=0.0046`。
→ **dynamo+vLLM 的 rollout 与权重同步链路本身是通的**，卡住的只是 step 2 的训练侧显存。

---

## 12. sglang 引擎 rc=-9 其实是磁盘满（2026-08-31）

> ⚠️ **本节的归因已被 §14 推翻两次并最终修正。** rc=-9 → ENOSPC 这条链是对的，
> 但「哪个盘满了」两次都猜错：真因是**容器可写层**（`$HOME/.cache`），不是 /tmp。
> 读完本节请接着读 §14。

17016074（续行链修复后的第一次跑）失败信息是：

```
RuntimeError: dynamo vllm_workers[1] exited rc=-9
```

rc=-9 = SIGKILL，看起来像被 OOM killer 干掉。**不是。** 翻到 shard 日志才看到真正的异常：

```
File "/sgl-workspace/sglang/.../compilation/backend.py", line 471, in __call__
    with open(graph_path, "w") as f:
OSError: [Errno 28] No space left on device
→ engine.launch_phase_sigquit_handler: Received sigquit from a child process
```

sglang 的 piecewise-cudagraph 编译器写不下图文件 → SIGQUIT → 父进程死 → watchdog 报 rc=-9。
**报错信息里没有 disk、没有 space、没有 cache** —— 只有一个误导性的 rc=-9。

**ENOSPC 本身是确凿的**（来自 shard 日志的异常栈）。但「是什么占满的」我一开始的
归因是错的，这里记下修正过程，因为它比结论更有用。

最初的推断：缓存路径不带 job id ——

```bash
JOB_CACHE_BASE=${VERL_NODE_CACHE_BASE:-/tmp/verl_${USER}_sgl0512_$(hostname)}
```

同一节点上每个 job 都往同一棵 /tmp 树写 triton/flashinfer/torch-extensions 缓存，
没有东西回收；08-31 一晚在同一批节点跑了三个作业，于是第四个撑满 /tmp。听起来自洽。

**但下一次跑（17016916）在脚本里加了 `df -h /tmp` 之后，实测是：**

```
/dev/md3   42T  3.0T  37T   8% /
```

**37T 可用。** 所以「累积缓存撑满磁盘」是站不住的 —— 缓存那点量级根本填不满 42T。
真实原因更可能是 17016074 那批特定节点（pool0-00814 等）当时的根分区状态：
别的租户的作业、或该节点自身异常。**属于节点级偶发，不是本仓库的配置问题。**

加 job id + reaper 仍然是对的（消除跨作业干扰、失败可归因），但它是**防御**，不是**修复**：
下次再遇到 ENOSPC，不要以为已经解决了，先看 `df` 那行输出。

**已修**：`JOB_CACHE_BASE` 加 `${SLURM_JOB_ID}`，并在脚本开头 reap 掉同前缀、
`-mmin +180` 的旧目录（时间窗保证不会误删并发启动的兄弟作业），顺带 `df -h /tmp` 留证。

### 12.1 方法论

- **rc=-9 / SIGKILL 不要直接读成「内存不够」。** 子进程用 SIGQUIT 通知父进程时，
  外层看到的信号和真实死因可以完全无关。**永远翻到最内层那个 shard 日志。**
- **节点本地 /tmp 是跨作业共享的可变状态。** 任何写 /tmp 的缓存路径都该带 job id，
  否则失败次数越多、残留越多，故障越像"随机"。
- **归因要先量一下再下结论。** 我把 ENOSPC 归给"自己的缓存累积"，讲得通、也促成了一个
  合理的改动 —— 但只要在脚本里加一行 `df -h /tmp` 就能看到 37T 可用，直接证伪。
  **凡是"资源被占满"的推断，先加一行测量再动手改。** 测量成本是一行，猜错成本是一整轮。
- 我一度用「成功跑的 shard log 里 grep 不到 `compilation/backend.py`」推断
  "成功跑没走编译路径" —— **错的**，那个字符串只在异常栈里出现，编译成功时本就不打印。
  grep 不到错误栈 ≠ 没走那条路径。

---

## 13. 代码审查：12 条已确认缺陷（2026-08-31）

对 `verl_dynamo/verl` 的全部改动（主仓显存补丁 + recipe 门面重构 + sglang adapter + NIXL 链 + retool）
做了分维度审查，每条发现再经一轮对抗性验证（要求 refute，站不住的丢弃）。
13 条发现里 **12 条确认、1 条被推翻**。

**重要前提：没有任何一条能解释 08-31 这一晚的失败。**
今晚 5 次失败全部落在启动脚本层（§10 续行链、§11 env 顺序、§12 磁盘、gpu_mem_util、端口冲突），
代码路径本身是干净的 —— 17015763 的 step 1 指标健康就是正面证据（§11.3）。
下面这些是**潜伏问题**，按"是否影响当前脚本"分组。

### 13.1 当前脚本不受影响，但默认值是坑

**`request_completion_token_ids` 默认 `False`，engine=sglang 拿不到真实 token id**（major）
`dynamo_async_server.py:421` 默认 False 且没有 sglang 的条件翻转；默认 payload 只请求
`nvext.extra_fields=["engine_data"]`，而代码自己的 docstring 管它叫 "vLLM engine token data"。
sglang 的生成走的是同一条 frontend `/v1/completions`（新增 README 亦承认 SGLang-native
token-in/token-out "Not wired yet"）。
**⚠️ 但失败形态比 §4 bug #1 更隐蔽，别照搬那条的判据。**
bug #1 的 1-token 退化有 `response_length/mean == 1.0` 这个刺眼信号；
默认配置下的 sglang 走的是**另一条**：`_extract_completion_token_ids` 拿不到 id 后
回退到 `tokenizer.encode(text)`（`:1688`）**重新编码**文本 ——
**trainer 于是对引擎从未采样过的 token 打分（retokenization drift）**。
`response_length` 完全正常，聚合指标一切正常，只有 actor 的日志里有一条 warning。
（只有当 text 也缺失时才会 raise，那条路反而是响的。）

我们所有脚本都显式传了 `++...dynamo.request_completion_token_ids=true`，所以实测不受影响。
**建议两件事**：① 默认值改成"engine=sglang 时为 true"；
② 加一个 launch-time fail-fast（`is_sglang and not request_completion_token_ids` 直接拒启动）——
因为这个失败模式**没有任何运行期指标能暴露它**，只能在启动时挡。
新增的 README sglang 章节也完全没提这个 flag，需要补。

**`verify_weight_sync` 拿第一次同步的快照比对之后每一步**（major）
`dynamo_sglang_rollout.py:382` 只在 `_verify_sample is None` 时存快照，全仓库没有任何地方重置它。
adapter 实例跨 step 存活，于是 step≥2 是「引擎的新权重 vs trainer 的 step-1 旧快照」——
**权重正常更新反而会触发 "WEIGHT SYNC" 报错**，与 docstring 的 "runs it after each sync" 矛盾。
默认 `false`，我们没开。**要用之前必须先修。**

**NaN 能通过权重同步校验**（minor）
同一函数：`abs(a-b)` 为 NaN 时，Python 的 `max` 在首位保留 NaN、在后续位置静默跳过，
`worst > 3e-2` 恒为 False，于是打印 `weight sync VERIFIED (max_abs_diff=nan)` 并返回成功。
传输层不拦：`resp.json(content_type=None)` → `json.loads` 把 `NaN` 字面量解析成 `float('nan')`。

### 13.2 会咬人的启动期问题

**`DYN_SYSTEM_PORT` 回退到临时端口，违反 dynamo 自己的 i16 约束**（minor）
`_allocate_stable_node_port` 回退路径 bind 端口 0，拿到 ≥32768 的临时端口；
而 i16 上限是模块自己在三处（`:87-88`、`:799-801`、`dynamo_sglang_engine.py:40-42`）声明的。
engine=sglang 时 `DYN_SYSTEM_PORT` 是**唯一**控制面，被拒 → `/engine/*` 永不服务 →
`wait_ready` 空转满 600s 再抛错。**失败现场看起来像"引擎慢"，实际是端口号越界。**

**`sglang.enable_rl=false` 是提供出来的配置，但必然启动失败**（minor）
`_build_sglang_cmd:1103` 支持不加 `--enable-rl`，单测也钉了这个分支，
但 `dynamo_async_server.py:2153` 无条件用 `call_tokenizer_manager("flush_cache")` 做就绪探针 ——
该路由只在 `--enable-rl` 时注册，404 被 `wait_ready` 当成 "还没起来" 重试满 600s。
**要么删掉这个配置项，要么让探针跟着 enable_rl 走。**

### 13.3 状态一致性

**部分 fan-out 失败会让 `_sglang_released_tags` 与引擎实际状态失步**（minor）
`_sglang_control_all` 用 `return_exceptions=True` 收集，任一 shard 出错就抛 RuntimeError ——
但返回 200 的 shard **已经真的执行了** release/resume。而 `_sglang_released_tags.update(wanted)`
在 fan-out **之后**（`:338`），所以一个 shard 失败会留下「N-1 个 shard 实际已释放，而记账说没释放」。
resume 侧对称。**这正是 §4 bug #3/#4 那一类的记账-实际背离，只是触发条件更窄。**

**控制面自检只覆盖 node 0 的 shard**（minor）
`_engine_control_endpoints` 是 per-actor 的节点局部状态，自检只在 `master = self.servers[0]` 上跑；
`_launch_slave` 起完 worker 直接返回，从不探测。且无补偿检查：`_healthcheck_frontend` 只数
`endpoint == "generate"` 的注册，不看 `/engine/*`。**slave 节点的控制面坏了要到第一次 refit 才暴露。**

**单测固化的语义与实际代码相反**（minor）
`test_dynamo_sglang.py` 的 `_FakeAdapter` 把 resume 实现成交集过滤、sleep_level==1 只映射 kv_cache，
docstring 却说自己 "pin the actor-side semantics"。而实际代码是 resume **widen 到所有已释放 tag**
（16441143 的修复）、sleep 两个 level 都释放两个 tag（16280143 的修复）。
**假对象钉住的是被推翻的旧语义 —— 它现在保护的是错误行为。**

### 13.4 门面重构

**`__all__` 列了 `VllmDynamoServerAdapter`，但它只通过 `__getattr__` 惰性绑定**（minor）
`from ... import *` 会对 `__all__` 每个名字求值，在无 vLLM 镜像上触发惰性导入并抛 RuntimeError；
`hasattr()` 同样会抛（hasattr 只吞 AttributeError）。
**这重新引入了 bug #5（16510923）想根治的失败形态**，只是入口从顶层 import 变成了星号导入。

**`_load_vllm_adapter` 的 `except Exception` 把所有失败都说成"没装 vLLM"**（minor）
注释理由（"PackageNotFoundError, not ImportError"）经 MRO 实测是错的：
`PackageNotFoundError → ModuleNotFoundError → ImportError`，`except ImportError` 本就覆盖。
后果：`dynamo_vllm_rollout.py` 缺文件、其模块级错误、vLLM 自身 init 里的 CUDA/transformers 错误，
**全部**被报成"这个镜像没装 vLLM"。而该文件是本次新增、且没出现在任何 diff 里，
基于 diff 同步集群代码时最容易漏掉它 —— 届时错误信息会把人引向完全错误的方向。

**LoRA 拒绝晚了一次同步**（minor）
`dynamo_sglang_rollout.py:304` 的守卫要求 `peft_config and base_sync_done`，
LoRA 首次同步 `base_sync_done=False` 得以穿过，而 `_push_bucket` 原样发送带 `.base_layer.` 的名字
（上游父类用 `_strip_lora_base_layer` 剥掉，此处既没 import 也没镜像实现）。

### 13.5 被推翻的一条

`bucketed_weight_transfer.py` 默认 `bucket_size_mb` 512→4096 "会给依赖默认值的调用方带来 4 GiB 意外分配" ——
**推翻**：全仓库唯一的实例化点显式传了该参数，sglang 路径压根不实例化它，smoke 脚本走 1024 MB。
不存在依赖默认值的调用方，是个惰性改动，不产生错误行为。

---

## 14. ENOSPC 的真相：容器可写层，不是 /tmp（2026-08-31，§12 的最终修正）

**这一节推翻了 §12 的两次归因，并给出实测确认的真因。归因过程本身比结论有价值。**

### 14.1 三次归因，前两次都错

| 轮次 | 归因 | 怎么被推翻的 |
|---|---|---|
| 1 | 我的三个作业把节点 /tmp 的缓存撑满了 | 加一行 `df -h /tmp` → **37T 可用**，直接证伪 |
| 2 | 那批节点当时根分区异常，属节点级偶发 | 换节点重跑（17017698）**又是 ENOSPC**，不可能连着偶发 |
| 3 | **容器可写层满了** | frontend 日志给出完整路径，实测确认 ✅ |

第 3 轮之所以能定死，是因为终于拿到了**带路径的**错误信息：

```
Error adding model from discovery model_name="/workspace/hf_models/Qwen3-30B-A3B-Base"
error="symlinking /root/.cache/dynamo/mdc/by-slug/.../config.json.tmp...
      -> /root/.cache/dynamo/mdc/blobs/...: No space left on device (os error 28)"
```

**`/root/.cache`** —— 容器只挂了 `${WORKSPACE}:/workspace`，`HOME` 保持默认 `/root`，
落在容器的可写层（几 GB，且与镜像里所有其他写入共享）。
而 `/tmp` 是宿主机挂进来的 42T 盘 —— **两个完全不同的文件系统**。
`df -h /tmp` 报 37T 可用从头到尾都是对的，只是量错了对象。

### 14.2 为什么设了 XDG_CACHE_HOME 也没用

脚本一直有 `export XDG_CACHE_HOME=${JOB_CACHE_BASE}/xdg`（指向 /tmp）。
但 **dynamo 的 model-discovery 缓存不遵守 XDG_CACHE_HOME**，直接写 `$HOME/.cache/dynamo/mdc/`。
所以只设 XDG 不够，**必须设 `HOME`**。

### 14.3 后果：控制面正常，数据面从来没建立

这次的失败形态值得记住，因为它**看起来完全不像磁盘问题**：

- 引擎启动 ✅、`/engine/control/*` 全通 ✅、**权重同步完成 ✅**（32 ranks，`buckets=57 step=0`）
- release/resume 循环正常 ✅、agent loop 起来了 ✅
- 然后每个 rollout 请求都 `RuntimeError: Dynamo frontend /v1/completions failed status=404`

frontend 日志：**`added model` 出现 0 次，404 出现 260 次**。
对照成功跑 16658934：`added model` 16 次、`Adding worker` 352 次、ENOSPC 0 次。

→ **控制面和数据面走的是不同通道**：控制面是各 worker 自己的 `DYN_SYSTEM_PORT`（不落盘），
数据面要 frontend 把模型注册进路由表，而注册要往 `$HOME/.cache/dynamo/mdc/` 写 blob。
盘满 → 注册失败 → 路由表空 → 404。**weight sync 全绿完全不能说明数据面可用。**

### 14.4 已修

三个 sglang 脚本（`4n` / `native_i100` / `rl_dynamo_sglang_i100`）统一加：

```bash
export HOME=${JOB_CACHE_BASE}/home     # JOB_CACHE_BASE 在 /tmp，带 SLURM_JOB_ID
mkdir -p "$HOME" "$HOME/.cache" ...
```

⚠️ 之前那次 sglang piecewise-cudagraph 编译的 ENOSPC（§12）**极可能是同一个根因**
（编译器同样走 `$HOME/.cache`）。`--disable-piecewise-cuda-graph` 因此可能不再必要，
但它与 `enforce_eager=True` 的意图一致，先留着；等 HOME 修复稳定后可以单独 ablate。

### 14.5 方法论

- **ENOSPC 必须问"哪个文件系统"，而不是"磁盘满没满"。** 容器里至少有三个候选：
  宿主机挂载（大）、容器可写层（小）、tmpfs。`df` 不带路径等于没测。
- **报错信息里的完整路径是最贵的证据，别让它被截断。** 我前两轮都只看到
  `No space left on device` 而没看到路径，于是猜了两次；一旦 grep 出完整 `error="..."`，
  根因一秒定死。**遇到截断的错误串，第一件事是把它完整取出来。**
- **"上游全绿"不代表下游可用。** weight sync 成功 + 控制面 200 会给人"引擎没问题"的强烈错觉，
  而真正决定能不能生成的是 frontend 的路由表。**验收要验最终能力（能不能 generate），
  不是验中间步骤。**
- 三次归因里，只有第 3 次是**测出来的**，前两次都是**推出来的**。
  推断也促成了合理改动（job-id 隔离仍然值得留），但**别把推断记成结论**。

---

## 15. ✅ dynamo + vLLM 跑通（job 17017864，2026-08-31）

**这是 §11 那条「vLLM 路径从来没跑完过 3 步」的终结。**

```bash
sbatch --export=ALL,STEPS=3 validate_vllm_kv_metrics.sbatch
```

按 §11.3 的五项判据逐条验收：

| 判据 | 结果 |
|---|---|
| `KVVAL_DONE rc=` | **0**（16718419 / 17015763 / 17016912 都是 1） |
| 完成步数 | step 1 / 2 / 3 全部 |
| `response_length/mean` | 961.05 → 868.24 → 704.63 |
| `rollout_actor_probs_pearson_corr` | 0.99937 / 0.99937 / 0.99924 |
| defer 探针 | 6 条 |
| `OutOfMemoryError` | **0** |

`timing_s/step` = 132.1 → 112.3 → 80.1 s。

### 15.1 显存补丁的实测行为（三步完整）

```
step1:  Before  7.17 → After  7.17 → offload 7.17     Adam 未分配，load 是 no-op
step2:  Before  7.17 → After 14.28 → offload 7.17     +7.11 GB，用完即卸
step3:  Before  7.17 → After 14.28 → offload 7.17     复现，无累积
```

Adam = 7.11 GB（4 节点 32 卡分片），与 sglang 成功跑 16658934 的 7.17/14.28 **逐位一致**；
也等于 §7.1 那组 14.28/28.50（2 节点 16 卡）的一半 —— 三处独立观测互相印证。
**每步 load→step→offload 干净归位，无泄漏。**

### 15.2 唯一需要的改动就是 export 的位置

`VERL_DEFER_OPTIMIZER_LOAD` 的代码从头到尾没动过，改的只是 sbatch 里 `export` 与
`ray start` 的**先后顺序**（§11.2）。改之前 17015763 在 step 2 backward 连续 OOM 16 次；
改之后同一配置零 OOM 跑完 3 步。**补丁一直是对的，只是从来没装上。**

---

## 16. sglang 路径进展：三个连环坑，逐个剥开（2026-08-31）

sglang 侧每修一个就往前走一段，失败点一路后移 —— 这个推进轨迹本身是有用的诊断信息：

| job | 死在哪 | 真因 | 修法 |
|---|---|---|---|
| 17014038 | 第一次 `update_weights` | 续行链吃掉 `trainer.nnodes=4`（§10） | 注释移出续行链 |
| 17016074/16916 | 引擎启动（rc=-9） | `$HOME/.cache` ENOSPC（§14） | — |
| 17017698 | **生成阶段 404** | 同上：MDC 注册写不进盘 | `export HOME` 到 /tmp |
| 17018473 | **trainer 初始化** | 改 HOME 后 wandb 丢凭据 | 复制 `/root/.netrc` |
| 17018992 | 待验 | | |

### 16.1 HOME 修复的效果（17017698 → 17018473）

| frontend 指标 | 修复前 | 修复后 |
|---|---|---|
| `added model` | **0** | **16** |
| `Adding worker` | 0 | 17 |
| `No space left` | 260 | **0** |
| `status=404` | 260 | **0** |

16 次 `added model` 与成功跑 16658934 完全一致 —— **数据面路由至此恢复正常**。

### 16.2 我自己引入的回归：wandb 凭据

改 HOME 之后 17018473 一路穿过引擎启动、frontend 注册、走进 trainer，然后死在：

```
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

查下来是个一直被掩盖的事实：**脚本里的 `export WANDB_API_KEY` 根本没生效过**。
它发生在外层脚本进程，而 trainer 跑在 Ray actor 里，拿不到。
之前每一次能用 wandb，靠的都是**镜像里 `/root/.netrc`**
（成功跑日志中的 `wandb: Currently logged in` + `netrc` 就是证据）。
我把 HOME 挪走，等于抽掉了这条一直在承重、却没人知道它在承重的路径。

**已修**：设完 HOME 立刻把 `/root/.netrc`（及 `.config/wandb`）复制过去。
⚠️ 更彻底的做法是让 `WANDB_API_KEY` 进 BOOTSTRAP（那样才是真正按脚本意图工作），
但那会改变现有行为，留给下一轮。

### 16.3 方法论

- **失败点持续后移 = 修对了。** 三次修复把 sglang 从「权重同步前」推到「生成阶段」
  再推到「trainer 初始化」。即使还没跑通，**看失败点有没有前进**就能判断上一个修复是否有效，
  不必等最终成功。
- **搬动环境时，先问「有什么在悄悄依赖它」。** `HOME` 看起来只是缓存位置，
  实际还承载着 wandb 凭据。我只验证了要修的那条路径（磁盘），没盘点 HOME 的其他用途。
  **改一个被广泛隐式依赖的变量，代价是要把它的所有消费者列出来。**
- **"一直能用" 不等于 "按设计在用"。** wandb 能用 ≠ `WANDB_API_KEY` 生效；
  和 §11 的 `VERL_DEFER_OPTIMIZER_LOAD`（设了但没生效）是同一类错觉，
  只是这次方向相反：**那次是设了没生效，这次是没生效却一直能用**。

---

## 17. sglang 的 `rollout_probs_diff` 异常：是 logprob 不一致，不是 token 错乱（2026-08-31）

### 17.1 现象

sglang 跑起来后，一个指标刺眼地不对：

| 运行 | 引擎 | `rollout_actor_probs_pearson_corr` | `rollout_probs_diff_mean` |
|---|---|---|---|
| 17017864 | vLLM | **0.99937** | 0.0055 |
| 17018992 | sglang | **0.0501** | 0.2724 |
| 16658934（8-26，跑满 20 步） | sglang | **0.155 / 0.074** | 0.242 / 0.345 |

**注意第三行**：这不是新引入的 —— 被 §8 当作 A/B 对照 arm 的那次 sglang 跑同样是 0.07–0.15。
**sglang 路径一直如此。**

### 17.2 我连着猜错两次

① 先猜是 §13.1 那条 major 说的 **retokenization drift**（拿不到 token id → `tokenizer.encode(text)`
重编码 → trainer 对引擎没采样过的 token 打分）。看起来完美吻合：`response_length` 正常、
只有 pearson 崩。
② 又猜可能是 token 链路错乱。

**两次都错。** 而且中间还被自己写进脚本的注释骗了一次 —— grep 到 2 处 "re-encode fallback"，
排除注释行后是 **0**（这是同一晚第五次被自己的注释误报，见 §10 方法论）。

### 17.3 dump 原始样本，一次定性

按 §9「先看原始样本，再看聚合指标」，直接读 `trainer.rollout_data_dir` 的 `1.jsonl`：

```
sample 1: "Here's how you can resolve this task:\n1. We'll use the codeinterpreter
           to solve the problem. \n2. The codeinterpreter will be used to expand
           the equations and find a relation between x and y..."
sample 2: "<reflection> Break down the problem into manageable steps. Step 1:
           Understand the recurrence relations and initial values for both sequences..."
```

**完全连贯的推理文本，带 retool 任务该有的 `codeinterpreter` / `<reflection>` 结构。**
token id 若错乱或重编码出错，解码必然是乱码。→ **token 链路正常，前两个猜测都被推翻。**

### 17.4 真实结论

pearson 低 = **sglang 引擎返回的 logprob 与 trainer 重算的 logprob 不一致**，与 token 无关。

**当前配置下不影响训练正确性**：trainer 用自己重算的 logprob 更新，
rollout logprob 只进诊断指标（§8.6 的三层用法里的第一层）。
这解释了为什么 16658934 那条线能实打实地学（`critic/score/mean` −0.875 → +0.008，
准确率 6.3% → 50%，§8.8）—— 如果 token 真错了，学不动。

⚠️ **但它会在两种情况下从「诊断噪声」变成「硬故障」**（§8.6 已预警）：
- `actor.use_rollout_log_probs=true`（Bypass：直接拿 rollout logprob 当 `old_log_probs`）
- **FullyAsync**（`fully_async_rollouter.py:733` 是 `assert`，硬性要求）

→ **上这两条路之前，必须先把 sglang 的 logprob 一致性查清楚。** 候选方向（均未验证）：
transformers 5.x 融合 grouped_mm 与 trainer 侧数值路径的差异、sglang logprob 是否已含
temperature 缩放、MoE 路由在两种实现下的不确定性。

### 17.5 方法论

- **一个指标异常时，先问「这个指标进不进梯度」。** pearson/probs_diff 是诊断量，
  异常刺眼但当前无害；`response_length` 是记账链的投影，异常就致命。
  **搞混这两类会导致要么虚惊、要么漏掉真问题。**
- **聚合指标能否定，不能定性。** 我从 pearson=0.05 出发猜了两个都错的因；
  dump 一次原始样本（三条样本、不到一分钟）就定死了。
  **代价对比：猜两轮 vs 读三条样本。**
- **"这个现象是不是我引入的"要先查历史。** 翻出 16658934 的同一指标（0.07–0.15）
  立刻把问题从「今天的回归」降级为「长期特征」，优先级完全不同。

---

## 18. 集群维护打断（2026-08-31）

sgl_v5（17018992）在 step 1 完成、step 2 权重同步完成后被 **CANCELLED**：
`ExitCode 0:0`、`Reason: None`、Elapsed 35:14 远未到 4h 上限。

同时观察到：`sinfo` 显示 91 节点进入 `maint`，且**新提交作业的 job id 从 17018992 重置为 423**。
→ 集群做了维护，作业是被外部收掉的，**与代码/脚本无关**。

紧接着重投的 job 423 **也**被取消：Elapsed 只有 00:01:06，同样 `Reason: None` / `0:0`。
两个样本坐实了维护窗口，不是随机故障。约 40 分钟后再查，集群已恢复
（1009 idle / 1256 jobs / 1542 nodes running，maint 只剩 87 节点），重投即正常排队。

判据留给下次：**取消原因写 `None` + ExitCode `0:0` + 远未到时限 + job id 计数器重置**
= 集群侧动作，不要往自己的改动上归因。
job id 从 17018992 掉到 423 是最硬的那条信号 —— 计数器重置只可能是 slurmctld 重建状态。
**处置：不要连续重投**（423 就是这么白烧的），先 `sinfo` 看 maint 节点数回落再投。
（对照 §12 的教训：那次我把节点级偶发归因成了自己的缓存累积。）

---

## 19. 三个代码层修复（2026-08-31）

§10–§16 修的全是启动脚本；这一节是**代码**改动，都在 `recipe/dynamo/dynamo_async_server.py`
（备份 `.bak_nccl_0831`）。三条互相独立，都不改 vLLM 路径行为。

### 19.1 dynamo 从不给 sglang 分配 NCCL 端口 —— 真实缺陷

job 3671 再次 `rc=-9`。按 §14 的教训翻到最内层 shard 日志：

```
DistNetworkError: The server socket has failed to listen on any local network
address. port: 31981, code: -98, name: EADDRINUSE
```

查证：`_build_sglang_cmd` **完全没有** `--nccl-port`，全文件也没有任何 nccl 端口管理代码，
所以 `ServerArgs.nccl_port=None` —— **sglang 自己随机选**。
同节点 4 个 shard 并发启动、彼此不知情，撞端口只是时间问题。
（同一晚 vLLM 也撞过一次 `zmq ... Address already in use (26129)`，是同类。）

**修法**：调用点改成
`self._build_sglang_cmd(served_model_name, tp, nccl_port=self._allocate_tcp_port())`。
`_allocate_tcp_port` 维护 per-actor 的 `_allocated_tcp_ports` 保留集，
而同节点所有 shard 共享一个 actor → **端口按构造互不相同，不再靠运气**。

⚠️ 提交前专门验了一个会让整个模块加载失败的风险：新签名用了 `Optional`，
若模块没导入它会直接 NameError。实测有 `from typing import Optional` +
`from __future__ import annotations`，且文件内已有 40 处同样用法 —— 安全。
**改签名加类型注解前，先确认那个名字在模块里存在。**

### 19.2 logprob 填充由静默改为响亮

`_normalize_log_probs` 原本把 `None` 和长度不足**静默**填成 `0.0`：

```python
result.append(0.0 if value is None else float(value))
result.extend([0.0] * (token_count - len(result)))
```

**`0.0` 不是中性值** —— verl 用 `exp(0.0) == 1.0` 把它读成"引擎对这个 token 100% 确定"。
于是 logprob 缺失的响应**看起来不像缺失**：`rollout_probs_diff_valid` 仍是 1，
所有指标形状正常，而 `rollout_probs_diff_*` / `rollout_actor_probs_pearson_corr`
量的其实是填充值。**与 §4 bug #1 的一 EOS 悬崖是同一种静默降级**，
所以按同样的方式处理：前 3 次 + 每 100 次 ERROR，点名指标不可信。

⚠️ **只加日志，不改数值行为** —— 缺失时没有"正确"的填充值可选，
贸然改成 NaN 会污染下游。先让问题可见，再谈怎么填。

### 19.3 sglang + 默认 token-id 配置 → 启动即拒绝

`request_completion_token_ids` 默认 `False`，而默认开着的另一条 nvext 通道
`engine_data` **只有 dynamo.vllm 会填**。所以 `engine=sglang` 不显式设这个 flag 就拿不到
token id，回退去重新编码响应文本 —— **trainer 对引擎从未采样过的 token 打分**，
而 `response_length` / `grad_norm` / reward 全部保持正常值，
**没有任何运行期指标能暴露它**（§13.1）。只能在启动时挡。

**修法**：`is_sglang` 且配置里该键**不存在**时 raise，错误信息直接给出要加的那行 hydra 覆盖。
显式写 `false` 仍然放行 —— 区分"没想过"和"想清楚了要关"。
实测本仓库所有走 dynamo 的脚本都已显式传 `=true`，不影响现有跑。

### 19.4 这三条与 pearson=0.05 的关系

**都不是它的根因** —— 根因仍未定性（见 §17，链路 5 段全部读过且正确）。
19.2 是**诊断增强**：装上之后，如果 sglang 真的返回不完整 logprob，
下次运行会直接打印 `[logprobs] engine returned X/Y usable logprobs`，
而不必再从聚合指标反推（我已经因此猜错三次）。

---

## 20. ✅ pearson=0.05 的根因：frontend 聚合时只保留第一个 chunk 的 logprob（2026-08-31）

§17 留下的未定性问题，用**实测**定死了。

### 20.1 决定性证据

向运行中的 dynamo frontend（job 4803）直接发 verl 同构请求，只变 `max_tokens`：

| max_tokens | text 字符数 | **n_logprobs** | n_token_ids |
|---|---|---|---|
| 1 | 5 | 1 | 1 |
| 5 | 17 | **1** | 5 |
| 20 | 56 | **1** | 20 |
| 64 | 243 | **1** | 64 |

**`token_logprobs` 恒为长度 1，而 `nvext.completion_token_ids` 正确跟踪生成长度。**
返回的那 1 个值本身是有效的（`None count: 0/1`，值形如 `-2.3423846`）。

→ **dynamo frontend 把流式 chunk 聚合成非流式响应时，token id 正确累积，
logprobs 却只保留了第一个 chunk 的。** 与 §8.5 对上：dynamo 内部强制流式
（`enable_disjoint_streaming_output`），客户端拿到的非流式响应是聚合出来的。

### 20.2 这如何变成 pearson=0.05

`_normalize_log_probs(values=[1 个], token_count=N)` 把剩下的 **N−1 个填成 0.0**，
而 verl 读成 `exp(0.0)=1.0`（"引擎 100% 确定"）。平均响应 ~960 token 时，
**959/960 ≈ 99.9% 的 rollout logprob 是假值**。

于是全部观测各就各位：

| 观测 | 解释 |
|---|---|
| `diff_max = 1.0` | 填充位 \|1.0 − actor_prob\|，actor_prob≈0 处取到 1.0 |
| `diff_mean = 0.2724` | 绝大多数是填充 → ≈ 1 − mean(actor_prob) → mean≈0.73，合理 |
| `pearson = 0.0501` | 近似常数但混入 1/N 真实值 → 接近 0 而**不是** NaN |
| `response_length` 正常、样本通顺 | token id 走 nvext，**链路本来就没坏**（§17.3 已验证） |

**vLLM 路径不受影响**（`diff_mean=0.0055`），说明其 handler/聚合路径能保住全部 logprob。

### 20.3 影响范围

- **当前训练正确性不受影响**：trainer 用自己重算的 `old_log_probs` 更新，
  rollout logprob 只进诊断指标（§8.6 三层用法的第一层）。
  16658934 能实打实地学（score −0.875 → +0.008）正是因为这一点。
- **但这些诊断指标全是假的**：sglang arm 的 `rollout_probs_diff_*` 和
  `rollout_actor_probs_pearson_corr` 不能用于任何判断，历史数据同样作废
  （16658934 的 0.07–0.15 是同一原因，不是"sglang 数值差异"）。
- ⚠️ **上 `use_rollout_log_probs` bypass 或 FullyAsync 会立刻变成训练正确性事故** ——
  那两条路直接把 rollout logprob 当 `old_log_probs` 用，等于拿 99.9% 的常数 1.0 训练。
  **在修好之前，sglang 路径严禁开这两个开关。**

### 20.4 修复方向（按侵入性排序，均未实施）

1. **`calculate_log_probs=False`**（配置级，零风险）：既然诊断本来就是假的，
   关掉它还能省掉 frontend 格式化 logprob 的开销。**代价**：失去 vLLM arm 的真诊断能力，
   A/B 时两边要同时关才可比。
2. **让 sglang handler 填 `nvext.engine_data.completion_logprobs`**：
   `_extract_completion_log_probs` **已经优先读这个字段**（先 engine_data 后
   `choice.logprobs`），所以只要 handler 供数就能绕开 OpenAI 聚合层，verl 侧零改动。
   dynamo 的 `decode_handler.py` 本来就算出了完整 logprobs，只是没放进 nvext。**推荐方向。**
3. **修 dynamo frontend 的聚合逻辑**：最根本，但改的是 Rust/OpenAI 兼容层，影响面最大。

### 20.5 方法论

- **恒定的观测值是最强的信号。** `n_logprobs` 在 max_tokens 1→64 之间**恒为 1**，
  一张四行的表就把"数值差异/token 错乱/多轮错位"三个假设一起毙掉。
  **设计实验时优先找"应该随 X 变化"的量，看它变不变。**
- 我在这个问题上从聚合指标猜了**三次全错**（retokenization drift、token 错乱、multi-turn 差异），
  而一次实测就定死了。§17.5 的教训在同一晚第四次应验：**聚合指标能否定，不能定性。**
- **代码读对了不等于结论对。** 我把 5 段链路逐行读完并确认"全部正确"——这是对的，
  bug 不在这 5 段里，而在**读不到的那一段**（dynamo frontend 的 Rust 聚合层）。
  **通读代码没发现问题时，要问的是"我没读到哪一段"，而不是"那就没问题"。**

---

## 21. ✅ 修复：`--stream-interval = max_response_length`（2026-08-31）

> ⚠️ **这不是修复，是 workaround。** 真正的根因（sglang logprob 是增量非累积）见 §28，
> 已 backport 上游 #11640；修好后本节的 flag 不再必要（§28.6 未用它即达 pearson 0.9993）。

§20 定位到「frontend 聚合时只保留第一个 chunk 的 logprob」后，**零代码改动**即可修复：
把 chunk 放大到整个响应，就没有「后续 chunk」可丢。

### 21.1 A/B 实测（同脚本、同配置，只差一个 flag）

| | 实验组 17500192 | 对照组 4803 |
|---|---|---|
| `--stream-interval` | **16384**（= `max_response_length`） | 未设（引擎默认 1） |
| **usable logprobs** | **1489/1494 = 99.67%** | **12/6944 = 0.17%** |

逐条告警（`[logprobs] engine returned X/Y`）：

```
实验组: 3/4  6/7  7/8  468/469  1005/1006      → missing 恒为 1
对照组: 1/2  1/10 1/7  1/585                    → usable  恒为 1
```

**完美对称，根因钉死**：
- 对照组 `usable ≡ 1`，缺失 = N−1 → 只有第一个 chunk 幸存
- 实验组 `missing ≡ 1`，**与响应长度无关**（4→1006 都只缺 1 个）→ 单 chunk，几乎无损

### 21.2 用法

```bash
STREAM_INTERVAL=16384 sbatch ... train_30b_dynamo_sglang_4n.sh
```

脚本已加 `STREAM_INTERVAL` 环境变量（默认空 = 引擎默认 1），
组装进 `engine_kwargs.dynamo.extra_args`。**取值应等于 `max_response_length`。**

⚠️ 验证注入是否真的生效，要看**容器内**的 echo，不是外层 `set -x` 的回显 ——
`DRIVER` 是单引号 heredoc，生成 head.sh 时打印的 `${SGLANG_EXTRA_JSON}` 是字面量，
这**不代表**没生效。判据：
```
SGLANG_EXTRA_JSON: "--disable-piecewise-cuda-graph","--stream-interval=16384"   ← 执行结果
```

### 21.3 与 §8.4「stream_interval 实测无效」并不矛盾

§8.4 测的是 interval 1 vs 100 对**rollout 吞吐**的影响，结论「无差异」（2.6981→2.6982）成立。
本节测的是它对 **logprob 完整性** 的影响 —— **同一个参数，两个完全不同的作用面**。
教训：**一个参数「在某个指标上无效」不能外推成「这个参数没用」。**
当时若顺手看一眼 logprob 覆盖率，这个 bug 半个月前就该暴露。

### 21.3b 吞吐代价：样本不足，暂无定论 ⚠️

把 chunk 放大到整个响应，直觉上会损失流水线并行度，所以量了 `timing_per_token_ms/gen`。
**只取尾部饱和层**（`resp_len MAX=16384`，§8.2 的方法）：

| interval | 尾部饱和层的 per_token_ms/gen | n |
|---|---|---|
| 1（对照 4803） | 2.414 / 2.372 / 3.110 | 3 |
| 16384（实验 17500192） | 3.360 | 1 |

⚠️ **本节最初写的是"实测不花速度、可以放心默认打开"——那是拿 1 个数据点
（step1 的 2.889，且 respmax=15623 并未饱和）跟对照组比出来的，下早了。**
补上 step2 的尾部饱和点 3.360 后，它高于对照组三个点中的两个，
**"无代价"这个说法目前站不住**。

现状只能说：两组区间有重叠（2.37–3.11 vs 3.36），但样本量 3 vs 1，
**判不出有无差异**。§8.4 那次是 n=12~28 的分层统计才敢下"无差异"的结论，本节远不够。

→ **要下结论必须补样本**，且必须按 §8.2 分层（`per_token_ms/gen` 与
`response_length/max` 的相关性 r=0.88~0.95）。
在此之前，`STREAM_INTERVAL` 的正确定位是"**修正确性问题的必需项**"（§20/§21.1），
而不是"免费的"。

**教训（同一晚第二次）**：§9 写着"小样本的'接近显著'多半会回归"，
我却在 n=1 时写下"可以放心默认打开"。**下"无差异"结论的门槛比下"有差异"更高**，
因为它天然被低功效偏袒。

### 21.4 遗留

- **`missing=1` 的 off-by-one**（0.33%）：恒定 1 个而非按比例，最可能是最后一个 token
  （EOS / finish chunk）的 logprob 未进聚合。不阻塞使用，未深究。
- **历史 sglang 数据作废**：所有在此之前的 sglang 跑，`rollout_probs_diff_*` /
  `rollout_actor_probs_pearson_corr` 都是在量填充值（§20.3），包括 16658934 的 0.07–0.15。
- **上 FullyAsync / `use_rollout_log_probs` 前必须带这个 flag**，否则等于拿 99.8% 的常数 1.0 训练。

### 21.5 方法论

- **让"应该随 X 变化"的量去证伪假设。** 对照组 `usable≡1`、实验组 `missing≡1` ——
  两个**恒定值**（而不是两条趋势线）直接锁死机制。找这种量比堆样本有效得多。
- **修复要能被同一个探针验证。** §19.2 那条告警是为了让问题可见而加的，
  结果它同时成了修复的验收工具 —— 加诊断时顺手想一下"它将来怎么证明我修好了"。

---

## 22. ✅ 两条路径都跑通 + vLLM/sglang 速度对比（2026-08-31 收尾）

### 22.1 验收结果

| | dynamo + **vLLM** (17017864) | dynamo + **sglang** (4803) |
|---|---|---|
| 完成步数 | 3/3，`KVVAL_DONE rc=0` | 3/3 |
| `response_length/mean` | 961 / 868 / 705 | 1145 / 1143 / 863 |
| defer 探针 | 6 条，`7.17→14.28→7.17` | 3 组，同样模式 |
| OOM | 0 | 0 |
| 硬故障 | 0 | 0 |
| pearson | **0.9994** | 0.052（见 §20/§21） |

**结论：两条路径均可跑通。** 今晚全部失败原因见 §10–§14、§19，其中 5 类是启动脚本问题，
1 类是代码缺陷（nccl 端口）。

### 22.2 速度对比（同为 4 节点 ×8 H100、batch=16、resp_len=16384）

> ⚠️ **本节的 gen 对比已被 §24 推翻。** 那 7–12× 全部来自 cuda graph 配置不对等，
> 不是引擎差异；对齐后 sglang 反而略快于 vLLM。训练侧对比仍然有效。


**生成侧**

| | vLLM | sglang | 倍数 |
|---|---|---|---|
| `timing_s/gen` | 81.1 / 78.3 / 48.5 | 707.8 / 693.9 / 687.3 | 8.7–14× |
| `timing_per_token_ms/gen` | 0.330 / 0.352 / 0.269 | 2.414 / 2.372 / 3.110 | **7.3–11.6×** |

**训练侧（几乎无差异，sglang 略快）**

| | vLLM | sglang |
|---|---|---|
| `timing_s/update_actor` | 28.0 / 24.8 / 23.4 | 26.4 / 20.7 / 21.5 |
| `mfu/actor` | 0.0106 / 0.0110 / 0.0084 | 0.0138 / 0.0179 / 0.0119 |

→ `timing_s/step` 的 752 vs 132 秒，**差距全部来自生成**。

### 22.3 ⚠️ 这个 gen 对比**不能**归因于引擎

唯一已知的不对等变量是 **cuda graph**：

```
vLLM  : 命令行无 --enforce-eager                → cuda graph 启用
sglang: enforce_eager=True → --disable-cuda-graph
        + --disable-piecewise-cuda-graph（为绕 §12 的 ENOSPC 加的）
```

7–12 倍里有多少是引擎、有多少是「一边开图一边关图」，**当前数据无法区分**。
**不要把它记成"sglang 推理慢 8 倍"** —— 我第一次就是这么说的，被当场质疑后才查出这个混淆。

反过来，`update_actor` 这一项是**干净**的：
- `ppo_max_token_len × sp` 两边都是 **18432**（vLLM 2304×8 / sglang 4608×4）
- ⚠️ **两边 transformers 都是 5.x**（vLLM 5.5.3 / sglang 5.8.1），MoE 都走融合 `grouped_mm`。
  **§6 记的"vLLM 用 4.57.6"只适用于当时 pin 了版本的那个脚本，`kv_metrics.sh` 并没有 pin。**
  这自洽地解释了为什么两边 `update_actor` 落在同一量级。

### 22.4 待办：摘掉 cuda graph 这个变量

跑一次 `enforce_eager=False` 的 sglang（并去掉 `--disable-piecewise-cuda-graph` ——
它针对的 ENOSPC 真因已由 §14 的 `HOME` 修复解决，现在多半是多余的）：

- gen 掉到 100 秒量级 → 差距主要是 cuda graph，sglang 引擎本身没问题
- gen 仍是 700 秒 → 才是真正的引擎差异

`enforce_eager=True` 是脚本从 vLLM 参照派生时继承来的，而 vLLM 那边**现在反而没设**，
所以它未必有存在理由。

### 22.5 方法论

- **归一化过的指标依然可能不可比。** `timing_per_token_ms/gen` 已按 token 归一，
  看起来是干净的对照量，但它掩盖不了 cuda graph 这种**引擎配置**差异。
  **归一化解决的是"工作量不同"，解决不了"执行条件不同"。**
- **同一条数据要分项判断可比性**：本次 `update_actor` 可比、`gen` 不可比，
  混在一句"sglang 慢 8 倍"里就全错了。

---

## 23. §21 修复的三方验收 + 一个被揭开的新问题（2026-08-31）

### 23.1 三方对照（同 4 节点 ×8 H100、batch=16、resp_len=16384）

| | usable logprobs | pearson | `probs_diff_mean` |
|---|---|---|---|
| sglang `interval=1`（对照 4803，3 步） | **0.17%** | 0.052 / 0.056 / 0.054 | 0.239 / 0.211 / 0.268 |
| sglang `interval=16384`（修复 17500192，2 步） | **99.80%** | **0.669 / 0.663** | **0.171 / 0.177** |
| vLLM（参照 17017864，3 步） | **无告警 = 完整** | 0.9994 ×3 | **0.0055 / 0.0049 / 0.0060** |

**vLLM 一次填充告警都没有** —— 坐实 logprob 聚合丢失是 **sglang 路径特有**，
不是 dynamo frontend 对所有引擎的通病。

### 23.2 ⚠️ 修复揭开了一个此前被完全掩盖的真实问题

> ⚠️ **本小节把残留偏差归给 sglang 引擎，已被 §26 证伪。** native sglang 用同一引擎
> 测得 pearson 0.9992 —— 缺陷在 dynamo 路径，不在引擎。结论以 §26 为准。

`STREAM_INTERVAL` 把 pearson 从 0.054 拉到 0.666，是巨大改善。**但远不到 vLLM 的 0.9994。**
`probs_diff_mean` 0.24 → 0.174，而 vLLM 是 0.005 —— **仍差 30 倍**。
残留的 0.2% 缺失（§21.4 那个 `missing=1`）解释不了这个量级。

→ **即使 logprob 数量补齐，sglang 引擎算出的 logprob 与 trainer 重算的仍系统性不一致**
（平均概率差 **17%**）。修复前的 0.054 是"在量填充值"的假象；
**0.666 才是 sglang 真实的一致性水平，而它依然不合格。**

**结论加强（原 §20.3 的警告不但成立，还要升级）**：
**即使带 `STREAM_INTERVAL=16384`，sglang 路径仍然不能开
`actor.use_rollout_log_probs` 或 FullyAsync。** 0.174 的概率偏差直接进 ratio
会严重扭曲策略梯度。当前配置安全，仅仅因为 trainer 用自己重算的 `old_log_probs`。

**待查方向**（均未验证）：sglang 返回的 logprob 是否已含 temperature / top_p 处理；
`attention_backend='fa3'` 的数值精度；MoE 路由的非确定性。
排查手法建议：固定 `temperature=0` + 单 token 提示，逐 token 比对引擎与 trainer 的 logprob，
先分离"系统性偏移"和"随机噪声"。

### 23.3 方法论

- **修好一个 bug 会暴露下一个。** 填充值把真实的 30 倍差异压成了看不见的噪声；
  pearson 从 0.054 升到 0.666 看着是胜利，但**真正的信息是"它没到 0.999"**。
  **验收时要盯着与参照系的残差，而不是与自己旧值的改善幅度。**
- **一个"坏指标"可能同时由两个独立原因造成**，修掉大的那个之后要重新评估小的，
  而不是宣布结案。

---

## 24. ✅ 推翻「sglang 生成慢」：7–12× 全部来自 cuda graph（2026-08-31）

**§22.2 那张 gen 对比表已作废，结论以本节为准。**

### 24.1 决定性对照（尾部饱和层 `resp_len MAX=16384`，§8.2 方法）

| 配置 | per_token_ms/gen | timing_s/gen | timing_s/step |
|---|---|---|---|
| **sglang + cuda graph**（17501980） | **0.304** | 82.7 | 112.9 |
| vLLM + cuda graph（17017864） | 0.330 / 0.352 | 81.1 / 78.3 | 132.1 / 112.3 |
| sglang 无 graph（17500192 / 4803） | 3.360 / 2.414 / 2.372 / 3.110 | 681–708 | 721–752 |

**三条结论**：
1. **sglang 与 vLLM 性能相当，sglang 甚至略快 8–14%**（0.304 vs 0.330/0.352）。
   绝对量同样吻合：gen 82.7 vs 81.1 s、step 112.9 vs 112.3 s。
2. **7–12× 差距 100% 来自 cuda graph**：sglang 自身开图 vs 关图相差 **7.8–11.1×**。
3. **开图不影响正确性、不 OOM**：`resp_len/mean` 866/1062、`usable logprob 99.85%`、
   `OOM=0`、`hardfail=0`；pearson 0.588/0.711 与不开图的 0.663/0.669 同量级
   → **logprob 一致性（§23.2）是与 cuda graph 无关的独立问题**。

### 24.2 根源：一个被继承下来的设置

> ⚠️ **本小节的结论仅对 dynamo arm 成立。** native arm 没有 `free_engine_on_train`，
> 那里的 `enforce_eager=True` 是省显存的**必需项**，开图会 OOM。详见 §25。

`actor_rollout_ref.rollout.enforce_eager=True` 是脚本从 vLLM 参照脚本派生时顺手带过来的
（HANDOFF §3 记录了"其余逐行相同"的派生方式），**而 vLLM 那边现在反而没有设**。
sglang 侧它翻译成 `--disable-cuda-graph`，白白损失一个数量级。

`gpu_memory_utilization=0.6` 下 cuda graph 捕获完全放得下（36 档 batch、max_bs=256），
**当初没有显存上的理由**。

### 24.2b 加速已由第二个样本确认（n=2）

§24.1 的尾部饱和层原本只有 1 个点。补上 job 17503483 step1（`resp_len MAX=16384`）：

| 跑 | per_token_ms/gen |
|---|---|
| sglang+graph 17501980 step2 | 0.304 |
| sglang+graph 17503483 step1 | 0.282 |
| **均值（n=2 确认）** | **0.293** |
| vLLM+graph（n=2 饱和层） | 0.330 / 0.352 → 均值 0.341 |

两个独立样本一致（0.282 / 0.304），**sglang 比 vLLM 快约 14%**。
§24.3 里"改默认前再跑 1–2 次"的条件已满足一半。

### 24.3 建议

**把 `ENFORCE_EAGER` 默认改为 `False`**（脚本已支持该环境变量，默认仍是 `True`，
保持向后兼容；本次实验用 `ENFORCE_EAGER=False,DISABLE_PIECEWISE=0` 触发）。
改默认前建议再跑 1–2 次确认稳定性 —— 本次是 n=1（尾部饱和层只有 1 个点）。

推荐组合：`ENFORCE_EAGER=False, DISABLE_PIECEWISE=0, STREAM_INTERVAL=16384`
（速度 + logprob 完整性）。

### 24.4 方法论

- **"不可比"不等于"没结论"，而是"还差一次实验"。** §22.3 我诚实地标注了
  cuda graph 是混淆变量、拒绝下结论 —— 但真正解决问题的是**去把那个变量摘掉**。
  标注混淆是及格线，消除混淆才是答案。
- **派生脚本时继承来的参数要逐条问"为什么在这里"。** §9 早就写过这条
  （"跑通的邻近脚本是证据，不是模板"），`enforce_eager` 又验证了一次：
  它在 vLLM 参照里有其历史原因，平移到 sglang 后**没人重新论证过**，代价是 10×。
- **一个数量级的差距通常不是"引擎就是这样"，先怀疑配置。** 我第一次给出
  "sglang 慢 8 倍"时被当场质疑 —— 那个质疑是对的，而且省下了一个错误结论进文档。

---

## 25. ⚠️ §24 的结论要限定范围：native arm 的 `enforce_eager=True` 是**必需**的（2026-09-01）

§24.2 写的是「`enforce_eager=True` 只是从 vLLM 参照脚本顺手继承的，没有技术理由」——
**这对 dynamo arm 成立，对 native arm是错的。**

### 25.1 实测

100 步 A/B 第一次尝试（17504537 dynamo / 17504538 native），两 arm 都设 `ENFORCE_EAGER=False`：

| arm | `free_engine_on_train` | 开 cuda graph | 结果 |
|---|---|---|---|
| dynamo（17503483 同配置） | **true** | 是 | ✅ 3 步通过，`pertok` 0.28–0.33 |
| native（17504538） | **无此机制** | 是 | ❌ `Cuda failure 2 'out of memory'`（NCCL 调用中） |

### 25.2 根因

dynamo 路径有 `engine_kwargs.dynamo.free_engine_on_train=true`：训练期引擎交还显存
（sleep/wake，§4 bug #2 就是修这条链的）。于是 cuda graph 的额外开销放得下。

native sglang 走 `rollout.name=sglang`，**没有这个机制**：引擎显存与训练显存全程共存，
再叠加 cuda graph 捕获（36 档 batch、`max_bs=256`）就 OOM。
→ **`enforce_eager=True` 在 native 脚本里是省显存的必要设置，不是历史遗留。**

### 25.3 对 A/B 的影响：两 arm 不对称，方案要选

| 方案 | 做法 | 代价 |
|---|---|---|
| **A（已采用）** | 各自最优：dynamo 开图、native 关图 | 差距混入"显存管理能力"，不能读成引擎速度 |
| B | 严格对齐：两边都关图 | step ~721s → 4h 内只能跑 ~18 步，100 步不可能 |

选 A 的理由：`free_engine_on_train` 是 dynamo backend 的**固有特性**，人为禁用它
去凑"公平"，回答的是一个没人问的假设问题。
**但结论必须写成"两个 backend 在这套 RL 栈上的可达性能差异"，
而不是"sglang 引擎在 dynamo 下更快"** —— 差距的正确归因是显存管理，
它恰好通过 cuda graph 兑现成了吞吐。

⚠️ 方案 A 下两 arm 步数天然不等（dynamo 100 步 ≈3.3h，native 4h 上限处 ≈18 步）。
对比取共同步数、按 §8.2 只比尾部饱和层。

### 25.4 方法论

- **"这个参数没必要"是一个有作用域的结论。** 我在 dynamo arm 上验证了它多余，
  就顺手推广到了 native arm —— 而两者的显存模型根本不同。
  **跨 arm 复用结论前，先问"这两个 arm 在这件事上真的同构吗"。**
- 代价：一次 4 节点作业。**但它换来了一个更重要的认知**：
  dynamo 相对 native 的优势不在引擎，而在**训练期能不能把显存让出来**。

---

## 26. ⚠️⚠️ §23.2 归因错误：logprob 残留偏差是 **dynamo 路径特有**，不是 sglang 引擎（2026-09-01）

> ⚠️ **本节的"第二个独立缺陷"已被 §28.6 证伪。** pearson 0.62 是丢 chunk 造成的错位，
> 与聚合丢失同一根因；修复后为 0.9993。"缺陷在 dynamo 路径不在引擎"这个判断仍然成立。

### 26.1 决定性对照（100 步 A/B 首批数据）

| | pearson | `probs_diff_mean` | usable logprobs |
|---|---|---|---|
| **native sglang**（17540003） | **0.9992** | **0.0052** | 无告警（不经该代码路径） |
| dynamo + sglang（17540002） | 0.58 – 0.69 | 0.17 – 0.20 | 99.8%（已修） |
| vLLM 参照（17017864） | 0.9994 | 0.0055 | 完整 |

**native 与 dynamo 用的是同一个 sglang 引擎、同一个 transformers 5.8.1、同一个
`attention_backend='fa3'`、同一份 verl trainer。** 唯一差别是 rollout 走不走 dynamo。

### 26.2 §23.2 的推测被证伪

§23.2 写的是「即使 logprob 数量补齐，**sglang 引擎**算出的 logprob 与 trainer 重算的仍系统性
不一致」，并列了三个候选：transformers 5.x 融合 MoE、fa3 数值精度、MoE 路由非确定性。
**三个都不成立** —— 同一个引擎在 native 路径上给出 0.9992。

→ **dynamo 路径存在第二个、与聚合丢失无关的 logprob 缺陷。**
`--stream-interval` 解决的是"只回传第一个 chunk"（usable 0.17%→99.8%，§21），
但补齐**数量**之后 pearson 仍停在 0.62，说明回传的**数值本身**也不对。

候选方向（全部未验证，且现在应只在 dynamo 侧找）：
frontend 的 logprob 转换/精度截断；`_normalize_log_probs` 之外的重排/错位；
残留的 `missing=1`（§21.4）是否恰好落在高影响位置；
dynamo 是否对 logprob 施加了 temperature/top_p 后处理而 native 没有。

**排查入口**：native 与 dynamo 同 prompt 同 seed 各取一条响应，
逐 token 比对二者的 logprob 与 trainer 重算值 —— 三方对齐能立刻分出
"数值偏移"与"位置错位"。

### 26.3 影响

- **§23.2 的警告依然成立且原因更明确**：dynamo+sglang 路径**禁止**开
  `use_rollout_log_probs` / FullyAsync。
- **native sglang 路径反而是安全的**（pearson 0.9992），若确需 bypass 模式，
  native 是当前唯一可用的 sglang 选项。
- §20 的聚合 bug 与本节的数值 bug 是**两个独立缺陷**，都在 dynamo frontend 一侧。

### 26.4 方法论

- **"同一个组件"的结论要靠"换掉周边"来验证，而不是靠读它的代码。**
  我把残留偏差归给 sglang 引擎，是因为 dynamo 侧代码看起来都对；
  真正证伪它的是**把 dynamo 拿掉再测一次**。
  **想确认 X 是不是元凶，最快的实验是在没有 X 的配置里重现它。**
- 这个对照是免费的 —— A/B 实验本来就要跑 native arm，
  **只是我此前没意识到它同时是一个诊断实验。**
  设计对比实验时，顺手想一想它还能证伪什么。

---

## 27. dynamo 的 logprob 偏差**随训练恶化**，native 完全不受影响（2026-09-01）

> ⚠️ **本节已被 §28.6 证伪。** 修复根因后 pearson 三步稳定在 0.9992~0.9993 且微升，
> 不存在"随训练恶化"。原观测是错位数据的假象。

§26 证明了残留偏差是 dynamo 路径特有。100 步 A/B 跑到中段又发现：**它还在变坏**。

### 27.1 数据

```
DYNAMO  pearson : 0.625 0.628 0.692 0.653 0.633 ... 0.507 0.517 ... 0.436 0.359 0.405 0.334
        probs_diff: 0.196 0.182 0.173 0.180 0.186 ... 0.152 0.148 ... 0.139 0.133 0.125 0.122
        score     : -0.85 -0.79 -0.88 -0.80 -0.85 ... -0.52 -0.39 ... -0.34 -0.18 -0.31 -0.64

NATIVE  pearson : 0.9992 0.9994 0.9994 0.9994 0.9994      ← 五步纹丝不动
        probs_diff: 0.0052 0.0059 0.0051 0.0053 0.0050
        score     : -0.88 -0.72 -0.89 -0.69 -0.82
```

dynamo 的 pearson 从 **0.63 跌到 0.33**（约 34 步），而 `probs_diff` 反而从 0.196
改善到 0.122 —— **两个指标方向相反**。

### 27.2 一个被否定的解释

我最初猜这是**统计效应**：`score` 从 −0.85 升到 −0.16 说明模型在学、输出分布变窄，
而 pearson 是相关系数，分母（方差）变小时对同样的噪声更敏感，所以相关性恶化而绝对差改善。
解释自洽。

**但 native 否定了它。** native 经历同样的训练、同样的分布收窄，
pearson 五步稳定在 0.9994（波动在小数点后四位）。
→ **分布变窄不足以解释 dynamo 的下降；那是真实的质量恶化。**

### 27.2b ⚠️ 上一节的推断过强，就地降级（同日修正）

§27.2 说"native 否定了统计效应解释"——**证据不足**。
native 只跑了 5 步，其 `score` 全程在 −0.82 附近（−0.88/−0.72/−0.89/−0.69/−0.82，无趋势），
**从没进入 dynamo 后期那个 score ≈ −0.1 的区间**。而 dynamo 在 score 同为 −0.80 的
step 1–10，pearson 正是 0.625。

严格能比的只有起点：

| 同一 score 水平（≈ −0.80） | pearson |
|---|---|
| dynamo（step 1–10） | **0.625** |
| native（step 1–5） | **0.9994** |

→ **起点差异 = dynamo 缺陷，确定成立**（§26 不受影响）。
→ **"pearson 随训练下降"的归因（缺陷恶化 vs 分布变窄的统计效应）目前无法区分**，
因为对照组没有走到同一 score 区间。native 4h 只能跑约 18 步，大概率也走不到。

**要区分需要**：让 native 跑到同等 score（多段续跑，但 §8.7 的 NCCL resume bug 挡路），
或改用与分布宽度无关的指标（如按 token 分桶的 |Δlogprob| 中位数），
或固定一批 prompt 在不同 step 上重复评估（消除数据分布漂移）。

### 27.3 影响：缺陷等级要上调

§26 把它记成"数值本身不对"（一个固定偏差）。观测到的是：
pearson 随步数单调下降（34 步腰斩，50 步后疑似在 0.33 触底）。
⚠️ 但**这个下降有两个尚未区分的候选原因**（见 §27.2b）：缺陷随训练放大，
或模型变确定后分布变窄带来的统计效应。**不要把"缺陷会恶化"当成已证实的结论。**

- 短跑（3–5 步）看到的 pearson≈0.65 **会系统性低估**长跑的实际劣化。
  §23 当时基于 2 步得出 0.66，那只是起点值。
- 这也解释了为什么 §20 时代（logprob 几乎全是填充）反而"稳定"在 0.05 —— 
  **噪声主导时看不到趋势**；修好聚合后，真实的劣化才显露出来。
- **禁止 bypass/FullyAsync 的结论进一步加强**：偏差不仅大，还在长跑中持续变大。

### 27.4 排查建议（下一步）

在 dynamo 侧对同一批 prompt 做**跨步对比**：固定 prompt/seed，
记录 step 1 与 step 30 的引擎 logprob、trainer 重算 logprob、以及二者之差，
看劣化是均匀放大还是集中在特定 token 位置（例如长响应尾部）。
配合 §21.4 那个恒定的 `missing=1` 一起看 —— 若缺失位置随响应变长而更靠后，两者可能同源。

### 27.5 方法论

- **对照组要"可比"才能证伪解释。** 我先编了一个统计学解释，又用 native 的 5 步平稳
  宣布推翻它 —— 但 native 的 score 从没进入 dynamo 后期的区间，**两者根本不在同一工作点**。
  用一个没走到同一状态的对照组去否定解释，和用它去支持解释一样不可靠。
  **先问"这个对照组覆盖了我要比较的那个区间吗"。**
- **"指标A变好、指标B变坏"通常意味着两者度量的不是同一件事**，
  不要挑对自己有利的那个说。这里 `probs_diff` 改善只是因为概率整体变小，
  而 pearson 揭示的结构性错配在恶化。

---

## 28. ✅ 根因确定并已修：sglang logprob 是**增量**不是累积（上游 #11640 backport，2026-09-01）

**这一节推翻/收束了 §20、§21、§23.2、§26、§27 的多处推断。先读这节。**

### 28.1 根因（一个语义误解）

`common/backend/logprobs.py::extract_from_sglang_meta` 按「`output_token_logprobs` 跨 chunk 累积」
的假设做切片：

```python
new_logprobs = output_token_logprobs[num_output_logprobs_so_far:]   # 错
return log_probs, top_logprobs, len(output_token_logprobs)          # 累计游标
```

**实际它是逐 chunk 增量的**（每个 chunk 的 meta_info 只带本 chunk 的条目）。于是从第 2 个 chunk 起
`so_far` 已 ≥ 数组长度，切片恒为空 → 该 chunk 不带 `log_probs` 字段 →
Rust `create_logprobs` 收到 `None` 直接返回 → 聚合器 `if let Some(logprobs)` **整块跳过**。
而 token_ids 走 nvext 另一条通道照常累积 → 长度对不上。

### 28.2 它解释了此前所有观测

| 观测 | 解释 |
|---|---|
| interval=1 时 `token_logprobs` 恒为 **1** | 只有第 1 个 chunk 幸存 |
| interval=16384 时 `missing=1` | 大 chunk 幸存，finish chunk 被丢 |
| `usable` 0.17% | 6944 token 只回来 12 个 |
| `k3_kl` = 1635（native 0.0024，**68 万倍**） | 缺失位被填 `0.0`（概率 1.0），`exp(14)` |
| `k1` 却正常（0.0037） | 线性估计量，正负抵消，**完全掩盖了问题** |

⚠️ **§21 的 `--stream-interval=16384` 是对的缓解、错的病因理解** —— 它只是把 chunk 数压到 1，
让"只有首 chunk 幸存"恰好等于"几乎全部幸存"。**修好根因后它不再必要。**

### 28.3 上游已修，我们落后

- 修复提交：**`1c4b411f10 feat(sglang): add engine-native generate endpoint (#11640)`**，2026-08-11 合并。
  logprob 修复搭在 TITO 那个 PR 里，**subject 完全没提 logprob** —— 只搜 subject 会漏掉。
- 我们 pin 在 `94accc7389`（2026-07-06），落后 **1098 个提交**。
- ⚠️ **第一次查时我得出"上游没修"，是因为本地 `origin/main` 缓存停在 7-26，过期 5 周。
  `git fetch` 后结论完全反转。** 查上游前先看 `.git/FETCH_HEAD` 的时间。

### 28.4 已 backport（3 文件 ~20 行，逐字照抄上游）

| 文件 | 改动 |
|---|---|
| `common/backend/logprobs.py` | 去掉累积游标；改 `num_output_tokens_in_chunk` 截断；返回 2-tuple |
| `sglang/.../decode_handler.py` | 包装函数 + 调用点传 `len(output_ids)`；删 `output_logprobs_per_choice` |
| `sglang/llm_engine.py` | 同语义手改 —— **上游已随 unified backend 删除此文件（#11831），无参考实现** |

备份 `.bak_11640_backport`。离线行为验证（模拟增量 chunk）：**OLD 2/4，NEW 4/4**。

⚠️ `create_logprobs`（Rust `delta.rs`）里的 `zip` 静默截断、`tokens` 与 `token_logprobs`
可不等长、`text_offset: vec![]` 恒空 —— **上游至今未修**，是独立的潜在缺陷。


### 28.6 ✅ 端到端验证通过（job 17560576，3 步，**不带任何 workaround**）

故意跑在 `stream-interval` **未设置**（引擎默认 1）的配置下 —— 那正是 bug 最严重的情形，
修复前实测 `usable=0.17%`。

| | step1 | step2 | step3 | native | vLLM |
|---|---|---|---|---|---|
| pearson | 0.99921 | 0.99926 | **0.99933** | 0.9992 | 0.9994 |
| `k3_kl` | 0.0021 | 0.0022 | **0.0024** | 0.0024 | 0.0019 |
| `probs_diff_mean` | 0.0047 | 0.0051 | 0.0049 | 0.0051 | 0.0055 |
| 填充告警 | **0** | **0** | **0** | — | — |

`OOM=0`、`hardfail=0`。**三条路径（dynamo / native / vLLM）至此完全对齐。**

**两个额外确证**：

1. **pearson 跨步不再衰减**（0.99921→0.99926→0.99933，反而微升）→ §27「随训练恶化」证伪。
2. **`k3 ≈ k1`**：修复前 k3=1635 / k1=0.0037（比值 **44 万**），修复后 0.0021 / 0.0020（比值 **1.05**）。
   KL 很小时 k3 理论上应 ≈ k1，只有存在极端离群值时 k3 才爆炸。
   → **`k3/k1` 比值本身就是一个离群值哨兵，比看 k3 绝对值更可靠**（k3 绝对值会随 KL 大小变化，
   比值不会）。以后诊断 logprob 链路先看这个比值。

### 28.7 连带撤销的结论

| 原结论 | 现状 |
|---|---|
| §21 `--stream-interval=16384` 是修复 | **降级为 workaround**，根因修好后不再必要（本次验证未用它） |
| §26 dynamo 有"第二个独立缺陷（数值本身不对）" | **不存在** —— 0.62 是丢 chunk 造成的错位，同一根因 |
| §27 logprob 质量"随训练恶化" | **证伪** —— 错位数据的假象，修复后三步稳定在 0.999 |
| §21.4 `missing=1` 是"minor、不阻塞" | 已在 §28.1 修正：它是同一 bug 的残留，现已消失 |

### 28.5 方法论

- **"小比例"缺陷的危害取决于下游用什么函数放大它。** 0.1% 的坏 token 经线性平均是 1.6×，
  经 `exp` 是 68 万×。判断"能不能忍"之前，先列出谁会消费这个量。
- **同名指标可能有多个实现。** `k3` 在 `core_algos.kl_penalty` 里 clamp 到 [-10,10]，
  在 `rollout_corr_helper` 里**不 clamp**。看到不可能的数值先确认是哪个实现。
- **诊断集里要故意留一个对尾部敏感的量。** k1 和 probs_diff 都掩盖了这个 bug，
  只有无 clamp 的 k3 把它放大到无法忽视。
- **查上游"有没有修"之前先 `git fetch`。** 我基于 5 周前的缓存给出过一个完全相反的结论。
- **注释里引用具体标识符/错误串会污染后续所有 grep 判据。** 同一晚被自己写的注释误伤 8 次
  （监控告警 3 次、"某路径没走过"的错误推断 1 次、backport 断言 1 次等）。
  写注释时把错误串改写或加标记，别原样贴。
