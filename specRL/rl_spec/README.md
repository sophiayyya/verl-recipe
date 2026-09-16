# Accelerating RL Training with Co-Adapted Diffusion Drafter
## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

## Code
Please refer to [zxgx/verl](https://github.com/zxgx/verl/tree/rl_spec_release).

## Requirements

- Docker image: `verlai/verl:vllm023.dev1`

- The trainer is supported by [VeOmni](https://github.com/ByteDance-Seed/VeOmni)
  ```
  pip3 install git+https://github.com/ByteDance-Seed/VeOmni.git@f90b3dc6fbb0ce693745223cc7a94064123dbf4d --ignore-requires-python --no-deps
  pip3 install flash-linear-attention==0.4.1
  pip3 install transformers==5.3.0
  ```

- (Optional) To run load-balance strategy DCP with Split-KV, install the planning tool, kahypar by:
  ```
  pip3 install kahypar==1.3.7
  ```

## Overview
The [code branch](https://github.com/zxgx/verl/tree/rl_spec_release) can serve as a one-stop codebase to SFT a speculative decoding drafter (MTP & DFlash) then apply it for co-training drafter with the verifier during the RL process.

Beyond the functionality support, we investigate accelerating the rollout process during RL training with diffusion drafter, i.e., [DFlash](https://github.com/z-lab/dflash).
Specifically, we design:
1. Split-KV, a tailored context parallelism for DFlash drafter training. Given that DFlash trains with constant amount of anchors sampled from each trajectory, Split-KV gathers anchor blocks (Q) while shards context (KV), achieving constant communication volumn regardless of context length.
2. Entropy boost anchor sampling, which selectively choose anchors based on the logprob computed from the verifier during `compute_logprob` of each RL training step, boosting the alignment between drafter and verifier during rollout.

## Evaluation
### RL
- datasets
  - geo3k
  - dapo
- entrypoint scripts
  - [`examples/rl_spec/geo3k`](https://github.com/zxgx/verl/tree/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/geo3k) contains all scripts for geo3k dataset
  - [`examples/rl_spec/dapo`](https://github.com/zxgx/verl/tree/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/dapo) contains all scripts for dapo dataset
#### Entropy-based anchor sampling

- Qwen3.5-35B-A3B, geo3k dataset

  <img src="https://raw.githubusercontent.com/zxgx/verl/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/assets/geo3k_qwen3_5_35b_a3b_rollout_accept_length.png" alt="Rollout accept length of Qwen3.5-35B-A3B on geo3k" width="520">

- Qwen3.5-9B, dapo dataset

  <img src="https://raw.githubusercontent.com/zxgx/verl/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/assets/dapo_qwen3_5_9b_rollout_accept_length.png" alt="Rollout accept length of Qwen3.5-9B on dapo" width="520">

#### Split-KV scaling benchmarks
The experiment results are reported from Qwen3.5-9B. As this part only involves the drafter overhead, the results should be applicable to Qwen3.5-35B-A3B.
- Context parallelism scaling over SP size and draft query length (anchor blocks)

  <img src="https://raw.githubusercontent.com/zxgx/verl/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/assets/dflash_cp_scaling.png" alt="Context parallelism scaling over SP size and draft query length" width="780">

- Imbalance ratio and planning time with different balance strategy

  <img src="https://raw.githubusercontent.com/zxgx/verl/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/assets/dflash_balance_strategies.png" alt="Imbalance ratio and planning time with different balance strategy" width="780">

#### Drafter Reproduction SFT
- dataset:
[`examples/rl_spec/collect_sft_dataset`](https://github.com/zxgx/verl/tree/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/collect_sft_dataset) reproduce the SFT dataset mentioned by DFlash paper.
- entrypoint scripts
[`examples/rl_spec/run_qwen3_5_35b_a3b_dflash_warmup_veomni.sh`](https://github.com/zxgx/verl/blob/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/run_qwen3_5_35b_a3b_dflash_warmup_veomni.sh) enables the DFlash drafter training.
[`examples/rl_spec/run_qwen3_5_35b_a3b_mtp_warmup_veomni.sh`](https://github.com/zxgx/verl/blob/300c431de72ef52c338ab738d13efea2d19f410e/examples/rl_spec/run_qwen3_5_35b_a3b_mtp_warmup_veomni.sh) enables the MTP drafter training.
- decoding performance of the resulting drafters
  | Drafter | Dataset | Throughput (tok/s) | Speedup | Accept length |
  | --- | --- | --- | --- | --- |
  | No drafter | gsm8k | 178.31 | 1.00 | 0.00 |
  | No drafter | humaneval | 177.84 | 1.00 | 0.00 |
  | Official weight from huggingface [Qwen3.5](http://huggingface.co/Qwen/Qwen3.5-35B-A3B) and [DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
  | MTP, 2 spec tokens | gsm8k | 269.65 | 1.51 | 2.78 |
  | MTP, 2 spec tokens | humaneval | 276.86 | 1.56 | 2.78 |
  | DFlash | gsm8k | 410.84 | 2.30 | 6.77 |
  | DFlash | humaneval | 487.95 | 2.74 | 6.77 |
  | Reproduction SFT |
  | MTP, e2e tv | gsm8k | 291.37 | 1.63 | 3.24 |
  | MTP, e2e tv | humaneval | 295.28 | 1.66 | 3.24 |
  | DFlash, ce | gsm8k | 405.65 | 2.27 | 6.49 |
  | DFlash, ce | humaneval | 458.00 | 2.58 | 6.49 |

The verifier model is Qwen3.5-35B-A3B. The testbed is 2xA100-80G, tp=2.
Note that the `Accept length` are the overall statistics on the two evaluation datasets, as vllm only reports engine-level spec statistics.
