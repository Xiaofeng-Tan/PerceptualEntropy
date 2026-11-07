<h1 align="center">When Policy Entropy Constraint Fails:<br>Preserving Diversity in Flow-based RLHF via Perceptual Entropy</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.12112"><img src="https://img.shields.io/badge/NeurIPS%202026-Spotlight-blue?style=flat" alt="NeurIPS 2026 Spotlight"></a>
</p>

<p align="center">
  <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng&nbsp;Tan</a><sup>1,2,*</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=JIKuf4AAAAAJ' target='_blank'>Jun&nbsp;Liu</a><sup>2</sup>&emsp;
  <a href='https://csgaobb.github.io/' target='_blank'>Bin-Bin&nbsp;Gao</a><sup>2</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=SD4CZKAAAAAJ' target='_blank'>Yuanting&nbsp;Fan</a><sup>2</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=XtU4nDYAAAAJ' target='_blank'>Xi&nbsp;Jiang</a><sup>3</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=fqte5H4AAAAJ' target='_blank'>Chengjie&nbsp;Wang</a><sup>2</sup>&emsp;
  <a href='https://hongsong-wang.github.io/' target='_blank'>Hongsong&nbsp;Wang</a><sup>1,†</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=PcmyXHMAAAAJ' target='_blank'>Feng&nbsp;Zheng</a><sup>3,†</sup>
</p>

<p align="center">
  <sup>1</sup>Southeast University&emsp;
  <sup>2</sup>Tencent YouTu Lab&emsp;
  <sup>3</sup>Southern University of Science and Technology
</p>

<p align="center">
  <sup>*</sup>Work done during an internship at Tencent YouTu Lab.&emsp;<sup>†</sup>Corresponding authors.
</p>

<p align="center">
  <a href="https://xiaofeng-tan.github.io/projects/PEC/index.html"><img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.12112"><img src="https://img.shields.io/badge/arXiv-2605.12112-b31b1b?style=flat&logo=arXiv&logoColor=red" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2605.12112"><img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper"></a>
  <a href="https://xiaofeng-tan.github.io/PerceptualEntropyLog/"><img src="https://img.shields.io/badge/Training-Log-blue?style=flat" alt="Training Log"></a>
</p>

**TL;DR:** We explore the entropy mechanism in flow matching and introduce **perceptual entropy**
to preserve diversity while improving image quality.

> **Insight (representation perspective)**
>
> Standard VAEs primarily extract low-level features, making VAE-space entropy `H` insensitive to
> semantic diversity. For example, slightly altering eye size in a portrait shifts VAE latents
> while leaving high-level semantics nearly unchanged. In contrast, perceptual entropy `H_perc`
> in a high-level representation space directly captures semantic variations and reliably detects
> diversity collapse.

<table>
<tr><th align="left" width="230"></th>
    <th align="left">VAE-space entropy <code>H</code></th>
    <th align="left">Perceptual entropy <code>H_perc</code></th></tr>
<tr><td><b>Measured in</b></td>
    <td>the generative latent space</td>
    <td>a high-level representation space</td></tr>
<tr><td><b>What the features carry</b></td>
    <td>low-level appearance</td>
    <td>semantics</td></tr>
<tr><td><b>Slightly altering eye size</b></td>
    <td>latents shift, semantics unchanged</td>
    <td>registers as a semantic variation</td></tr>
<tr><td><b>Detects diversity collapse</b></td>
    <td>no</td>
    <td>yes</td></tr>
</table>

---

## Contents

- [News](#news)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Image generation (FLUX.1-dev)](#image-generation-flux1-dev)
- [Video generation (Wan2.1-T2V)](#video-generation-wan21-t2v)
- [Notes](#notes)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

---

## News

- **[2026/11]** Training code released for both modalities.
- **[2026/09]** Accepted by **NeurIPS 2026** as a **Spotlight**.
- **[2026/05]** Paper on [arXiv](https://arxiv.org/abs/2605.12112).

---

## Quick start

Two independent projects, one per modality. They share no code — pick the one you need.

**Images**, on 8 GPUs:

```bash
cd image
conda create -n percep_entropy python=3.10 -y && conda activate percep_entropy
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .

export HF_TOKEN="..."                 # FLUX.1-dev is gated
python scripts/download_flux.py       # -> pretrained_ckpt/flux
python scripts/download_ps.py         # -> pretrained_ckpt/ps

bash scripts/train.sh pickscore_flux_8gpu_reward_entropy_only_005
```

**Video**, on 8 GPUs:

```bash
cd video
conda create -n percep_entropy_video python=3.12 -y && conda activate percep_entropy_video
pip install -e .

export PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers
bash scripts/train_wan21.sh percec_005 0.05 0.0
```

That last argument pair is `<emb_weight> <latent_weight>`: leave the second at `0.0` for Perc-EC,
set both for Perc-VC. Checkpoints go to `logs/<recipe>/` (images) or `saves/<tag>/` (video).

No model weights are shipped — everything is downloaded from its official source. See
[Notes](#notes) for version pins, 40 GB cards, and the other things worth knowing before a long run.

---

## Project structure

```text
perceptual-entropy/
├── image/      # FLUX.1-dev: Flow-GRPO and DDPO backbones  (accelerate + DeepSpeed)
└── video/      # Wan2.1-T2V: Flow-GRPO backbone            (Ray + FSDP2)
```

<details>
<summary><b>Full directory tree</b> (click to expand)</summary>

```text
perceptual-entropy/
├── image/                                  # FLUX.1-dev: Flow-GRPO and DDPO backbones
│   ├── config/
│   │   ├── grpo.py                         # every recipe, incl. Perc-EC / Perc-VC weights
│   │   └── base.py                         # defaults shared by all recipes
│   ├── flow_grpo/
│   │   ├── rewards.py                      # reward registry; pickscore_score returns image_embs
│   │   ├── pickscore_scorer.py             # PickScore, exposing the perceptual embedding
│   │   ├── stat_tracking.py                # per-prompt group statistics / DDPO running baseline
│   │   ├── ema.py                          # EMA of the LoRA parameters
│   │   ├── aesthetic_scorer.py             # other rewards: aesthetic, CLIP, ImageReward,
│   │   ├── clip_scorer.py                  #   OCR, Qwen-VL (unused by the paper's recipes)
│   │   ├── imagereward_scorer.py
│   │   ├── ocr.py
│   │   ├── qwenvl.py
│   │   ├── prompts.py                      # prompt samplers
│   │   ├── fsdp_utils.py
│   │   └── diffusers_patch/                # pipelines returning per-step SDE log-probs
│   │       ├── flux_pipeline_with_logprob.py
│   │       ├── sd3_sde_with_logprob.py     # the SDE step both pipelines call
│   │       ├── sd3_pipeline_with_logprob.py
│   │       ├── wan_pipeline_with_logprob.py
│   │       ├── train_dreambooth_lora_flux.py   # prompt encoding helpers
│   │       └── ...                         # qwenimage / kontext variants
│   ├── scripts/
│   │   ├── train.sh                        # single entry point: train.sh <recipe>
│   │   ├── train_flux.py                   # training loop; perceptual reward shaping lives here
│   │   ├── download_flux.py                # -> pretrained_ckpt/flux
│   │   ├── download_ps.py                  # -> pretrained_ckpt/ps
│   │   └── accelerate_configs/             # deepspeed_zero{1,2}, multi_gpu, multi_node{,_ddp}
│   ├── data/prompts_test.txt               # 400 prompts for inference
│   ├── dataset/                            # training / test prompt splits
│   │   ├── pickscore/                      # Pick-a-Pic captions (the paper's main reward)
│   │   ├── ocr/                            # OCR prompts
│   │   ├── geneval/                        # GenEval metadata
│   │   └── counting_edit/
│   ├── pretrained_ckpt/                    # (not in git) FLUX.1-dev + PickScore, see Setup
│   ├── logs/                               # (not in git) checkpoints, checkpoint-<step>/lora/
│   ├── pyproject.toml
│   └── LICENSE                             # MIT, inherited from Flow-GRPO
│
├── video/                                  # Wan2.1-T2V: Flow-GRPO backbone, Ray + FSDP2
│   ├── unirl/
│   │   ├── train_diffusion.py              # Hydra entry point
│   │   ├── algorithms/
│   │   │   ├── pec.py                      # perceptual entropy + reward shaping
│   │   │   ├── flowgrpo.py                 # the GRPO surrogate
│   │   │   └── base.py
│   │   ├── trainer/
│   │   │   ├── diffusion.py                # rollout -> reward -> shaping -> advantage -> update
│   │   │   ├── eval_suites.py              # in-loop evaluation
│   │   │   ├── residency.py                # GPU residency / offload
│   │   │   ├── base.py
│   │   │   └── hydra.py
│   │   ├── models/
│   │   │   ├── wan21/                      # Wan2.1 pipeline, 3D VAE, UMT5 text encoder
│   │   │   └── types/                      # model bundle / pipeline interfaces
│   │   ├── reward/
│   │   │   ├── service.py                  # attaches rewards (and embeddings) to rollouts
│   │   │   └── local/
│   │   │       ├── video.py                # frame-averaged PickScore + temporal consistency
│   │   │       ├── pickscore.py            # inner image reward, exposes last_image_embs
│   │   │       └── registry.py
│   │   ├── rollout/engine/trainside/       # sampling with the live training weights
│   │   ├── sde/                            # noise schedule, SDE kernels, index schedule
│   │   ├── train/
│   │   │   ├── backend/fsdp/               # FSDP2 sharding
│   │   │   ├── stack/                      # the training stack
│   │   │   ├── configs.py                  # FSDP / LoRA / optimizer config objects
│   │   │   ├── lora.py
│   │   │   ├── ema.py
│   │   │   └── optim.py
│   │   ├── distributed/                    # Ray device pools, tensor transport
│   │   ├── types/                          # samples, segments, conditions, advantages
│   │   ├── data/                           # prompt data sources
│   │   └── utils/                          # logging, media, memory
│   ├── examples/diffusion/wan21/
│   │   └── wan21_t2v_percec.yaml           # the recipe; pec.* weights come from the CLI
│   ├── scripts/
│   │   ├── train_wan21.sh                  # single node
│   │   ├── train_wan21_multinode.sh        # Ray cluster over N nodes (the paper used 12 x 8)
│   │   ├── sweep_wan21.sh                  # all settings back to back, resumable
│   │   └── run_with_swanlab.py             # mirrors wandb into SwanLab
│   ├── data/
│   │   ├── video_prompts.txt               # 50k training prompts (DanceGRPO)
│   │   └── prompts100.txt                  # 100-prompt fixed eval set (DanceGRPO)
│   ├── saves/                              # (not in git) checkpoints
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── LICENSE                             # Apache-2.0, inherited from UniRL
│
├── .gitignore
└── README.md
```

</details>

---

## Image generation (FLUX.1-dev)

### Models to download

| Model | Role | Source | Goes to |
|---|---|---|---|
| FLUX.1-dev | base model | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `image/pretrained_ckpt/flux` |
| PickScore v1 | reward model | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) | `image/pretrained_ckpt/ps` |

FLUX.1-dev is gated: accept its licence on the Hub, then `export HF_TOKEN=...` before running the
two download scripts. They put the files exactly where the configs expect them.

### Training

One entry point. The argument is a recipe name from `config/grpo.py`:

```bash
bash scripts/train.sh pickscore_flux_8gpu_reward_entropy_only_005   # Perc-EC, lambda = 0.05
bash scripts/train.sh pickscore_flux_8gpu_pcvae_01                  # Perc-VC, w1 = w2 = 0.1
bash scripts/train.sh pickscore_flux_8gpu_ddpo_pec_003              # Perc-EC on DDPO

NUM_GPUS=16 PORT=29600 bash scripts/train.sh pickscore_flux_16gpu_40g_pec_01
```

Checkpoints appear under the recipe's `save_dir` as `checkpoint-<step>/lora/`.

### Recipes

Every recipe is Perc-EC or Perc-VC. The names date from the paper's development, so they read
differently from the method names:

| Name contains | Method | Meaning |
|---|---|---|
| `reward_entropy_only_*`, `pec_*` | **Perc-EC** | perceptual term only, `latent_weight = 0` |
| `pcvae_*`, `dual_entropy_on_reward_*` | **Perc-VC** | both terms, `w1 = w2` |
| `ddpo_*` | either | DDPO backbone instead of Flow-GRPO |
| `*_40g` | either | same setting, 40 GB memory path on |

Trailing digits are the weight with the decimal point dropped: `005` is 0.05, `01` is 0.1. So
`pickscore_flux_8gpu_pcvae_003` is Perc-VC at w₁ = w₂ = 0.03, and
`pickscore_flux_16gpu_40g_pec_01` is Perc-EC at λ = 0.1 on 16 forty-gigabyte cards.

Perc-EC is available at λ ∈ {0.01, 0.03, 0.05, 0.1} and Perc-VC at w ∈ {1e-4, 1e-3, 0.01, 0.03,
0.05, 0.1}; `config/grpo.py` has the full list. Two helpers there start with an underscore
(`_pickscore_flux_base`, `_pickscore_flux_ddpo_base`) — those are the shared Flow-GRPO and DDPO
configuration the recipes build on, not runnable recipes.

---

## Video generation (Wan2.1-T2V)

### Models to download

| Model | Role | Source |
|---|---|---|
| Wan2.1-T2V-1.3B-Diffusers | base model | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) |
| PickScore v1 | reward model | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) |
| CLIP-ViT-H-14 | PickScore's vision tower | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |

The two reward models load through the HuggingFace cache. Pre-download them, point `HF_HOME` at
that cache, and set `HF_HUB_OFFLINE=1` if you want training to stay off the network.

Prompts are already in `video/data/`, from [DanceGRPO](https://github.com/XueZeyue/DanceGRPO):
`video_prompts.txt` (50k for training) and `prompts100.txt` (the fixed eval set).

### Training

Single node:

```bash
export PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers

bash scripts/train_wan21.sh percec_005 0.05 0.0     # Perc-EC, lambda = 0.05
bash scripts/train_wan21.sh percvc_005 0.05 0.05    # Perc-VC, w1 = w2 = 0.05
```

Multiple nodes — the paper's runs used 12 × 8 = 96 GPUs:

```bash
NODES="10.0.0.1 10.0.0.2 ... 10.0.0.12" \
PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers \
  bash scripts/train_wan21_multinode.sh +pec.emb_weight=0.05 +pec.latent_weight=0.0
```

The script brings up a Ray cluster over `NODES` (head first, password-less ssh), waits for every
GPU to register, then runs the Hydra driver on the head. Ray places the ranks, so there is no
torchrun and no rank argument. Anything after the options is passed straight to Hydra.

Useful knobs: `DEGRADED_NODE=<ip>` makes one host contribute one GPU fewer, which is how the
96-GPU runs actually used 95 (a dead card on one host); `EXCLUDE_NODES` drops hosts entirely;
`ENV_FILE=/path/to/env.sh` is sourced on every node, which is where site-specific settings belong
(conda activation, proxy, `HF_HOME`, network interface names). The NCCL and InfiniBand defaults in
the script are a starting point — interface and HCA names are fabric-specific.

To run the whole sweep (Perc-EC and Perc-VC at 0.03 / 0.05 / 0.1) back to back:

```bash
NODES="..." PRETRAINED_MODEL=... bash scripts/sweep_wan21.sh
```

Each job resumes from its own newest checkpoint, so restarting after an interruption continues
rather than starting over.

### How the video reward works

Frame-averaged PickScore (weight 0.7) plus a temporal-consistency term (weight 0.3), configured in
`examples/diffusion/wan21/wan21_t2v_percec.yaml`. Because PickScore exposes its image embeddings,
perceptual entropy on video is computed exactly as on images: the per-video embedding is the mean
over `sample_frames` frames.

`+reward.backend.config.return_embeddings=true` is mandatory whenever `emb_weight != 0` — the
trainer raises if the backend returns no embeddings. Both launch scripts pass it for you.

---

## Notes

### Running on 40 GB cards

The `*_40g` recipes set `config.mem40g = True`. That streams the T5 text encoder from pinned host
memory block by block (bit-identical output, −9.5 GiB), shares one PickScore between training and
in-loop evaluation, slices VAE decode, and gathers images to host in chunks. Peak memory drops to
about 31.5 GiB. Two environment variables matter on these runs:

```bash
export NUMPY_MADVISE_HUGEPAGE=0                      # otherwise large numpy allocations can stall
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Version pins that matter

`image/pyproject.toml` pins what the results were produced with: Python 3.10, torch 2.6.0 (cu126),
diffusers 0.33.1, transformers 4.40.0, accelerate 1.4.0, peft 0.10.0, deepspeed 0.16.4, numpy
1.26.4. Three of those are load-bearing:

| Pin | Why |
|---|---|
| `peft==0.10.0` | later versions changed the LoRA state-dict layout, so existing `checkpoint-*/lora/` directories stop loading |
| `transformers==4.40.0` | newer releases move the T5 and CLIP encoder internals that the FLUX pipeline patches rely on |
| `numpy<2` | the torch 2.6 wheels are built against the numpy 1.x ABI |

If a compatible install misbehaves, `image/requirements-frozen.txt` is the environment verbatim —
173 packages, every transitive dependency pinned:

```bash
pip install -r requirements-frozen.txt
pip install -e . --no-deps
```

Use it only as a fallback; it will fight anything else sharing the conda environment.

### If DeepSpeed complains

`scripts/train.sh` defaults to ZeRO-2. Some accelerate versions raise `no_sync context manager is
incompatible with gradient partitioning logic of ZeRO stage 2`. Plain DDP computes the same thing,
only the sharding differs:

```bash
ACCEL=scripts/accelerate_configs/multi_gpu.yaml bash scripts/train.sh <recipe>
```

### Logging

Both projects log to [SwanLab](https://swanlab.cn/) — run `swanlab login` once. The image project
reads `SWANLAB_PROJECT` for the project name. The video project logs through `wandb` natively, and
`scripts/run_with_swanlab.py` is a thin wrapper that mirrors it into SwanLab.

### Remote reward services

Nothing here needs one: every Perc-EC and Perc-VC recipe scores with a local reward. But four
reward functions inherited from Flow-GRPO do talk to an HTTP service, and upstream hard-codes its
own hosts in the source. Those are environment variables now, defaulting to `127.0.0.1`:
`DEQA_SERVER_URL`, `UNIFIEDREWARD_REMOTE_URL`, `UNIFIEDREWARD_SERVER_URL`, `QWENVL_SERVER_URL`.
If you add one of these rewards without setting its variable, the call retries against localhost
and the run stalls instead of failing fast.

`image/scripts/accelerate_configs/multi_node.yaml` had an internal `main_process_ip` for the same
reason and now reads `127.0.0.1`. Single-node recipes never read that file; for multi-node image
training, either edit it or pass `--main_process_ip` and `--num_machines` to accelerate.

---

## Acknowledgement

Built on [Flow-GRPO](https://github.com/yifan123/flow_grpo) (the image codebase),
[UniRL](https://github.com/Tencent-Hunyuan/UniRL) (the video codebase, v0.2.0),
[ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch),
[DanceGRPO](https://github.com/XueZeyue/DanceGRPO) (video prompts) and
[diffusers](https://github.com/huggingface/diffusers).

`image/` is MIT, inherited from Flow-GRPO. `video/` is Apache-2.0, inherited from UniRL. See the
`LICENSE` inside each directory.

---

## Citation

```bibtex
@article{tan2026perceptualentropy,
  title={When Policy Entropy Constraint Fails: Preserving Diversity in Flow-based RLHF via Perceptual Entropy},
  author={Tan, Xiaofeng and Liu, Jun and Gao, Bin-Bin and Fan, Yuanting and Jiang, Xi and Wang, Chengjie and Wang, Hongsong and Zheng, Feng},
  journal={arXiv preprint arXiv:2605.12112},
  year={2026}
}
```
