<div align="center">

# Spatial Memory Intelligence (SMI)

### Endowing World Models with Understanding-Driven Long-Term Memory

<p align="center">
  <a href="https://arxiv.org/abs/2610.02521"><img src="https://img.shields.io/badge/Paper-arXiv-0088CC?style=flat-square&amp;logo=arxiv&amp;logoColor=white&amp;labelColor=555555" alt="Paper" width="116" height="24"></a>
  <a href="https://spatial-memory-intelligence.github.io/"><img src="https://img.shields.io/badge/Website-Project%20Page-9CDEC9?style=flat-square&amp;labelColor=555555" alt="Project Website" width="158" height="24"></a>
  <a href="https://spatial-memory-intelligence.github.io/#demos"><img src="https://img.shields.io/badge/Video-Comparisons-8B5CF6?style=flat-square&amp;labelColor=555555" alt="Video Comparisons" width="146" height="24"></a>
  <a href="https://huggingface.co/datasets/yycfq1314/SMI"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=flat-square&amp;labelColor=555555" alt="Data &amp; Models" width="127" height="24"></a>
</p>


</div>

## Overview

As historical memory grows, managing spatial memory becomes increasingly complex, requiring coordinated organization, maintenance, sparsification, and retrieval. This calls for a more intelligent and comprehensive memory-management approach.

We propose **Spatial Memory Intelligence (SMI)**, the first understanding-driven unified spatial-memory manager. Four coordinated atomic operations bring semantic and spatial reasoning into the memory-management pipeline, achieving comprehensive improvements in memory sparsity, spatial consistency, and generation stability across multiple baselines, benchmarks, and world-model backbones.

![SMI overview: organizing spatial memory, retrieving relevant observations, and improving consistency and stability.](File/teaser.png)

## Method

SMI maintains persistent spatial memory through four coordinated operations:

| Operation | Role in spatial-memory management |
| --- | --- |
| **Spatial clustering** | Group spatially related observations into coherent memory clusters. |
| **Within-cluster sparsification** | Remove redundant observations while retaining useful spatial evidence. |
| **Action-aware retrieval** | Select relevant memories using recent context and the next user action. |
| **Reliability-aware filtering** | Keep unreliable generated observations from entering persistent memory. |

Dedicated instructions and supervision teach the MLLM the semantic and spatial judgments required by each operation. Together, these operations support both memory updates and retrieval throughout long-horizon generation.

![SMI framework: an understanding model coordinates four atomic operations around persistent spatial memory.](File/framework.png)

## Installation

Use Linux, Python 3.10, CUDA 12.4, and `ffmpeg` / `ffprobe`.

```bash
git clone https://github.com/xbyym/Spatial-Memory-Intelligence.git
cd Spatial-Memory-Intelligence
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py
```

The environment uses PyTorch 2.6, Transformers 5.3, Diffusers 0.35, FlashAttention 2.7, and SageAttention 2.2. Runs have been tested on 96 GB NVIDIA H20 GPUs. HY1.5 and Wan2.2 support **four or eight GPUs for one inference job**.

## Models and data

Download the SMI weights and training data from [Hugging Face](https://huggingface.co/datasets/yycfq1314/SMI). There is one Qwen controller for HY1.5 and one for Wan2.2.

```bash
# Sign in to access the private assets.
hf auth login

# SMI controllers.
python scripts/download_assets.py --component controllers \
  --models-config configs/models.local.json

# Video generation backbones and encoders.
python scripts/download_assets.py --component generators \
  --models-config configs/models.local.json

# For Qwen training.
python scripts/download_assets.py --component data --extract
python scripts/download_assets.py --component training-init
```

## Inference

[scripts/run_inference.py](scripts/run_inference.py) runs **one input JSON** with either the Base generator or the Qwen-controlled SMI implementation. The same entry accepts your own first frame, prompt, and camera trajectory.

| Method | Execution |
| --- | --- |
| `--method base` | Geometric FoV memory retrieval, with no Qwen controller. |
| `--method qwen` | SMI memory management with the backbone-specific Qwen controller. |

### HY1.5

Choose four or eight GPUs for a single inference job:

```bash
INPUT=examples/hy15/case0/input.json
GPUS=0,1,2,3

# For one eight-GPU job, use:
# GPUS=0,1,2,3,4,5,6,7

# Base
python scripts/run_inference.py --backbone hy15 --input "$INPUT" \
  --models configs/models.local.json --method base --gpus "$GPUS" \
  --output outputs/hy15/base --execute

# Qwen / SMI
python scripts/run_inference.py --backbone hy15 --input "$INPUT" \
  --models configs/models.local.json --method qwen --gpus "$GPUS" \
  --output outputs/hy15/qwen --execute
```

Each command launches one job across all GPUs in `GPUS`. Run the command for the method you want. For Qwen inference, the generator and controller share those GPUs.

### Wan2.2

Use the same four- or eight-GPU setting with a Wan2.2 input:

```bash
INPUT=examples/wan22/case0/input.json
GPUS=0,1,2,3

# For one eight-GPU job, use:
# GPUS=0,1,2,3,4,5,6,7

# Base
python scripts/run_inference.py --backbone wan22 --input "$INPUT" \
  --models configs/models.local.json --method base --gpus "$GPUS" \
  --output outputs/wan22/base --execute

# Qwen / SMI
python scripts/run_inference.py --backbone wan22 --input "$INPUT" \
  --models configs/models.local.json --method qwen --gpus "$GPUS" \
  --output outputs/wan22/qwen --execute
```

Runtime settings are read from [configs/inference.hy15.json](configs/inference.hy15.json) or [configs/inference.wan22.json](configs/inference.wan22.json); pass `--settings` to use a custom settings file.

### Launch plans and outputs

Omit `--execute` to generate a launch plan without loading models or starting inference. Each output directory contains the resolved generator command, controller command, GPU assignment, input hashes, logs, and generated videos.

The launcher waits for its assigned GPU group to become idle. Add `--resume` to continue an unchanged plan and skip a completed run; use a new output directory after changing inputs or settings.

For HY1.5, `--geometry-seed N` controls point sampling, and `--geometry-points /path/to/points.npy` lets Base and Qwen reuse the same saved point set. The launch plan records the file hash.

## Qwen training

Choose four or eight GPUs:

```bash
# Four GPUs
GPUS=0,1,2,3

# Or eight GPUs
# GPUS=0,1,2,3,4,5,6,7
```

Train the HY1.5 controller with the selected GPU configuration:

```bash
python scripts/train_qwen.py --backbone hy15 --data datasets \
  --init-model models/qwen-init --output outputs/train-hy15 \
  --gpus "$GPUS" --execute
```

Train the Wan2.2 controller:

```bash
python scripts/train_qwen.py --backbone wan22 --data datasets \
  --init-model models/qwen-init --output outputs/train-wan22 \
  --gpus "$GPUS" --execute
```

Omit `--execute` to inspect the training command. Both recipes use a global batch size of 64, learning rate `2e-5`, 3,500 steps, cosine scheduling, 3% warmup, bf16, and seed 0. The language model is trained with the vision tower and projector frozen. Gradient accumulation adjusts to the GPU count.

For a short test on either four or eight GPUs, add `--stop-after-steps 2 --max-samples 32 --total-batch-size 8` and use a separate output directory. Checkpoints include optimizer and RNG states.

## License

SMI's original contributions are licensed under [Apache-2.0](LICENSE). Third-party components retain their own licenses and notices.

## Acknowledgements

We thank the teams behind [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), and [Qwen](https://github.com/QwenLM/Qwen3.5) for their open-source models and implementations.

## Citation

If you find this paper useful for your research, please kindly cite our paper:

```bibtex
@article{yang2026smi,
  title={Spatial Memory Intelligence: Endowing World Models with Understanding-Driven Long-Term Memory},
  author={Ying Yang and Guiyu Zhang and Lianghua Huang and Chang Nie and Chenyang Si and Haofan Wang and Shaoshuai Shi and Li Jiang},
  journal={arXiv preprint arXiv:2610.02521},
  year={2026},
  url={https://arxiv.org/abs/2610.02521}
}
```
