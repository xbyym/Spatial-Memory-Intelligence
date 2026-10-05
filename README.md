<div align="center">

# Spatial Memory Intelligence (SMI)

### Endowing World Models with Understanding-Driven Long-Term Memory

[Paper](https://arxiv.org/abs/2610.02521) · [Project Website](https://spatial-memory-intelligence.github.io/) · [Video Comparisons](https://spatial-memory-intelligence.github.io/#demos) · [Method](https://spatial-memory-intelligence.github.io/#method)

We are currently preparing the code, datasets, model weights, and other materials for release.

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

## Video Comparisons

The project website presents synchronized, side-by-side video comparisons, grouped by world-model backbone: **HY1.5** and **Wan2.2**.

Each example compares SMI with Base. Selected examples also include FramePack, Deep Forcing, MoC, VMem, and MemFlow. Dynamic red boxes highlight selected spatial and structural inconsistencies.

**[Explore the interactive comparisons →](https://spatial-memory-intelligence.github.io/#demos)**

## Release Status

This public repository currently provides the project overview and supporting figures in `File/`. Training, inference, and evaluation code is coming soon.

| Resource | Status |
| --- | --- |
| Project website and video demonstrations | [Available](https://spatial-memory-intelligence.github.io/) |
| Training, inference, and evaluation code | Coming soon |
| Paper | [Available on arXiv](https://arxiv.org/abs/2610.02521) |
| Model weights on Hugging Face | Link to be added |
| Dataset | Link to be added |

Installation requirements, runnable examples, and reproduction instructions will accompany the code release.

## Authors

[Ying Yang](https://github.com/xbyym)<sup>1,*</sup>, [Guiyu Zhang](https://grenoble-zhang.github.io/)<sup>1,2,*</sup>, [Lianghua Huang](https://github.com/huanglianghua)<sup>2</sup>, [Chang Nie](https://github.com/Clare-Nie)<sup>4</sup>, [Chenyang Si](https://chenyangsi.top/)<sup>4</sup>, [Haofan Wang](https://haofanwang.github.io/)<sup>5</sup>, [Shaoshuai Shi](https://shishaoshuai.com/)<sup>6</sup>, and [Li Jiang](https://llijiang.github.io/)<sup>1,3,†</sup>.

\* Equal contribution. † Corresponding author.

<sup>1</sup> The Chinese University of Hong Kong, Shenzhen · <sup>2</sup> Alibaba Group · <sup>3</sup> Shenzhen Loop Area Institute · <sup>4</sup> Nanjing University · <sup>5</sup> Lovart AI · <sup>6</sup> Voyager Research, Didi Chuxing
