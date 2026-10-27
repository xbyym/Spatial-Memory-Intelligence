# Video generation models

SMI uses the video generation backbones and encoders distributed by their original publishers.

| Component | Source |
| --- | --- |
| HunyuanVideo 1.5 | [Tencent Hunyuan](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| Action-conditioned HY/Wan weights | [HY-WorldPlay](https://huggingface.co/tencent/HY-WorldPlay) |
| Wan2.2 TI2V 5B | [Wan-AI](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

```bash
python scripts/download_assets.py --component generators --models-config configs/models.local.json
```

The script also downloads the required text and image encoders. Authenticate with Hugging Face when the publisher requires account access. Model paths are written to `configs/models.local.json`; the downloaded files can be stored elsewhere with `--root /path/to/assets`.
