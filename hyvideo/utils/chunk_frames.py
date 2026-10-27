"""Save Base chunk keyframes using the same format as the SMI memory writer."""
# Copyright 2026 The SMI authors. Licensed under Apache-2.0.
from pathlib import Path

import imageio.v2 as imageio
import torch


class ChunkFrameWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, chunk_i, video, viewmats=None, action=None):
        """video is the VAE's float32 C,F,H,W output in [0,1]."""
        rgb = (video.detach().cpu().permute(1, 2, 3, 0) * 255).clamp(0, 255).to(torch.uint8)
        for name, index in [('mid', len(rgb) // 2), ('last', len(rgb) - 1)]:
            path = self.output_dir / f'chunk_{chunk_i:03d}_{name}.png'
            imageio.imwrite(path, rgb[index].numpy())
