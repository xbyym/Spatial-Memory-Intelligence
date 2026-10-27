# Video generation script with distributed inference support
import torch
from hyvideo.utils.reproducibility import configure, record_runtime
configure()
record_runtime("wan22_generator")
import sys, os
import time
import argparse
import json
from typing import List, Dict

sys.path.append(os.path.abspath("."))
import numpy as np

try:
    import transformers.utils as _transformers_utils

    if not hasattr(_transformers_utils, "FLAX_WEIGHTS_NAME"):
        _transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except Exception:
    pass

from wan.inference.helper import (
    MyVAE,
    CHUNK_SIZE,
)
# Lazy import pose utilities to avoid hyvideo's parallel state init at module load
# (hyvideo/generate.py calls initialize_parallel_state() at import time which
# conflicts with WAN's own distributed init when running in single-GPU mode)
from wan.inference.pipeline_wan_w_mem_relative_rope import WanPipeline
from wan.models.dits.arwan_w_action_w_mem_relative_rope import WanTransformer3DModel
from wan.distributed import (
    get_world_group,
    maybe_init_distributed_environment_and_model_parallel,
)

from wan.models.par_vae.tools import DistController
from wan.models.par_vae.context_parallel.wrapper_vae import DistWrapper
from tqdm import tqdm

from diffusers.utils import export_to_video
from safetensors.torch import load_file



def str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")



def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to safe filename format.

    Args:
        text: Input text (e.g., prompt)
        max_length: Maximum length of output filename

    Returns:
        Sanitized filename string
    """
    import re

    # Remove or replace invalid filename characters
    # Strip special characters that are unsafe for filenames
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    # Replace spaces and commas with underscores
    text = re.sub(r"[\s,]+", "_", text)
    # Remove multiple consecutive underscores
    text = re.sub(r"_+", "_", text)
    # Strip leading/trailing underscores
    text = text.strip("_")

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")

    return text


class WanRunner:
    def __init__(
        self,
        model_id,
        ckpt_path,
        ar_model_path,
    ):
        # Get distributed environment info from torchrun
        # Extract rank information from environment variables
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.model_id = model_id
        self.ckpt_path = ckpt_path
        self.ar_model_path = ar_model_path

        # Set device
        # Assign CUDA device based on local rank for multi-GPU setup
        device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.device = device
        torch.cuda.set_device(self.local_rank)

        # Initialize distributed environment - needed even for single GPU
        # Setup distributed training for multi-process inference
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # Initialize model parallel with project's custom function
        # Enable sequence parallelism (sp_size=1 for single GPU)
        maybe_init_distributed_environment_and_model_parallel(
            1,
            sp_size=self.world_size,
            distributed_init_method="env://",
        )

        # Initialize models
        self._init_models()

    def _init_models(self):
        # Load VAE for encoding/decoding video frames
        self.vae = (
            MyVAE.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.bfloat16
            )
            .eval()
            .requires_grad_(False)
        )

        # Load main diffusion pipeline - load directly to GPU
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, vae=self.vae, torch_dtype=torch.bfloat16
        )
        # Move pipeline to GPU immediately after loading each component
        self.pipe.to(self.device)

        # Reuse the same VAE for distributed wrapper instead of loading another copy
        dist_vae = self.vae
        vae_chunk_dim = 4  # chunk width dim for vae input
        # Setup VAE for context-parallel decoding across GPUs
        dist_controller = DistController(
            self.rank, self.local_rank, self.world_size, None
        )
        dist_vae = DistWrapper(dist_vae, dist_controller, None, vae_chunk_dim)
        self.pipe.dist_vae = dist_vae

        # Load autoregressive transformer with action conditioning
        transformer_ar_action = WanTransformer3DModel.from_pretrained(
            self.ar_model_path,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        # Add action parameters to enable camera control
        transformer_ar_action.add_discrete_action_parameters()

        # Load trained checkpoint weights
        # Load directly to GPU to avoid OOM on systems with limited RAM
        state_dict = torch.load(
            self.ckpt_path, map_location=self.device
        )

        state_dict = state_dict["generator"]
        # Clean up state dict keys from training artifacts
        state_dict = {
            key.replace("model.", "") if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        state_dict = {
            (
                key.replace("_fsdp_wrapped_module.", "")
                if key.startswith("_fsdp_wrapped_module.")
                else key
            ): value
            for key, value in state_dict.items()
        }

        transformer_ar_action.load_state_dict(state_dict, strict=True)
        # Transformer already on GPU from device_map, just ensure dtype
        self.pipe.transformer = transformer_ar_action.to(dtype=torch.bfloat16)
        # Ensure entire pipeline is on GPU
        self.pipe.to(self.device)

    def predict(self, input_dict):
        prompt = input_dict.get("prompt", "")
        negative_prompt = input_dict.get("negative_prompt", "")
        num_frames = input_dict.get("num_frames", 189)
        num_inference_steps = input_dict.get("num_inference_steps", 50)
        height = input_dict.get("height", 704)
        width = input_dict.get("width", 1280)
        image_path = input_dict.get("image_path", None)
        use_memory = input_dict.get("use_memory", True)
        context_window_length = input_dict.get("context_window_length", 16)
        pose = input_dict.get("pose", "w-96")  # Default to forward movement
        num_chunk = input_dict.get("num_chunk", 4)
        qwen_memory_config = input_dict.get("qwen_memory_config", {}) or {}
        if hasattr(self.pipe, "configure_qwen_memory"):
            self.pipe.configure_qwen_memory(**qwen_memory_config)
        self.pipe.wan_memory_fov_top_k = max(
            1, int(input_dict.get("wan_memory_fov_top_k", 1))
        )
        seed = input_dict.get("seed", 0)
        torch.manual_seed(seed)

        all_video = []
        run_args = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            few_step=True,
            first_chunk_size=CHUNK_SIZE,
            return_dict=False,
            image_path=image_path,
            use_memory=use_memory,
            context_window_length=context_window_length,
        )

        # Convert pose input to tensors.  Batch evaluation callers may pass a
        # precomputed pose dict (or JSON path) so that trajectories generated
        # from navigation inputs are preserved exactly.
        # Lazy import to avoid hyvideo's parallel state init at module load time
        # (by this point, WAN's distributed init has already completed)
        from hyvideo.generate import pose_string_to_json, pose_to_input

        if isinstance(pose, str) and not pose.endswith(".json"):
            pose = pose_string_to_json(pose)
        expected_latents = int(num_chunk) * CHUNK_SIZE
        all_viewmats, all_Ks, all_action = pose_to_input(pose, expected_latents)

        all_save = []
        for chunk_i in range(num_chunk):
            torch.cuda.synchronize()
            begin_time = time.time()

            start_idx = chunk_i * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE

            curr_viewmats = all_viewmats[start_idx:end_idx]
            curr_Ks = all_Ks[start_idx:end_idx]
            curr_action = all_action[start_idx:end_idx]

            curr_viewmat_save = curr_viewmats[0].cpu().numpy().tolist()
            curr_action_save = curr_action[0].item()
            curr_save = {"viewmat": curr_viewmat_save, "action": curr_action_save}
            all_save.append(curr_save)

            self.pipe(
                **run_args,
                chunk_i=chunk_i,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )

            if self.rank == 0:
                print(
                    "Generate time for chunk ", chunk_i, "is", time.time() - begin_time
                )

            # decode
            chunk_video_parts = []
            for i in range(4):
                decode_start_time = time.time()
                video = self.pipe.decode_next_latent(output_type="np")
                decode_time = time.time() - decode_start_time
                if self.rank == 0 and i == 0:
                    print(f"Decode latent {i}: {decode_time:.4f} seconds")
                all_video.append(video)
                chunk_video_parts.append(video)

            if (
                self.rank == 0
                and (
                    qwen_memory_config.get("judge_enable", False)
                    or qwen_memory_config.get("intermediate_chunks_dir")
                )
                and hasattr(self.pipe, "record_qwen_memory_chunk")
            ):
                chunk_video = np.concatenate(chunk_video_parts, axis=1)
                self.pipe.record_qwen_memory_chunk(
                    chunk_i,
                    chunk_video,
                    action=curr_action,
                    viewmats=curr_viewmats,
                    Ks=curr_Ks,
                )

        if self.rank == 0:
            video = np.concatenate(all_video, axis=1)
            return {"video": video, "pose": pose}
        else:
            return {"video": None, "pose": pose}
