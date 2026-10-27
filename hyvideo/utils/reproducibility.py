"""Reproducible inference controls and per-process evidence for SMI runs."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import threading

_audit_lock = threading.Lock()


def configure(*, matmul_tf32=False):
    """Call before loading a model; unsupported deterministic operations raise."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Transformers forwards this flag to FlashAttention's deterministic backward.
    # torch.use_deterministic_algorithms alone does not set it in Transformers 5.3.
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    import numpy as np
    import torch

    seed = int(os.environ.get("SMI_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Numerical precision is separate from RNG determinism. HY1.5's reference
    # VAE uses TF32 matmul; disabling it changes the first-image conditioning.
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = False
    return seed


def configure_qwen():
    configure()
    # Use the same explicit reference linear-attention implementation on every
    # replica. Third-party fused kernels are not governed by PyTorch's flag.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name in ("FusedRMSNormGated", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        setattr(qwen, name, None)
    qwen.is_fast_path_available = False


def tensor_digest(tensor):
    import torch
    value = tensor.detach().to("cpu").contiguous()
    # uint8 view also supports bfloat16, which NumPy cannot represent directly.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def audit_generation(inputs, generated_ids, backend):
    """Record actual model input hashes and generated tokens, not just a seed."""
    folder = os.environ.get("SMI_REPRO_DIR")
    if not folder:
        return
    import torch
    record = {
        "backend": backend,
        "inputs": {key: tensor_digest(value) for key, value in sorted(inputs.items())
                   if isinstance(value, torch.Tensor)},
        "output_tokens": generated_ids.detach().cpu().tolist(),
        "do_sample": False,
    }
    fingerprint = json.dumps(record["inputs"], sort_keys=True).encode()
    record["input_sha256"] = hashlib.sha256(fingerprint).hexdigest()
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        with (destination / f"qwen-{os.getpid()}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def record_runtime(role):
    folder = os.environ.get("SMI_REPRO_DIR")
    if not folder:
        return
    import torch
    packages = {}
    for name in ("torch", "transformers", "diffusers", "flash-attn", "sageattention",
                 "cloudpickle", "remote-pdb", "numpy", "pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    qwen = sys.modules.get("transformers.models.qwen3_5.modeling_qwen3_5")
    qwen_backend = None if qwen is None else (
        "fused" if getattr(qwen, "is_fast_path_available", False) else "torch_reference")
    data = {"role": role, "pid": os.getpid(), "packages": packages,
            "cuda": torch.version.cuda, "seed": int(os.environ.get("SMI_SEED", "0")),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "flash_attention_deterministic": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "qwen_linear_attention": qwen_backend if "qwen" in role else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"runtime-{os.getpid()}.json").write_text(json.dumps(data, indent=2) + "\n")
