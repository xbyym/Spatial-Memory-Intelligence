"""Check the imports and four-GPU requirement used by the inference runner."""
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_entrypoints():
    """Exercise real CLI imports without loading weights or taking a GPU."""
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("QWEN_", "HY15_", "ORIG_FOV_", "CUDA_", "SMI_"))}
    env.update(PYTHONPATH=os.pathsep.join(str(root / path) for path in
               (".", "wan", "vendor/SpatialStack", "vendor/SpatialStack/src")),
               HY_ROOT=str(root),
               SPATIALSTACK_ROOT=str(root / "vendor/SpatialStack"),
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    commands = [["-m", "hyvideo.generate", "--help"],
                ["scripts/generate_wan22.py", "--help"],
                ["scripts/qwen_judge_wan22.py", "--help"],
                ["vendor/SpatialStack/src/qwen_vl/train/train_qwen.py", "--help"]]
    for arguments in commands:
        result = subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode or "usage:" not in result.stdout.lower():
            raise RuntimeError(f"Inference entry point failed: {arguments}\n"
                               + result.stdout + result.stderr)
    return {"hy15": "passed", "wan22": "passed", "qwen_judge": "passed", "qwen_training": "passed", "gpu_hidden_for_cli_checks": True}


def main():
    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                                      capture_output=True, text=True)
    check_output = dependency_check.stdout.strip()
    metadata_warning = None
    if dependency_check.returncode:
        # PyPI's decord 0.6.0 py3 wheel retains a stale cp36 tag in WHEEL.
        # Accept only this exact metadata finding, then exercise its decoder below.
        wheel = importlib.metadata.distribution("decord").read_text("WHEEL") or ""
        if (check_output == "decord 0.6.0 is not supported on this platform"
                and importlib.metadata.version("decord") == "0.6.0"
                and "Tag: cp36-cp36m-manylinux2010_x86_64" in wheel
                and not dependency_check.stderr.strip()):
            metadata_warning = check_output
        else:
            raise RuntimeError(dependency_check.stdout + dependency_check.stderr)
    import torch
    packages = {}
    for module, distribution in [("transformers", "transformers"), ("diffusers", "diffusers"),
        ("flash_attn", "flash-attn"), ("qwen_vl_utils", "qwen-vl-utils"),
        ("accelerate", "accelerate"), ("peft", "peft"), ("decord", "decord"),
        ("sageattention", "sageattention"), ("cloudpickle", "cloudpickle"),
        ("remote_pdb", "remote-pdb")]:
        importlib.import_module(module)
        packages[distribution] = importlib.metadata.version(distribution)
    from transformers import Qwen3_5ForConditionalGeneration
    del Qwen3_5ForConditionalGeneration
    entrypoints = check_entrypoints()
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
               for i in range(torch.cuda.device_count())]
    report = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "packages": packages, "gpus": devices,
              "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
              "pip_check_metadata_warning": metadata_warning,
              "inference_entrypoint_checks": entrypoints}
    print(json.dumps(report, indent=2))
    if len(devices) < 4 or not report["ffprobe"] or not report["ffmpeg"]:
        raise SystemExit("Inference requires at least four CUDA GPUs and ffmpeg/ffprobe")
    import decord
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "decoder_check.mp4"
        subprocess.run([report["ffmpeg"], "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=64x64:r=4:d=1", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(fixture)], check=True)
        reader = decord.VideoReader(str(fixture), ctx=decord.cpu(0))
        frames = reader.get_batch(list(range(len(reader)))).asnumpy()
        if frames.shape != (4, 64, 64, 3):
            raise RuntimeError(f"Unexpected decoded fixture shape: {frames.shape}")
        print(json.dumps({"decord_decode_check": "passed", "frames": len(reader)}))


if __name__ == "__main__":
    main()
