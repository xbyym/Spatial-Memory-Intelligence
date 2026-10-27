"""Launch the released HY1.5 or Wan2.2 Qwen SFT recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["hy15", "wan22"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--init-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29510)
    parser.add_argument("--stop-after-steps", type=int, default=3500)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--total-batch-size", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(set(devices)) != len(devices) or args.total_batch_size % len(devices):
        parser.error("GPUs must be unique and total batch size divisible by GPU count")
    if not 1 <= args.stop_after_steps <= 3500:
        parser.error("--stop-after-steps must be 1..3500; LR schedule remains 3500 steps")
    for key in ("data", "init_model", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, SMI_DATA_ROOT=str(args.data),
               SMI_REPRO_DIR=str(args.output / "reproducibility"), PYTHONHASHSEED=str(args.seed),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               FLASH_ATTENTION_DETERMINISTIC="1",
               OMP_NUM_THREADS="3", WANDB_DISABLED="true",
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "vendor/SpatialStack"),
                                          str(ROOT / "vendor/SpatialStack/src")]))
    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(devices)}",
               f"--master_port={args.port}", str(ROOT / "vendor/SpatialStack/src/qwen_vl/train/train_qwen.py")]
    options = {
        "model_name_or_path": args.init_model, "dataset_use": f"smi_{args.backbone}_train",
        "output_dir": args.output, "tune_mm_llm": True, "tune_mm_vision": False,
        "tune_mm_mlp": False, "use_geometry_encoder": False, "bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.total_batch_size // len(devices),
        "learning_rate": 2e-5, "mm_projector_lr": 1e-5, "vision_tower_lr": 1e-6,
        "optim": "adamw_torch", "model_max_length": 12800, "data_flatten": False,
        "max_pixels": 576 * 28 * 28, "min_pixels": 16 * 28 * 28,
        "num_train_epochs": 1, "max_steps": 3500, "stop_after_steps": args.stop_after_steps,
        "warmup_ratio": 0.03, "lr_scheduler_type": "cosine", "weight_decay": 0.01,
        "logging_steps": 10, "save_steps": 3500, "save_total_limit": 10,
        "save_only_model": False, "gradient_checkpointing": True,
        "dataloader_num_workers": 4, "group_by_modality_length": True,
        "seed": args.seed, "data_seed": args.seed, "report_to": "none",
        "max_samples": args.max_samples, "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    for key, value in options.items():
        command.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_plan.json").write_text(json.dumps({"command": command,
        "backbone": args.backbone, "data": str(args.data), "init_model": str(args.init_model),
        "seed": args.seed, "total_batch_size": args.total_batch_size,
        "note": "Optimizer/RNG states are saved for resumability; historical release checkpoints contain model weights only."}, indent=2) + "\n")
    print(json.dumps({"command": command, "execute": args.execute}, indent=2))
    if args.execute:
        if not (args.init_model / "config.json").is_file():
            parser.error("--init-model must contain the downloaded training initialization checkpoint")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
