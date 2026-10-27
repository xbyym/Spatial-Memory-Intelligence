#!/usr/bin/env python3
"""Generate navigation videos with one persistent Wan2.2 pipeline.

Launch with torchrun. Every rank loads the same input JSONL and generates the
same inputs; rank 0 saves each video as ``case_<id>_combined.mp4``.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video


ROOT = Path(__file__).resolve().parents[1]
HY_ROOT = Path(os.environ.get("HY_ROOT", ROOT)).resolve()

for path in (str(ROOT), str(HY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vendor.navigation.navigation_to_poses import navigation_to_poses, navigation_to_poses_orbit
from wan.generate import WanRunner


RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def log(message):
    if RANK == 0:
        print(message, flush=True)


def extract_prompt(prompt_dict):
    for key in sorted(prompt_dict, key=lambda item: int(item)):
        text = str(prompt_dict[key]).strip()
        if text:
            return text
    return ""


def load_samples(path, names=None, max_samples=None):
    with open(path, "r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    if names:
        selected = set(names)
        samples = [sample for sample in samples if sample.get("name") in selected]
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples


def resolve_sample_path(sample, field, jsonl_path):
    """Resolve generation assets relative to the input manifest."""
    value = sample.get(field)
    if not value:
        raise ValueError(f"input requires {field}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(jsonl_path).resolve().parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{sample.get('name', 'input')}: missing {field}: {candidate}"
        )
    return str(candidate)


def append_status(path, payload):
    if RANK != 0 or not path:
        return
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_pose(sample, jsonl_path):
    if sample.get("requires_explicit_poses"):
        pose_path = resolve_sample_path(sample, "pose_json_path", jsonl_path)
        with open(pose_path, "r", encoding="utf-8") as handle:
            pose_json = json.load(handle)
        pose_keys = list(pose_json)
        expected_keys = [str(index) for index in range(len(pose_keys))]
        if pose_keys != expected_keys:
            raise ValueError(
                f"{sample.get('name')}: explicit pose keys must be ordered 0..N-1"
            )
        total_latents = len(pose_keys)
        if not total_latents or total_latents % 4:
            raise ValueError(
                f"{sample.get('name')}: explicit pose count {total_latents} "
                "must be positive and divisible by 4"
            )
        num_chunk = total_latents // 4
        num_frames = (total_latents - 1) * 4 + 1
        return pose_path, num_chunk, num_frames

    kwargs = {
        "navigation": sample["navigation"],
        "chunk_length": sample["chunk_length"],
        "fps": 16,
        "temporal_compression": 4,
    }
    if sample.get("perspective", "first_person") == "third_person":
        pose_json, _, total_latents = navigation_to_poses_orbit(**kwargs)
    else:
        pose_json, _, total_latents = navigation_to_poses(**kwargs)
    if total_latents % 4:
        raise ValueError(
            f"{sample.get('name')}: total_latents={total_latents} is not divisible by 4"
        )
    num_chunk = total_latents // 4
    num_frames = (total_latents - 1) * 4 + 1
    return pose_json, num_chunk, num_frames


def qwen_config(args, case_name, perspective):
    intermediate_dir = None
    if args.qwen_intermediate_root:
        intermediate_dir = str(Path(args.qwen_intermediate_root) / case_name)
    return {
        "judge_enable": args.mode == "qwen",
        "judge_url": args.qwen_memory_judge_url,
        "judge_model_path": args.qwen_memory_judge_model_path,
        "judge_device": args.qwen_memory_judge_device,
        "judge_dtype": args.qwen_memory_judge_dtype,
        "select_fov_top_k": args.qwen_memory_select_fov_top_k,
        "select_top_k": args.qwen_memory_select_top_k,
        "local_context_chunks": args.qwen_memory_local_context_chunks,
        "perspective": str(perspective or "first_person"),
        "fov_fill_min_chunks": args.qwen_memory_fov_fill_min_chunks,
        "allow_fallback": args.qwen_memory_allow_fallback,
        "frames_per_chunk": args.qwen_memory_frames_per_chunk,
        "request_timeout_sec": args.qwen_memory_request_timeout_sec,
        "select_do_sample": args.qwen_memory_select_do_sample,
        "select_temperature": args.qwen_memory_select_temperature,
        "select_top_p": args.qwen_memory_select_top_p,
        "select_seed": args.qwen_memory_select_seed,
        "select_policy": args.qwen_memory_select_policy,
        "visual_quality_enable": args.qwen_memory_visual_quality_enable,
        "intermediate_chunks_dir": intermediate_dir,
        "update_enable": args.qwen_memory_update_enable,
        "update_fov_top_k": args.qwen_memory_update_fov_top_k,
        "class_capacity": args.qwen_memory_class_capacity,
        "member_batch_size": args.qwen_memory_member_batch_size,
        "confirm_prototype_min_chunks": args.qwen_memory_confirm_prototype_min_chunks,
        "prototype_min_fov_overlap": args.qwen_memory_prototype_min_fov_overlap,
        "drop_min_fov_overlap": args.qwen_memory_drop_min_fov_overlap,
        "disable_sparse": args.qwen_memory_disable_sparse,
        "sparse_member_from_ends": args.qwen_memory_sparse_member_from_ends,
        "select_active_only": args.qwen_memory_select_active_only,
        "select_filter_visual_quality": args.qwen_memory_select_filter_visual_quality,
        "strict": args.qwen_memory_strict,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--status_jsonl", default=None)
    parser.add_argument("--mode", choices=["baseline", "qwen"], required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", type=str_to_bool, default=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--ar_model_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_sample_seed",
        type=str_to_bool,
        default=False,
        help="Use each JSONL row's seed when present; otherwise use --seed.",
    )
    parser.add_argument("--context_window_length", type=int, default=16)
    parser.add_argument("--wan_memory_fov_top_k", type=int, default=2)
    parser.add_argument("--qwen_memory_judge_url", default=None)
    parser.add_argument("--qwen_memory_judge_model_path", default=None)
    parser.add_argument("--qwen_memory_judge_device", default="cuda:7")
    parser.add_argument("--qwen_memory_judge_dtype", default="bfloat16")
    parser.add_argument("--qwen_memory_select_fov_top_k", type=int, default=7)
    parser.add_argument("--qwen_memory_select_top_k", type=int, default=3)
    parser.add_argument("--qwen_memory_local_context_chunks", type=int, default=3)
    parser.add_argument("--qwen_memory_fov_fill_min_chunks", type=int, default=0)
    parser.add_argument("--qwen_memory_allow_fallback", type=str_to_bool, default=False)
    parser.add_argument("--qwen_memory_frames_per_chunk", type=int, default=1)
    parser.add_argument("--qwen_memory_request_timeout_sec", type=float, default=180.0)
    parser.add_argument("--qwen_memory_select_do_sample", type=str_to_bool, default=False)
    parser.add_argument("--qwen_memory_select_temperature", type=float, default=0.7)
    parser.add_argument("--qwen_memory_select_top_p", type=float, default=0.9)
    parser.add_argument("--qwen_memory_select_seed", type=int, default=None)
    parser.add_argument("--qwen_memory_select_policy", default="coverage_only")
    parser.add_argument("--qwen_memory_visual_quality_enable", type=str_to_bool, default=True)
    parser.add_argument("--qwen_memory_update_enable", type=str_to_bool, default=True)
    parser.add_argument("--qwen_memory_update_fov_top_k", type=int, default=6)
    parser.add_argument("--qwen_memory_class_capacity", type=int, default=7)
    parser.add_argument("--qwen_memory_member_batch_size", type=int, default=8)
    parser.add_argument("--qwen_memory_confirm_prototype_min_chunks", type=int, default=3)
    parser.add_argument("--qwen_memory_prototype_min_fov_overlap", type=float, default=0.0)
    parser.add_argument("--qwen_memory_drop_min_fov_overlap", type=float, default=0.9)
    parser.add_argument("--qwen_memory_disable_sparse", type=str_to_bool, default=False)
    parser.add_argument("--qwen_memory_sparse_member_from_ends", default="-2,-3")
    parser.add_argument("--qwen_memory_select_active_only", type=str_to_bool, default=True)
    parser.add_argument("--qwen_memory_select_filter_visual_quality", type=str_to_bool, default=True)
    parser.add_argument("--qwen_memory_strict", type=str_to_bool, default=True)
    parser.add_argument("--qwen_intermediate_root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if RANK == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    samples = load_samples(args.jsonl_path, args.names, args.max_samples)
    log(f"Loaded {len(samples)} samples from {args.jsonl_path}")
    log(f"Loading persistent WAN2.2 pipeline mode={args.mode} world_size={WORLD_SIZE}")
    runner = WanRunner(args.model_id, args.ckpt_path, args.ar_model_path)

    for index, sample in enumerate(samples, start=1):
        name = sample.get("name", "unnamed")
        output_path = output_dir / f"{name}_combined.mp4"
        if args.skip_existing and output_path.exists():
            log(f"[{index}/{len(samples)}] skip existing {output_path}")
            append_status(
                args.status_jsonl,
                {"case": name, "status": "skipped", "output": str(output_path)},
            )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            continue

        started = time.time()
        local_error = None
        local_traceback = None
        log(f"[{index}/{len(samples)}] start {name}")
        try:
            image_path = resolve_sample_path(sample, "image_path", args.jsonl_path)
            prompt = extract_prompt(sample.get("prompt", {}))
            if not prompt:
                raise ValueError("empty prompt")
            pose, num_chunk, num_frames = build_pose(sample, args.jsonl_path)
            sample_seed = (
                int(sample.get("seed", args.seed))
                if args.use_sample_seed
                else int(args.seed)
            )
            result = runner.predict(
                {
                    "prompt": prompt,
                    "negative_prompt": "",
                    "num_frames": num_frames,
                    "num_inference_steps": args.num_inference_steps,
                    "height": args.height,
                    "width": args.width,
                    "image_path": image_path,
                    "use_memory": True,
                    "context_window_length": args.context_window_length,
                    "seed": sample_seed,
                    "pose": pose,
                    "num_chunk": num_chunk,
                    "wan_memory_fov_top_k": args.wan_memory_fov_top_k,
                    "qwen_memory_config": qwen_config(
                        args,
                        name,
                        sample.get("perspective", "first_person"),
                    ),
                }
            )
            if RANK == 0:
                raw_video = result["video"][0]
                raw_frame_count = len(raw_video)
                if raw_frame_count != num_frames:
                    raise ValueError(
                        f"{name}: generated {raw_frame_count} raw frames, "
                        f"expected {num_frames}"
                    )
                delivery_frame_count = int(
                    sample.get("delivery_frame_count") or raw_frame_count
                )
                if delivery_frame_count < raw_frame_count:
                    raise ValueError(
                        f"{name}: delivery_frame_count={delivery_frame_count} is "
                        f"smaller than raw_frame_count={raw_frame_count}"
                    )
                raw_output_path = None
                delivery_video = raw_video
                output_fps = 24 if sample.get("requires_explicit_poses") else 16
                if delivery_frame_count > raw_frame_count:
                    raw_output_dir = output_dir / "raw1437"
                    raw_output_dir.mkdir(parents=True, exist_ok=True)
                    raw_output_path = raw_output_dir / f"{name}_raw{raw_frame_count}.mp4"
                    export_to_video(raw_video, str(raw_output_path), fps=output_fps)
                    pad_count = delivery_frame_count - raw_frame_count
                    delivery_video = np.concatenate(
                        [raw_video, np.repeat(raw_video[-1:], pad_count, axis=0)],
                        axis=0,
                    )
                export_to_video(delivery_video, str(output_path), fps=output_fps)
                elapsed = round(time.time() - started, 3)
                append_status(
                    args.status_jsonl,
                    {
                        "case": name,
                        "status": "complete",
                        "output": str(output_path),
                        "num_chunk": num_chunk,
                        "num_frames_arg": num_frames,
                        "raw_frame_count": raw_frame_count,
                        "delivery_frame_count": len(delivery_video),
                        "output_fps": output_fps,
                        "seed": sample_seed,
                        "model_family": "wan2.2",
                        "model_id": args.model_id,
                        "action_checkpoint": args.ckpt_path,
                        "raw_output": (
                            str(raw_output_path) if raw_output_path is not None else None
                        ),
                        "elapsed_sec": elapsed,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                log(f"[{index}/{len(samples)}] complete {name} elapsed={elapsed}s")
        except Exception as exc:
            local_error = str(exc)
            local_traceback = traceback.format_exc()

        gathered_errors = [local_error]
        gathered_tracebacks = [local_traceback]
        if torch.distributed.is_initialized():
            gathered_errors = [None] * WORLD_SIZE
            gathered_tracebacks = [None] * WORLD_SIZE
            torch.distributed.all_gather_object(gathered_errors, local_error)
            torch.distributed.all_gather_object(gathered_tracebacks, local_traceback)
        if any(error is not None for error in gathered_errors):
            first_rank = next(
                rank for rank, error in enumerate(gathered_errors) if error is not None
            )
            if RANK == 0:
                append_status(
                    args.status_jsonl,
                    {
                        "case": name,
                        "status": "failed",
                        "error_rank": first_rank,
                        "error": gathered_errors[first_rank],
                        "traceback": gathered_tracebacks[first_rank],
                    },
                )
                print(gathered_tracebacks[first_rank], flush=True)
            raise RuntimeError(
                f"{name} failed on rank {first_rank}: {gathered_errors[first_rank]}"
            )

    log("All requested Wan2.2 videos complete")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
