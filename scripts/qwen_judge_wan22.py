"""Persistent Wan2.2 Qwen controller for the four released memory operations."""
import argparse
import base64
import atexit
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from hyvideo.utils.reproducibility import configure_qwen, audit_generation, record_runtime
from hyvideo.utils.qwen_results import decode_result, parse_result, skipped_result
configure_qwen()
record_runtime("wan22_qwen")
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
def strip_thinking(text):
    text = text.strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)
    return text.strip()
def add_chunk_images(content, title, chunk):
    if chunk.get("hide_metadata", False):
        content.append({"type": "text", "text": f"{title}:"})
        for path in chunk.get("frames", []):
            content.append({"type": "image", "url": str(path)})
        return
    chunk_i = chunk.get("chunk_i", chunk.get("id", "unknown"))
    cluster_id = chunk.get("cluster_id", None)
    prototype_status = chunk.get("prototype_status", None)
    visual_quality_score = chunk.get("visual_quality_score", None)
    meta = [f"chunk={chunk_i}"]
    if cluster_id is not None:
        meta.append(f"cluster={cluster_id}")
    if prototype_status is not None:
        meta.append(f"prototype_status={prototype_status}")
    if visual_quality_score is not None:
        meta.append(f"visual_quality_score={int(float(visual_quality_score))}")
    content.append({"type": "text", "text": f"{title} ({', '.join(meta)}), frames in temporal order:"})
    for path in chunk.get("frames", []):
        content.append({"type": "image", "url": str(path)})
def _load_image_from_source(source):
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    source = str(source)
    if source.startswith("file://"):
        source = source[len("file://") :]
    if source.startswith("data:image"):
        _, data = source.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    return Image.open(source).convert("RGB")
def _image_to_data_uri(img):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
def _spatialstack_select_worker_main(
    request_queue,
    response_queue,
    worker_config,
):
    device = worker_config["device"]
    try:
        judge = SpatialStackMemoryJudge(
            model_path=worker_config["model_path"],
            device=device,
            dtype=worker_config["dtype"],
            spatialstack_root=worker_config["spatialstack_root"],
            use_flash_attn2=worker_config["use_flash_attn2"],
            enable_linear_fast_path=worker_config["enable_linear_fast_path"],
            image_max_pixels=worker_config["image_max_pixels"],
            image_min_pixels=worker_config["image_min_pixels"],
            select_workers=1,
            serialize_generate=True,
            select_worker_devices=[],
        )
        response_queue.put({"type": "ready", "device": device})
    except Exception as exc:
        response_queue.put(
            {
                "type": "init_error",
                "device": device,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return
    while True:
        request = request_queue.get()
        if request is None or request.get("type") == "shutdown":
            return
        request_id = request.get("request_id")
        try:
            request_type = request.get("type")
            if request_type == "judge_task":
                start = time.perf_counter()
                raw, result = judge.judge(
                    request["task"],
                    request["payload"],
                    request.get("max_new_tokens", 256),
                    generation_options=request.get("generation_options", None),
                )
                response_queue.put(
                    {
                        "type": "result",
                        "request_id": request_id,
                        "device": device,
                        "raw": raw,
                        "result": result,
                        "elapsed": time.perf_counter() - start,
                    }
                )
            else:
                raise ValueError(f"Unsupported worker request type: {request_type}")
        except Exception as exc:
            response_queue.put(
                {
                    "type": "error",
                    "request_id": request_id,
                    "device": device,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
class SpatialStackSelectProcessWorker:
    def __init__(
        self,
        model_path,
        device,
        dtype,
        spatialstack_root,
        use_flash_attn2,
        enable_linear_fast_path,
        image_max_pixels,
        image_min_pixels,
    ):
        self.device = torch.device(device)
        self._ctx = mp.get_context("spawn")
        self._request_queue = self._ctx.Queue(maxsize=2)
        self._response_queue = self._ctx.Queue(maxsize=2)
        self._request_lock = threading.Lock()
        self._next_request_id = 0
        self._ready = False
        self._init_error = None
        worker_config = {
            "model_path": str(model_path),
            "device": str(device),
            "dtype": str(dtype),
            "spatialstack_root": str(spatialstack_root),
            "use_flash_attn2": bool(use_flash_attn2),
            "enable_linear_fast_path": bool(enable_linear_fast_path),
            "image_max_pixels": int(image_max_pixels),
            "image_min_pixels": int(image_min_pixels),
        }
        self._process = self._ctx.Process(
            target=_spatialstack_select_worker_main,
            args=(self._request_queue, self._response_queue, worker_config),
            daemon=True,
        )
        self._process.start()
    def _wait_until_ready(self):
        if self._ready:
            return
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        timeout = float(os.getenv("QWEN_MEMORY_SELECT_WORKER_READY_TIMEOUT", "900"))
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if not self._process.is_alive():
                raise RuntimeError(f"select worker {self.device} exited before becoming ready")
            try:
                message = self._response_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            message_type = message.get("type")
            if message_type == "ready":
                self._ready = True
                print(
                    f"[select-worker] process ready on {self.device}",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if message_type == "init_error":
                self._init_error = (
                    f"select worker {self.device} failed to initialize: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
                raise RuntimeError(self._init_error)
        raise TimeoutError(f"select worker {self.device} did not become ready within {timeout}s")
    def preload(self):
        self._wait_until_ready()
        return str(self.device)
    def judge_task(self, task, payload, max_new_tokens, generation_options=None):
        with self._request_lock:
            self._wait_until_ready()
            self._next_request_id += 1
            request_id = self._next_request_id
            self._request_queue.put(
                {
                    "type": "judge_task",
                    "request_id": request_id,
                    "task": str(task),
                    "payload": payload,
                    "max_new_tokens": int(max_new_tokens),
                    "generation_options": dict(generation_options or {}),
                }
            )
            timeout = float(os.getenv("QWEN_MEMORY_SELECT_WORKER_TIMEOUT", "600"))
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                if not self._process.is_alive():
                    raise RuntimeError(f"select worker {self.device} exited while running request")
                try:
                    message = self._response_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if message.get("request_id") != request_id:
                    continue
                if message.get("type") == "error":
                    raise RuntimeError(
                        f"select worker {self.device} failed: "
                        f"{message.get('error')}\n{message.get('traceback', '')}"
                    )
                return message["raw"], message["result"], message["elapsed"]
            raise TimeoutError(f"select worker {self.device} timed out after {timeout}s")
    def close(self):
        try:
            self._request_queue.put_nowait({"type": "shutdown"})
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
_TRAINING_PROMPT_DIR = Path(__file__).resolve().parent / "training"
from hyvideo.utils import qwen_prompts as hy15_prompts


HY15_PROMPT_PROFILE = "hy15_motion_groups_lastframe"


WAN22_PROMPT_PROFILE = "wan22_compact_turncross_strict_visualonly_20260905"


WAN22_CREATE_INSTRUCTION = (
    _TRAINING_PROMPT_DIR / "wan22_qwen_memory_create_compact_prompt.txt"
).read_text(encoding="utf-8")


WAN22_SPARSE_INSTRUCTION = (
    _TRAINING_PROMPT_DIR / "wan22_qwen_memory_sparse_compact_prompt.txt"
).read_text(encoding="utf-8")


WAN22_VISUAL_QUALITY_INSTRUCTION = (
    _TRAINING_PROMPT_DIR / "wan22_qwen_visual_quality_compact_prompt.txt"
).read_text(encoding="utf-8")


WAN22_SELECT_COVERAGE_INSTRUCTION = (
    _TRAINING_PROMPT_DIR / "wan22_qwen_memory_select_coverage_turncross_strict_prompt_noxyz_20260905.txt"
).read_text(encoding="utf-8").rstrip()


def _memory_prompt_profile():
    profile = str(os.getenv("QWEN_MEMORY_PROMPT_PROFILE", WAN22_PROMPT_PROFILE)).strip().lower()
    if profile not in {WAN22_PROMPT_PROFILE, HY15_PROMPT_PROFILE}:
        raise ValueError(f"Unsupported Qwen prompt profile: {profile}")
    return profile


def _uses_wan22_compact_prompts():
    return _memory_prompt_profile() == WAN22_PROMPT_PROFILE


def _active_create_instruction():
    return WAN22_CREATE_INSTRUCTION if _uses_wan22_compact_prompts() else hy15_prompts.CREATE


def _active_sparse_instruction():
    return WAN22_SPARSE_INSTRUCTION if _uses_wan22_compact_prompts() else hy15_prompts.SPARSE


def _active_visual_quality_instruction():
    return WAN22_VISUAL_QUALITY_INSTRUCTION if _uses_wan22_compact_prompts() else hy15_prompts.VISUAL_QUALITY


def _active_select_coverage_instruction():
    return WAN22_SELECT_COVERAGE_INSTRUCTION if _uses_wan22_compact_prompts() else hy15_prompts.COVERAGE


def _coverage_prompt_audit():
    instruction = _active_select_coverage_instruction()
    sample_content = _build_select_content(
        {
            "current_chunk": {"chunk_i": 11, "frames": ["/audit/current.png"]},
            "current_chunk_i": 12,
            "planned_camera_motion": (
                "camera_motion=turn_right; "
                "local_translation(right=0.0000, up=0.0000, forward=0.0000); "
                "local_rotation(yaw=12.0000deg, pitch=0.0000deg, roll=0.0000deg)"
            ),
            "coverage_candidates": [
                {
                    "option_label": "A",
                    "chunk_i": 3,
                    "visual_quality_score": 8,
                    "frames": ["/audit/candidate_a.png"],
                }
            ],
        },
        instruction,
        "coverage_candidates",
    )
    rendered_text = "\n".join(
        str(item.get("text", ""))
        for item in sample_content
        if item.get("type") == "text"
    )
    forbidden_phrases = (
        "Reference for current " + "motion=",
        "exact user-labeled examples available:",
        "Use the provided user manual reference examples",
        "related-motion guidance",
    )
    return {
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
        "rendered_text_chars": len(rendered_text),
        "forbidden_hits": [phrase for phrase in forbidden_phrases if phrase in rendered_text],
        "coverage_xyz_enabled": False,
    }


def _json_compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _select_coverage_max_candidates():
    """Return the profile-specific coverage cap shared by retry and parsing."""
    profile_cap = 3 if _uses_wan22_compact_prompts() else 2
    try:
        requested = int(
            os.getenv("QWEN_MEMORY_SELECT_TOP_K", str(profile_cap))
        )
        return max(1, min(profile_cap, requested))
    except Exception:
        return profile_cap


def _motion_key_from_text(motion_text):
    text = str(motion_text or "").strip()
    marker = "camera_motion="
    if marker in text:
        value = text.split(marker, 1)[1].split(";", 1)[0].strip()
        return value or text
    return text


def _is_exact_turncross_motion(motion_text):
    return _motion_key_from_text(motion_text) in {
        "turn_left+move_right",
        "turn_right+move_left",
    }


def _validate_wan22_compact_raw_coverage_result(
    result,
    candidates,
    planned_camera_motion="",
):
    if not _uses_wan22_compact_prompts():
        return
    if not isinstance(result, dict) or set(result) != {
        "motion", "acceptable", "selected"
    }:
        raise ValueError(
            "WAN2.2 compact coverage output must contain exactly "
            "motion, acceptable, and selected"
        )
    valid_options = [
        str(item.get("option_label", chr(ord("A") + idx)))
        for idx, item in enumerate(candidates or [])
    ]
    valid_option_set = set(valid_options)
    motion = result.get("motion")
    if not isinstance(motion, dict) or set(motion) != {
        "supports", "neutral", "opposes"
    }:
        raise ValueError(
            "WAN2.2 compact motion must contain exactly "
            "supports, neutral, and opposes"
        )
    flattened = []
    for group in ("supports", "neutral", "opposes"):
        options = motion[group]
        if not isinstance(options, list):
            raise ValueError(f"WAN2.2 compact motion.{group} must be a list")
        flattened.extend(str(option).strip() for option in options)
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != valid_option_set
    ):
        raise ValueError(
            "WAN2.2 compact motion groups must partition every displayed "
            "candidate exactly once"
        )
    acceptable = result.get("acceptable")
    selected = result.get("selected")
    if (
        not isinstance(acceptable, list)
        or not isinstance(selected, list)
    ):
        raise ValueError(
            "WAN2.2 compact acceptable and selected must both be lists"
        )
    acceptable = [str(option).strip() for option in acceptable]
    selected = [str(option).strip() for option in selected]
    if (
        len(acceptable) != len(set(acceptable))
        or not set(acceptable).issubset(valid_option_set)
    ):
        raise ValueError(
            "WAN2.2 compact acceptable contains duplicate or unknown options"
        )
    if (
        len(selected) != len(set(selected))
        or not set(selected).issubset(set(acceptable))
        or len(selected) > _select_coverage_max_candidates()
    ):
        raise ValueError(
            "WAN2.2 compact selected must be a unique subset of acceptable "
            "with at most max_selected options"
        )
    if _uses_wan22_compact_prompts() and set(selected) & {
        str(option).strip() for option in motion["opposes"]
    }:
        raise ValueError(
            "WAN2.2 turn-cross selected must not contain an option classified "
            "as opposes"
        )
    if (
        _uses_wan22_compact_prompts()
        and _is_exact_turncross_motion(planned_camera_motion)
        and set(acceptable)
        & {str(option).strip() for option in motion["opposes"]}
    ):
        raise ValueError(
            "WAN2.2 strict turn-cross acceptable must not contain an option "
            "classified as opposes"
        )


def _decode_memory_result(task, raw, payload):
    candidates = payload.get("coverage_candidates") or payload.get("candidates") or []
    result = decode_result(task, raw, candidates, _select_coverage_max_candidates())
    if task == "select_coverage" and not result.get("skipped"):
        try:
            _validate_wan22_compact_raw_coverage_result(
                parse_result(raw), candidates,
                payload.get("planned_camera_motion", payload.get("target_motion_text", "")))
        except (ValueError, TypeError, KeyError) as exc:
            return skipped_result(exc)
    return result


def _build_visual_quality_content(payload):
    candidate = (
        payload.get("generated_chunk", None)
        or payload.get("candidate", None)
        or payload.get("current_chunk", None)
        or {}
    )
    content = [{"type": "text", "text": _active_visual_quality_instruction()}]
    add_chunk_images(content, "Generated frame", candidate)
    return content


def _build_create_content(payload):
    current = payload.get("current_chunk", {}) or {}
    candidate_class = payload.get("candidate_class", {}) or {}
    prototype = payload.get("prototype", None) or candidate_class.get("prototype", {}) or {}
    class_id = payload.get("class_id", candidate_class.get("cluster_id", None))
    try:
        prototype_fov_overlap = float(
            payload.get(
                "prototype_fov_overlap",
                candidate_class.get("prototype_fov_overlap", prototype.get("fov_overlap", 0.0)),
            )
            or 0.0
        )
    except Exception:
        prototype_fov_overlap = 0.0
    content = [
        {
            "type": "text",
            "text": (
                f"{_active_create_instruction()}\n\n"
                f"Current chunk_id={current.get('chunk_i', 'unknown')}\n"
                f"Candidate class_id={class_id}\n"
                f"Prototype chunk_id={prototype.get('chunk_i', 'unknown')}\n"
                f"prototype_fov_overlap={prototype_fov_overlap:.4f}"
            ),
        }
    ]
    add_chunk_images(content, "Current middle frame", current)
    add_chunk_images(content, "Prototype middle frame", prototype)
    return content


def _build_sparse_content(payload):
    candidate = payload.get("candidate_chunk", {}) or {}
    remaining = payload.get("remaining_chunks", []) or []
    if "coverage_by_remaining_chunks" not in payload:
        raise ValueError("sparse payload missing coverage_by_remaining_chunks")
    try:
        coverage_by_remaining_chunks = float(payload["coverage_by_remaining_chunks"])
    except Exception as exc:
        raise ValueError(
            f"invalid sparse coverage_by_remaining_chunks: {payload.get('coverage_by_remaining_chunks')!r}"
        ) from exc
    content = [
        {
            "type": "text",
            "text": (
                f"{_active_sparse_instruction()}\n\n"
                f"class_id={payload.get('cluster_id', payload.get('class_id', None))}\n"
                f"candidate_chunk_id={candidate.get('chunk_i', payload.get('candidate_chunk_i', 'unknown'))}\n"
                f"remaining_chunk_ids={_json_compact(payload.get('remaining_chunk_ids', []))}\n"
                f"coverage_by_remaining_chunks={coverage_by_remaining_chunks:.4f}\n"
                f"scenario_assumed_removed_chunk_ids={_json_compact(payload.get('scenario_assumed_removed_chunk_ids', []))}"
            ),
        }
    ]
    for chunk in remaining:
        add_chunk_images(content, "Remaining class middle frame", chunk)
    add_chunk_images(content, "Candidate removal middle frame", candidate)
    return content


def _build_select_content(payload, instruction, candidate_key):
    current = (
        payload.get("current_chunk", None)
        or (payload.get("local_chunks", []) or [{}])[-1]
        or {}
    )
    candidates = payload.get(candidate_key, None) or payload.get("candidates", []) or []
    motion_text = payload.get("planned_camera_motion", payload.get("target_motion_text", ""))
    if candidate_key == "coverage_candidates":
        target_next_chunk_id = payload.get(
            "target_next_chunk_id",
            payload.get("current_chunk_i", "unknown"),
        )
        lines = [instruction, ""]
        lines.extend(
            [
                f"current_chunk_id: {current.get('chunk_i', 'unknown')}",
                f"target_next_chunk_id: {target_next_chunk_id}",
                f"Planned camera motion: {motion_text}",
                "Candidates:",
            ]
        )
        for idx, candidate in enumerate(candidates):
            option = str(candidate.get("option_label", chr(ord("A") + idx)))
            if _uses_wan22_compact_prompts():
                lines.append(f"{option}: chunk_id={candidate.get('chunk_i')}")
            else:
                visual_quality_score = candidate.get("visual_quality_score", None)
                lines.append(
                    f"{option}: chunk_id={candidate.get('chunk_i')} "
                    f"visual_quality_score={visual_quality_score}"
                )
        lines.append("Image: Current")
        for idx, candidate in enumerate(candidates):
            option = str(candidate.get("option_label", chr(ord("A") + idx)))
            lines.append(f"Image: Candidate-{option}")
    else:
        lines = [
            instruction,
            "",
            f"Current chunk_id={current.get('chunk_i', 'unknown')}",
            f"Planned camera motion: {motion_text}",
            "Candidates:",
        ]
        for idx, candidate in enumerate(candidates):
            option = str(candidate.get("option_label", chr(ord("A") + idx)))
            visual_quality_score = candidate.get("visual_quality_score", None)
            class_id = candidate.get("class_id", candidate.get("cluster_id", None))
            class_text = "null" if class_id is None else str(class_id)
            lines.append(
                (
                    f"- option={option}; chunk_id={candidate.get('chunk_i')}; "
                    f"class_id={class_text}; "
                    f"visual_quality_score={visual_quality_score}"
                )
            )
    text = "\n".join(lines)
    if candidate_key == "coverage_candidates" and _uses_wan22_compact_prompts():
        if any(marker in text.lower() for marker in ("xyz", "x/y/z", "centroid")):
            raise ValueError("No-XYZ coverage input contains coordinate guidance")
    content = [{"type": "text", "text": text}]
    add_chunk_images(content, "Current frame", {**current, "hide_metadata": True})
    for idx, candidate in enumerate(candidates):
        option = str(candidate.get("option_label", chr(ord("A") + idx)))
        add_chunk_images(content, f"Candidate-{option}", {**candidate, "hide_metadata": True})
    return content


class SpatialStackMemoryJudge:
    def __init__(
        self,
        model_path,
        device,
        dtype,
        spatialstack_root=str(Path(__file__).resolve().parents[1] / "vendor/SpatialStack"),
        use_flash_attn2=True,
        enable_linear_fast_path=True,
        image_max_pixels=576 * 28 * 28,
        image_min_pixels=16 * 28 * 28,
        select_workers=1,
        serialize_generate=True,
        select_worker_devices=None,
    ):
        self.model_path = str(Path(model_path))
        self.device = torch.device(device)
        self.image_max_pixels = int(image_max_pixels)
        self.image_min_pixels = int(image_min_pixels)
        self.select_workers = max(1, int(select_workers))
        self.serialize_generate = bool(serialize_generate)
        self._generate_lock = threading.Lock() if self.serialize_generate else None
        self.select_worker_devices = [
            str(x).strip()
            for x in (select_worker_devices or [])
            if str(x).strip()
        ]
        self.main_judge_tasks = {
            item.strip()
            for item in str(
                os.getenv("QWEN_MEMORY_MAIN_JUDGE_TASKS", "")
            ).split(",")
            if item.strip()
        }
        self.select_worker_judges = []
        self._sampling_generate_lock = threading.Lock()
        self._select_worker_warmup_lock = threading.Lock()
        self._select_workers_warmed = False
        self._worker_warmup_lock = threading.Lock()
        self._worker_warmup_started = False
        self._worker_warmup_done = False
        self._worker_warmup_error = None
        self._worker_warmup_elapsed = 0.0
        self._worker_warmup_thread = None
        self._select_process_workers = []

        spatialstack_root = str(Path(spatialstack_root))
        if spatialstack_root not in sys.path:
            sys.path.insert(0, spatialstack_root)

        from scripts.inference.infer import (
            patch_qwen3_5_flash_attention, patch_qwen3_5_linear_attention_fallback,
            prepare_raw_visual_inputs, resolve_model_class, resolve_model_family,
        )
        self._prepare_raw_visual_inputs = prepare_raw_visual_inputs

        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        self.model_family = resolve_model_family(config)
        if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
            raise ValueError("The released controller requires a visual Qwen3.5 checkpoint")
        self.use_geometry_model = False
        model_class = resolve_model_class(self.model_family, self.use_geometry_model)
        if self.model_family == "qwen3_5":
            patch_qwen3_5_flash_attention()
            if not bool(enable_linear_fast_path):
                patch_qwen3_5_linear_attention_fallback()

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "config": config,
            "torch_dtype": torch_dtype,
            "device_map": str(self.device),
        }
        if bool(use_flash_attn2):
            self.model = model_class.from_pretrained(
                **model_kwargs,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self.model = model_class.from_pretrained(**model_kwargs).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            max_pixels=self.image_max_pixels,
            min_pixels=self.image_min_pixels,
            padding_side="left",
        )

        for worker_device in self.select_worker_devices:
            if str(torch.device(worker_device)) == str(self.device):
                print(
                    f"[select-worker] reusing main judge on {worker_device} for memory operations",
                    file=sys.stderr,
                    flush=True,
                )
                self.select_worker_judges.append(self)
                continue
            print(
                f"[select-worker] starting spatialstack controller process on {worker_device}",
                file=sys.stderr,
                flush=True,
            )
            worker = SpatialStackSelectProcessWorker(
                model_path=self.model_path,
                device=worker_device,
                dtype=dtype,
                spatialstack_root=spatialstack_root,
                use_flash_attn2=use_flash_attn2,
                enable_linear_fast_path=enable_linear_fast_path,
                image_max_pixels=self.image_max_pixels,
                image_min_pixels=self.image_min_pixels,
            )
            self._select_process_workers.append(worker)
            self.select_worker_judges.append(worker)
        if self.select_worker_judges:
            print(
                f"[select-worker] configured {len(self.select_worker_judges)} controller workers",
                file=sys.stderr,
                flush=True,
            )
            if self.main_judge_tasks:
                print(
                    "[select-worker] tasks pinned to main judge: "
                    + ",".join(sorted(self.main_judge_tasks)),
                    file=sys.stderr,
                    flush=True,
                )
            atexit.register(self.close_select_workers)
        if (
            self._select_process_workers
            and str(os.getenv("QWEN_MEMORY_SELECT_PRELOAD_WORKERS", "0")).strip().lower()
            in {"1", "true", "yes"}
        ):
            start = time.perf_counter()
            print(
                f"[select-worker] preloading {len(self._select_process_workers)} process workers",
                file=sys.stderr,
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=len(self._select_process_workers)) as executor:
                futures = [
                    executor.submit(worker.preload)
                    for worker in self._select_process_workers
                ]
                for future in as_completed(futures):
                    device = future.result()
                    print(
                        f"[select-worker] preload done on {device}",
                        file=sys.stderr,
                        flush=True,
                    )
            print(
                f"[select-worker] preload all done elapsed={time.perf_counter() - start:.3f}s",
                file=sys.stderr,
                flush=True,
            )

    def close_select_workers(self):
        for worker in getattr(self, "_select_process_workers", []):
            try:
                worker.close()
            except Exception:
                pass

    def _warmup_image_uri(self, offset=0):
        img = Image.new("RGB", (518, 518), (128, 128, 128))
        x0 = 150 + int(offset)
        img.paste((235, 235, 235), (x0, 90, x0 + 120, 430))
        img.paste((35, 35, 35), (x0 + 170, 170, x0 + 230, 350))
        return _image_to_data_uri(img)

    def _warmup_single_runner(self, runner, idx):
        uri = self._warmup_image_uri(idx)
        chunk = {"chunk_i": 0, "frames": [uri]}
        requests = [
            ("visual_quality", {"generated_chunk": chunk}, 64),
            ("create", {"current_chunk": chunk, "prototype": chunk, "cluster_id": idx,
                        "prototype_fov_overlap": 1.0}, 128),
            ("sparse", {"cluster_id": idx, "candidate_chunk_i": 0,
                        "candidate_chunk": chunk, "remaining_chunks": [chunk],
                        "remaining_chunk_ids": [0], "coverage_by_remaining_chunks": 1.0}, 128),
            ("select_coverage", {"current_chunk_i": 1, "current_chunk": chunk,
                                 "coverage_candidates": [dict(chunk, option_label="A")],
                                 "planned_camera_motion": "camera_motion=turn_right"}, 128),
        ]
        for task, payload, tokens in requests:
            runner.judge_task(task, payload, tokens)

    def warmup_workers(self):
        with self._worker_warmup_lock:
            if self._worker_warmup_done:
                return
            if self._worker_warmup_error is not None:
                raise RuntimeError(self._worker_warmup_error)
            self._worker_warmup_started = True

        start = time.perf_counter()
        try:
            runners = list(self.select_worker_judges) if self.select_worker_judges else [self]
            print(
                f"[worker-warmup] warming {len(runners)} workers",
                file=sys.stderr,
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=max(1, len(runners))) as executor:
                futures = [
                    executor.submit(self._warmup_single_runner, runner, idx)
                    for idx, runner in enumerate(runners)
                ]
                for future in as_completed(futures):
                    future.result()
            elapsed = time.perf_counter() - start
            with self._worker_warmup_lock:
                self._worker_warmup_done = True
                self._worker_warmup_elapsed = float(elapsed)
            print(
                f"[worker-warmup] all workers done elapsed={elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            error = f"{repr(exc)}\n{traceback.format_exc()}"
            with self._worker_warmup_lock:
                self._worker_warmup_error = error
            print(f"[worker-warmup] failed: {error}", file=sys.stderr, flush=True)
            raise

    def start_worker_warmup(self):
        with self._worker_warmup_lock:
            if self._worker_warmup_started or self._worker_warmup_done:
                return
            self._worker_warmup_started = True
            self._worker_warmup_thread = threading.Thread(
                target=self.warmup_workers,
                name="spatialstack-worker-warmup",
                daemon=True,
            )
            self._worker_warmup_thread.start()

    def skip_worker_warmup(self):
        with self._worker_warmup_lock:
            self._worker_warmup_started = False
            self._worker_warmup_done = True
            self._worker_warmup_error = None
            self._worker_warmup_elapsed = 0.0
        print("[worker-warmup] skipped by QWEN_MEMORY_SKIP_WORKER_WARMUP", file=sys.stderr, flush=True)

    def warmup_status(self):
        with self._worker_warmup_lock:
            return {
                "started": bool(self._worker_warmup_started),
                "done": bool(self._worker_warmup_done),
                "error": self._worker_warmup_error,
                "elapsed_sec": float(self._worker_warmup_elapsed),
                "workers": len(self.select_worker_judges) if self.select_worker_judges else 1,
            }

    def judge_task(self, task, payload, max_new_tokens, generation_options=None):
        start = time.perf_counter()
        raw, result = self.judge(
            task,
            payload,
            max_new_tokens,
            generation_options=generation_options,
        )
        return raw, result, time.perf_counter() - start

    def _content_to_messages(self, content, system_prompt="You are a careful visual reasoning assistant."):
        converted = []
        for item in content:
            if item.get("type") == "text":
                converted.append({"type": "text", "text": str(item.get("text", ""))})
            elif item.get("type") == "image":
                source = item.get("image", item.get("url", None))
                if source is None:
                    continue
                converted.append(
                    {
                        "type": "image",
                        "image": _image_to_data_uri(_load_image_from_source(source)),
                    }
                )
        return [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": converted},
        ]]

    def _generate_from_content(
        self,
        content,
        max_new_tokens,
        system_prompt=None,
        generation_options=None,
    ):
        generation_options = dict(generation_options or {})
        messages = self._content_to_messages(
            content,
            system_prompt=system_prompt or "You are a careful visual reasoning assistant.",
        )
        chat_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.model_family == "qwen3_5":
            chat_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(messages, **chat_kwargs)

        raw_image_inputs = self._prepare_raw_visual_inputs(messages)
        image_inputs = raw_image_inputs
        model_inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.device)

        do_sample = bool(generation_options.get("do_sample", False))
        if do_sample:
            raise ValueError("SMI reproducible inference requires do_sample=False")
        temperature = max(0.01, float(generation_options.get("temperature", 0.7)))
        top_p = max(0.01, min(1.0, float(generation_options.get("top_p", 0.9))))
        seed = generation_options.get("seed", None)

        def _do_generate():
            if do_sample and seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            generate_kwargs = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": do_sample,
                "max_new_tokens": int(max_new_tokens),
                "use_cache": True,
            }
            if do_sample:
                generate_kwargs.update(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
            with torch.inference_mode():
                return self.model.generate(
                    **model_inputs,
                    **generate_kwargs,
                )

        start = time.perf_counter()
        lock = self._sampling_generate_lock if do_sample else self._generate_lock
        if lock is not None:
            with lock:
                output_ids = _do_generate()
        else:
            output_ids = _do_generate()
        elapsed = time.perf_counter() - start
        generated_ids = output_ids[0][model_inputs.input_ids.shape[-1] :]
        audit_generation(model_inputs, generated_ids, "wan22_spatialstack")
        answer = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return strip_thinking(answer), elapsed

    def _judge_via_update_worker(self, task, payload, max_new_tokens):
        if str(task).strip() in self.main_judge_tasks:
            return None
        if not self._select_process_workers:
            return None
        if str(task) == "sparse":
            branch_ids = payload.get("scenario_assumed_removed_chunk_ids", []) or []
            branch_key = 0
            for idx, chunk_id in enumerate(branch_ids):
                try:
                    branch_key += (idx + 1) * int(chunk_id)
                except Exception:
                    continue
            try:
                key = (
                    int(payload.get("candidate_chunk_i", 0) or 0) * 1009
                    + int(payload.get("member_position", 0) or 0) * 97
                    + int(payload.get("member_from_end", 0) or 0) * 13
                    + branch_key
                )
            except Exception:
                key = int(payload.get("cluster_id", 0) or 0)
        elif str(task) == "select_coverage":
            try:
                key = int(payload.get("current_chunk_i", 0) or 0) * 17
            except Exception:
                key = 0
        else:
            key = int(payload.get("cluster_id", 0) or 0)
        start_index = key % len(self._select_process_workers)
        worker_errors = []
        for offset in range(len(self._select_process_workers)):
            runner = self._select_process_workers[
                (start_index + offset) % len(self._select_process_workers)
            ]
            try:
                raw, result, elapsed = runner.judge_task(task, payload, max_new_tokens)
            except Exception as exc:
                worker_errors.append(
                    {
                        "device": str(getattr(runner, "device", "worker")),
                        "error": repr(exc)[:500],
                    }
                )
                continue
            if isinstance(result, dict):
                result = dict(result)
                result["worker_device"] = str(getattr(runner, "device", "worker"))
                result["worker_elapsed_sec"] = float(elapsed)
                if worker_errors:
                    result["worker_failover_count"] = len(worker_errors)
                    result["worker_failures"] = worker_errors
            return raw, result
        raise RuntimeError(
            f"all select workers failed for task={task}: {worker_errors}"
        )

    def judge(self, task, payload, max_new_tokens, generation_options=None):
        generation_options = dict(generation_options or {})
        if task in {"visual_quality", "create", "sparse", "select_coverage"}:
            if task != "visual_quality":
                worker_result = self._judge_via_update_worker(task, payload, max_new_tokens)
                if worker_result is not None:
                    return worker_result
            if task == "select_coverage":
                content = _build_select_content(
                    payload, _active_select_coverage_instruction(), "coverage_candidates")
            else:
                content = {"visual_quality": _build_visual_quality_content,
                           "create": _build_create_content, "sparse": _build_sparse_content}[task](payload)
            kwargs = dict(max_new_tokens=max_new_tokens, generation_options=generation_options)
            if task == "visual_quality":
                kwargs["system_prompt"] = "You are a strict visual quality scorer."
            raw, _elapsed = self._generate_from_content(content, **kwargs)
            return raw, _decode_memory_result(task, raw, payload)
        raise ValueError(f"Unsupported memory operation: {task}")


def make_handler(judge, default_max_new_tokens):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SMI-Wan22-Qwen/1.0"

        def _write_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                prompt_audit = _coverage_prompt_audit()
                warmup_status = (
                    judge.warmup_status()
                    if hasattr(judge, "warmup_status")
                    else {"done": True, "started": True, "error": None}
                )
                if warmup_status.get("error"):
                    self._write_json(
                        500,
                        {
                            "ok": False,
                            "model_path": judge.model_path,
                            "prompt_profile": _memory_prompt_profile(),
                            "coverage_prompt_audit": prompt_audit,
                            "warmup": warmup_status,
                        },
                    )
                elif not warmup_status.get("done", True):
                    self._write_json(
                        503,
                        {
                            "ok": False,
                            "model_path": judge.model_path,
                            "prompt_profile": _memory_prompt_profile(),
                            "coverage_prompt_audit": prompt_audit,
                            "warmup": warmup_status,
                        },
                    )
                else:
                    self._write_json(
                        200,
                        {
                            "ok": True,
                            "model_path": judge.model_path,
                            "prompt_profile": _memory_prompt_profile(),
                            "coverage_prompt_audit": prompt_audit,
                            "warmup": warmup_status,
                        },
                    )
            else:
                self._write_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/judge":
                self._write_json(404, {"ok": False, "error": "not found"})
                return
            request = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                task = request.get("task")
                payload = request.get("payload", {})
                max_new_tokens = int(request.get("max_new_tokens", default_max_new_tokens))
                generation_options = request.get("generation_options", {}) or {}
                start = time.perf_counter()
                raw, result = judge.judge(
                    task,
                    payload,
                    max_new_tokens=max_new_tokens,
                    generation_options=generation_options,
                )
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "task": task,
                        "result": result,
                        "raw": raw,
                        "generation_options": generation_options,
                        "elapsed_sec": time.perf_counter() - start,
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                failed_request_root = os.getenv("QWEN_MEMORY_FAILED_REQUEST_DIR")
                if failed_request_root and request is not None:
                    try:
                        directory = Path(failed_request_root)
                        directory.mkdir(parents=True, exist_ok=True)
                        record_path = directory / f"failed_{time.time_ns()}_{os.getpid()}.json"
                        with record_path.open("x", encoding="utf-8") as stream:
                            json.dump({"request": request, "error": repr(exc)}, stream, ensure_ascii=False, indent=2)
                        print(f"[failed-request] {record_path}", flush=True)
                    except Exception:
                        traceback.print_exc()
                payload = {"ok": False, "error": repr(exc)}
                raw = getattr(exc, "raw", None)
                if raw is not None:
                    payload["raw"] = raw
                self._write_json(500, payload)

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    return Handler


def _exit_on_shutdown(signum, frame):
    # Python's default SIGTERM bypasses finally/atexit and orphans CUDA workers.
    raise SystemExit(128 + signum)


def serve_until_stopped(server, judge):
    try:
        server.serve_forever()
    finally:
        try:
            server.server_close()
        finally:
            if hasattr(judge, "close_select_workers"):
                judge.close_select_workers()


def main():
    signal.signal(signal.SIGTERM, _exit_on_shutdown)
    parser = argparse.ArgumentParser(description="Persistent Qwen3.5 visual memory judge server.")
    parser.add_argument(
        "--backend",
        choices=["spatialstack"],
        default="spatialstack",
    )
    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument(
        "--spatialstack-root",
        default=str(Path(__file__).resolve().parents[1] / "vendor/SpatialStack"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7869)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.getenv("QWEN_MEMORY_IMAGE_MAX_PIXELS", 576 * 28 * 28)),
        help="Per-image processor pixel-area ceiling; keep identical to training MAX_PIXELS.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=int(os.getenv("QWEN_MEMORY_IMAGE_MIN_PIXELS", 16 * 28 * 28)),
        help="Per-image processor pixel-area floor; keep identical to training MIN_PIXELS.",
    )
    parser.add_argument("--select-workers", type=int, default=1)
    parser.add_argument(
        "--select-worker-devices",
        default="",
        help="Comma-separated devices for SpatialStack memory operations workers, e.g. cuda:0,cuda:1",
    )
    parser.add_argument("--no-flash-attn2", action="store_true")
    parser.add_argument("--disable-linear-fast-path", action="store_true")
    parser.add_argument("--parallel-generate", action="store_true")
    args = parser.parse_args()

    if args.image_min_pixels <= 0 or args.image_max_pixels < args.image_min_pixels:
        parser.error("--image-max-pixels must be >= --image-min-pixels > 0")

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is not available for Qwen3.5 judge server.")

    if args.backend == "spatialstack":
        judge = SpatialStackMemoryJudge(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            spatialstack_root=args.spatialstack_root,
            use_flash_attn2=not args.no_flash_attn2,
            enable_linear_fast_path=not args.disable_linear_fast_path,
            image_max_pixels=args.image_max_pixels,
            image_min_pixels=args.image_min_pixels,
            select_workers=args.select_workers,
            serialize_generate=not args.parallel_generate,
            select_worker_devices=[
                x.strip()
                for x in str(args.select_worker_devices).split(",")
                if x.strip()
            ],
        )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(judge, args.max_new_tokens),
    )
    if (
        hasattr(judge, "skip_worker_warmup")
        and str(os.getenv("QWEN_MEMORY_SKIP_WORKER_WARMUP", "0")).strip().lower()
        in {"1", "true", "yes"}
    ):
        judge.skip_worker_warmup()
    elif hasattr(judge, "start_worker_warmup"):
        judge.start_worker_warmup()
    print(
        f"{args.backend} memory judge server listening on http://{args.host}:{args.port}/judge",
        flush=True,
    )
    serve_until_stopped(server, judge)


if __name__ == "__main__":
    main()
