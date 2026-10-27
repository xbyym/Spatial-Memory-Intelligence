import html

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Any, Callable, Dict, List, Optional, Union

import os
import time
import urllib.error
import urllib.request
import regex as re
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel
import transformers.utils as _transformers_utils

if not hasattr(_transformers_utils, "FLAX_WEIGHTS_NAME"):
    _transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from models.utils import (
    shard_latents_dim_across_sp,
    select_mem_frames_wan,
)
from hyvideo.utils.retrieval_context import (
    calculate_fov_overlap_similarity,
    generate_points_in_sphere,
)
from distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_parallel_local_rank,
    get_sp_world_size,
)
from distributed.communication_op import sequence_model_parallel_all_gather
from models.par_vae.tools import distribute_data_to_gpus, gather_results
from inference.helper import CHUNK_SIZE

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy
else:
    ftfy = None


def basic_clean(text):
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = (
            self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        if os.getenv("WORLD_SIZE", "0") != "0":
            self.parallel_run = True
        else:
            self.parallel_run = False
        self.par_vae_encode = False
        self.par_vae_decode = self.parallel_run and True
        self.wan_memory_fov_top_k = 1
        self.qwen_memory_judge_enable = False
        self.qwen_memory_judge_url = None
        self.qwen_memory_judge_model_path = "/opt/liblibai-models/model-weights/Qwen/Qwen3.5-4B"
        self.qwen_memory_judge_device = None
        self.qwen_memory_judge_dtype = "bfloat16"
        self.qwen_memory_select_fov_top_k = 7
        self.qwen_memory_select_top_k = 3
        self.qwen_memory_local_context_chunks = 2
        self.qwen_memory_perspective = "first_person"
        self.qwen_memory_fov_fill_min_chunks = 0
        self.qwen_memory_allow_fallback = False
        self.qwen_memory_frames_per_chunk = 1
        self.qwen_memory_request_timeout_sec = 180.0
        self.qwen_memory_select_do_sample = False
        self.qwen_memory_select_temperature = 0.7
        self.qwen_memory_select_top_p = 0.9
        self.qwen_memory_select_seed = None
        self.qwen_memory_select_policy = "coverage_only"
        self.qwen_memory_visual_quality_enable = True
        self.qwen_memory_intermediate_chunks_dir = None
        self.qwen_memory_update_enable = False
        self.qwen_memory_update_commit_mode = "immediate"
        self.qwen_memory_update_fov_top_k = 6
        self.qwen_memory_class_capacity = 7
        self.qwen_memory_member_batch_size = 8
        self.qwen_memory_confirm_prototype_min_chunks = 3
        self.qwen_memory_prototype_min_fov_overlap = 0.0
        self.qwen_memory_drop_min_fov_overlap = 0.9
        self.qwen_memory_disable_sparse = False
        self.qwen_memory_sparse_member_from_ends = (-2, -3)
        self.qwen_memory_update_min_vq = 6
        self.qwen_memory_select_active_only = True
        self.qwen_memory_select_filter_visual_quality = True
        self.qwen_memory_strict = False
        self._qwen_memory_judge = None
        self._qwen_memory_records = {}
        self._qwen_memory_classes = {}
        self._qwen_chunk_to_cluster = {}
        self._qwen_active_memory_chunk_ids = set()
        self._qwen_update_candidate_scores_by_chunk = {}
        self._qwen_next_cluster_id = 0

    def _is_rank0(self):
        return (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_rank() == 0
        )

    def _qwen_log(self, message):
        if self._is_rank0():
            print(message, flush=True)

    def configure_qwen_memory(
        self,
        judge_enable=False,
        judge_url=None,
        judge_model_path=None,
        judge_device=None,
        judge_dtype="bfloat16",
        select_fov_top_k=7,
        select_top_k=3,
        local_context_chunks=2,
        perspective="first_person",
        fov_fill_min_chunks=0,
        allow_fallback=False,
        frames_per_chunk=1,
        request_timeout_sec=180.0,
        select_do_sample=False,
        select_temperature=0.7,
        select_top_p=0.9,
        select_seed=None,
        select_policy="coverage_only",
        visual_quality_enable=True,
        intermediate_chunks_dir=None,
        update_enable=False,
        update_fov_top_k=6,
        class_capacity=7,
        member_batch_size=8,
        confirm_prototype_min_chunks=3,
        prototype_min_fov_overlap=0.0,
        drop_min_fov_overlap=0.9,
        disable_sparse=False,
        sparse_member_from_ends=None,
        select_active_only=True,
        select_filter_visual_quality=True,
        strict=False,
    ):
        self.qwen_memory_judge_enable = bool(judge_enable)
        self.qwen_memory_judge_url = (
            str(judge_url).strip() if judge_url is not None else None
        )
        if self.qwen_memory_judge_url in ("", "none", "None", "null", "NULL"):
            self.qwen_memory_judge_url = None
        if judge_model_path is not None:
            self.qwen_memory_judge_model_path = str(judge_model_path).strip()
        self.qwen_memory_judge_device = (
            str(judge_device).strip() if judge_device is not None else None
        )
        if self.qwen_memory_judge_device in ("", "none", "None", "null", "NULL"):
            self.qwen_memory_judge_device = None
        self.qwen_memory_judge_dtype = str(judge_dtype or "bfloat16").strip()
        if self.qwen_memory_judge_dtype not in ("bfloat16", "float16"):
            self.qwen_memory_judge_dtype = "bfloat16"
        self.qwen_memory_select_fov_top_k = max(1, int(select_fov_top_k))
        self.qwen_memory_select_top_k = max(0, int(select_top_k))
        self.qwen_memory_local_context_chunks = max(1, int(local_context_chunks))
        perspective = str(perspective or "first_person").strip().lower()
        if perspective not in {"first_person", "third_person"}:
            raise ValueError(f"unsupported Qwen coverage perspective: {perspective}")
        self.qwen_memory_perspective = perspective
        self.qwen_memory_fov_fill_min_chunks = max(0, int(fov_fill_min_chunks))
        if allow_fallback:
            raise ValueError("Qwen fallback is unsupported; invalid decisions skip the operation")
        self.qwen_memory_allow_fallback = False
        if not self.qwen_memory_allow_fallback:
            self.qwen_memory_fov_fill_min_chunks = 0
        if self.qwen_memory_select_top_k > 0:
            self.qwen_memory_fov_fill_min_chunks = min(
                self.qwen_memory_fov_fill_min_chunks,
                self.qwen_memory_select_top_k,
            )
        self.qwen_memory_frames_per_chunk = max(1, int(frames_per_chunk))
        self.qwen_memory_request_timeout_sec = max(1.0, float(request_timeout_sec))
        self.qwen_memory_select_do_sample = bool(select_do_sample)
        self.qwen_memory_select_temperature = max(0.01, float(select_temperature))
        self.qwen_memory_select_top_p = max(0.01, min(1.0, float(select_top_p)))
        self.qwen_memory_select_seed = (
            int(select_seed)
            if select_seed is not None and str(select_seed).strip() != ""
            else None
        )
        policy = str(select_policy or "coverage_only").strip().lower()
        if policy not in {"coverage_only", "fov_top2"}:
            raise ValueError(f"Unsupported memory retrieval policy: {policy}")
        self.qwen_memory_select_policy = policy
        self.qwen_memory_visual_quality_enable = bool(visual_quality_enable)
        self.qwen_memory_intermediate_chunks_dir = (
            os.path.abspath(str(intermediate_chunks_dir))
            if intermediate_chunks_dir
            else None
        )
        self.qwen_memory_update_enable = bool(update_enable)
        update_commit_mode = os.getenv(
            "QWEN_MEMORY_UPDATE_COMMIT_MODE",
            "immediate",
        ).strip().lower()
        if update_commit_mode not in {"immediate", "deferred_remote"}:
            raise ValueError(
                "QWEN_MEMORY_UPDATE_COMMIT_MODE must be immediate or "
                f"deferred_remote, got {update_commit_mode!r}"
            )
        self.qwen_memory_update_commit_mode = update_commit_mode
        self.qwen_memory_update_fov_top_k = max(1, int(update_fov_top_k))
        self.qwen_memory_class_capacity = max(1, int(class_capacity))
        self.qwen_memory_member_batch_size = max(1, int(member_batch_size))
        self.qwen_memory_confirm_prototype_min_chunks = max(
            1, int(confirm_prototype_min_chunks)
        )
        self.qwen_memory_prototype_min_fov_overlap = max(
            0.0, min(1.0, float(prototype_min_fov_overlap))
        )
        self.qwen_memory_drop_min_fov_overlap = max(
            0.0, min(1.0, float(drop_min_fov_overlap))
        )
        self.qwen_memory_disable_sparse = bool(disable_sparse)
        self.qwen_memory_sparse_member_from_ends = (
            self._parse_qwen_sparse_member_from_ends(sparse_member_from_ends)
        )
        try:
            self.qwen_memory_update_min_vq = max(
                1, min(10, int(os.getenv("QWEN_MEMORY_UPDATE_MIN_VQ", "6")))
            )
        except Exception:
            self.qwen_memory_update_min_vq = 6
        self.qwen_memory_select_active_only = bool(select_active_only)
        self.qwen_memory_select_filter_visual_quality = bool(
            select_filter_visual_quality
        )
        self.qwen_memory_strict = bool(strict)
        if not self.qwen_memory_allow_fallback:
            self.qwen_memory_strict = True
        self._qwen_memory_records = {}
        self._qwen_memory_classes = {}
        self._qwen_chunk_to_cluster = {}
        self._qwen_active_memory_chunk_ids = set()
        self._qwen_update_candidate_scores_by_chunk = {}
        self._qwen_next_cluster_id = 0
        if self.qwen_memory_judge_enable and self._is_rank0():
            if self.qwen_memory_intermediate_chunks_dir:
                os.makedirs(self.qwen_memory_intermediate_chunks_dir, exist_ok=True)
            self._qwen_log(
                "[WAN Qwen Memory Config] "
                f"enable={self.qwen_memory_judge_enable} "
                f"mode={'http' if self.qwen_memory_judge_url else 'direct'} "
                f"url={self.qwen_memory_judge_url} "
                f"model_path={self.qwen_memory_judge_model_path} "
                f"select_fov_top_k={self.qwen_memory_select_fov_top_k} "
                f"select_top_k={self.qwen_memory_select_top_k} "
                f"local_context_chunks={self.qwen_memory_local_context_chunks} "
                f"perspective={self.qwen_memory_perspective} "
                f"fov_fill_min_chunks={self.qwen_memory_fov_fill_min_chunks} "
                f"allow_fallback={self.qwen_memory_allow_fallback} "
                f"select_policy={self.qwen_memory_select_policy} "
                f"visual_quality={self.qwen_memory_visual_quality_enable} "
                f"update_enable={self.qwen_memory_update_enable} "
                f"update_commit_mode={self.qwen_memory_update_commit_mode} "
                f"update_fov_top_k={self.qwen_memory_update_fov_top_k} "
                f"update_min_vq={self.qwen_memory_update_min_vq} "
                f"class_capacity={self.qwen_memory_class_capacity} "
                f"disable_sparse={self.qwen_memory_disable_sparse} "
                f"select_filter_visual_quality={self.qwen_memory_select_filter_visual_quality} "
                f"strict={self.qwen_memory_strict} "
                f"frames_dir={self.qwen_memory_intermediate_chunks_dir}"
            )

    def _parse_qwen_sparse_member_from_ends(self, value):
        if value is None:
            value = os.getenv("QWEN_MEMORY_SPARSE_MEMBER_FROM_ENDS", "-2,-3")
        if isinstance(value, (list, tuple)):
            raw_items = value
        else:
            raw_items = str(value).replace(";", ",").split(",")
        parsed = []
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            try:
                parsed.append(int(text))
            except Exception:
                continue
        return tuple(parsed or (-2, -3))

    def _qwen_select_sampling_options(self, task, payload=None):
        if task != "select_coverage":
            return None
        if not bool(getattr(self, "qwen_memory_select_do_sample", False)):
            return None
        seed = getattr(self, "qwen_memory_select_seed", None)
        if seed is not None:
            try:
                chunk_i = int((payload or {}).get("current_chunk_i", 0))
            except Exception:
                chunk_i = 0
            task_offset = 29
            seed = int(seed) + chunk_i * 1009 + task_offset
        options = {
            "do_sample": True,
            "temperature": float(getattr(self, "qwen_memory_select_temperature", 0.7)),
            "top_p": float(getattr(self, "qwen_memory_select_top_p", 0.9)),
        }
        if seed is not None:
            options["seed"] = int(seed)
        return options

    def _qwen_memory_request(self, task, payload, max_new_tokens=256):
        if not getattr(self, "qwen_memory_judge_enable", False):
            return None
        url = getattr(self, "qwen_memory_judge_url", None)
        if not url:
            raise ValueError("Qwen inference requires the controller server URL")
        if url.rstrip("/").endswith("/health"):
            url = url.rstrip("/")[:-len("/health")] + "/judge"
        elif not url.rstrip("/").endswith("/judge"):
            url = url.rstrip("/") + "/judge"
        request_body = {
            "task": task,
            "payload": payload,
            "max_new_tokens": int(max_new_tokens),
        }
        generation_options = self._qwen_select_sampling_options(task, payload)
        if generation_options:
            request_body["generation_options"] = generation_options
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        timeout = float(getattr(self, "qwen_memory_request_timeout_sec", 180.0))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"WAN Qwen memory judge HTTP {exc.code}: {error_body}"
            ) from exc
        result = json.loads(data)
        if not result.get("ok", False):
            raise RuntimeError(result.get("error", "WAN Qwen memory judge failed"))
        return result

    def _qwen_frames_root(self):
        root = getattr(self, "qwen_memory_intermediate_chunks_dir", None)
        if not root:
            return None
        os.makedirs(root, exist_ok=True)
        return root

    def record_qwen_memory_chunk(
        self,
        chunk_i,
        video,
        action=None,
        viewmats=None,
        Ks=None,
    ):
        if not self._is_rank0():
            return
        judge_enable = bool(getattr(self, "qwen_memory_judge_enable", False))
        if not judge_enable and not self._qwen_frames_root():
            return
        chunk_i = int(chunk_i)
        arr = np.asarray(video)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4:
            self._qwen_log(
                f"[WAN Qwen Record Skip] chunk={chunk_i} unexpected_video_shape={arr.shape}"
            )
            return
        arr = arr.astype(np.float32)
        if arr.size and arr.min() < 0.0:
            arr = (arr + 1.0) * 0.5
        if arr.size and arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames = []
        mid_frame_path = None
        root = self._qwen_frames_root()
        if root and len(arr) > 0:
            frame_dir = os.path.abspath(
                os.path.join(root, f"chunk_{chunk_i:03d}_frames")
            )
            os.makedirs(frame_dir, exist_ok=True)
            for frame_i, frame in enumerate(arr):
                frame_path = os.path.join(frame_dir, f"frame_{frame_i:03d}.png")
                Image.fromarray(frame).save(frame_path)
                frames.append(frame_path)
            mid_index = len(arr) // 2
            mid_frame_path = os.path.abspath(
                os.path.join(root, f"gen_chunk_{chunk_i:03d}_mid.png")
            )
            Image.fromarray(arr[mid_index]).save(mid_frame_path)
        record = {
            "chunk_i": chunk_i,
            "frames": frames,
            "mid_frame": mid_frame_path,
            "mid_frame_index": (len(arr) // 2) if len(arr) > 0 else None,
            "visual_quality_score": None,
            "visual_quality_response": None,
            "visual_quality_error": None,
            "action": self._tensor_to_plain(action),
            "viewmats": self._tensor_to_plain(viewmats),
            "Ks": self._tensor_to_plain(Ks),
            "memory_bank_decision_finalized": False,
            "qwen_memory_update_pending_until_remote": False,
        }
        self._qwen_memory_records[chunk_i] = record
        if not judge_enable:
            self._qwen_log(
                f"[WAN Chunk Persist] chunk={chunk_i} frames={len(frames)}"
            )
            return
        update_enable = bool(getattr(self, "qwen_memory_update_enable", False))
        score = None
        if not bool(getattr(self, "qwen_memory_visual_quality_enable", True)) and not update_enable:
            self._qwen_log(
                f"[WAN Qwen VisualQuality Skip] chunk={chunk_i} frames={len(frames)}"
            )
        else:
            try:
                score = self._qwen_get_chunk_visual_quality_score(chunk_i)
                self._qwen_log(
                    f"[WAN Qwen VisualQuality] chunk={chunk_i} score={score} frames={len(frames)}"
                )
            except Exception as exc:
                record["visual_quality_error"] = repr(exc)
                self._qwen_log(
                    f"[WAN Qwen VisualQuality Error] chunk={chunk_i} error={repr(exc)}"
                )
                if bool(getattr(self, "qwen_memory_strict", False)):
                    raise
        if update_enable:
            min_vq = self._qwen_memory_update_min_visual_quality()
            if score is not None and int(score) < int(min_vq):
                self._qwen_finalize_memory_update_for_chunk(
                    chunk_i,
                    trigger_chunk_i=chunk_i,
                    deferred_until_remote=False,
                )
            elif (
                getattr(self, "qwen_memory_update_commit_mode", "immediate")
                == "immediate"
            ):
                self._qwen_finalize_memory_update_for_chunk(
                    chunk_i,
                    trigger_chunk_i=chunk_i,
                    deferred_until_remote=False,
                )
            else:
                record["qwen_memory_update_pending_until_remote"] = True
                record["qwen_memory_remote_eligible_chunk_i"] = int(
                    chunk_i
                    + max(1, int(getattr(self, "qwen_memory_local_context_chunks", 2)))
                    + 1
                )
                self._qwen_log(
                    f"[WAN Qwen Update Pending] chunk={chunk_i} vq={score} "
                    f"remote_eligible_target={record['qwen_memory_remote_eligible_chunk_i']}"
                )

    def _qwen_finalize_memory_update_for_chunk(
        self,
        chunk_i,
        trigger_chunk_i,
        deferred_until_remote,
    ):
        chunk_i = int(chunk_i)
        trigger_chunk_i = int(trigger_chunk_i)
        record = self._qwen_memory_records.get(chunk_i)
        if record is None:
            return None
        if bool(record.get("memory_bank_decision_finalized", False)):
            return record.get("qwen_memory_update", None)
        try:
            update_trace = self._update_qwen_memory_bank_for_chunk(chunk_i)
            decision = (update_trace or {}).get("decision", None)
            assigned = (update_trace or {}).get("assigned_cluster_id", None)
            matched = (update_trace or {}).get("matched_cluster_id", None)
            sparse = (update_trace or {}).get("sparsify_result", None)
            sparse_applied = bool(
                isinstance(sparse, dict) and not sparse.get("skipped", False)
            )
            self._qwen_log(
                f"[WAN Qwen Update] chunk={chunk_i} trigger_chunk={trigger_chunk_i} "
                f"deferred_until_remote={bool(deferred_until_remote)} "
                f"decision={decision} matched={matched} assigned={assigned} "
                f"sparse={sparse_applied}"
            )
        except Exception as exc:
            self._qwen_log(
                f"[WAN Qwen Update Error] chunk={chunk_i} "
                f"trigger_chunk={trigger_chunk_i} error={repr(exc)}"
            )
            raise
        record["memory_bank_decision_finalized"] = True
        record["qwen_memory_update_pending_until_remote"] = False
        record["qwen_memory_update_deferred_until_remote"] = bool(
            deferred_until_remote
        )
        record["qwen_memory_update_commit_mode"] = (
            "deferred_remote" if deferred_until_remote else "immediate"
        )
        record["qwen_memory_update_trigger_chunk_i"] = trigger_chunk_i
        if isinstance(update_trace, dict):
            update_trace["qwen_memory_update_deferred_until_remote"] = bool(
                deferred_until_remote
            )
            update_trace["qwen_memory_update_commit_mode"] = record[
                "qwen_memory_update_commit_mode"
            ]
            update_trace["qwen_memory_update_trigger_chunk_i"] = trigger_chunk_i
            trace_path = self._save_qwen_update_trace(chunk_i, update_trace)
            update_trace["trace_path"] = trace_path
        return update_trace

    def _qwen_finalize_remote_eligible_memory_chunks(
        self,
        current_chunk_i,
        current_frame_idx,
        temporal_context_size,
        chunk_size,
    ):
        if not bool(getattr(self, "qwen_memory_update_enable", False)):
            return []
        if (
            getattr(self, "qwen_memory_update_commit_mode", "immediate")
            != "deferred_remote"
        ):
            return []
        history_stop = max(0, int(current_frame_idx) - int(temporal_context_size))
        eligible_chunk_ids = [
            int(hist_idx) // int(chunk_size)
            for hist_idx in range(0, history_stop, int(chunk_size))
        ]
        finalized = []
        for chunk_i in eligible_chunk_ids:
            record = self._qwen_memory_records.get(int(chunk_i))
            if record is None or bool(
                record.get("memory_bank_decision_finalized", False)
            ):
                continue
            if not bool(record.get("qwen_memory_update_pending_until_remote", False)):
                continue
            self._qwen_finalize_memory_update_for_chunk(
                chunk_i,
                trigger_chunk_i=current_chunk_i,
                deferred_until_remote=True,
            )
            finalized.append(int(chunk_i))
        return finalized

    def _tensor_to_plain(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._tensor_to_plain(x) for x in value]
        try:
            return int(value)
        except Exception:
            return value

    def _sample_qwen_frame_paths(self, chunk_i, max_frames=None, sample_mode="middle"):
        record = self._qwen_memory_records.get(int(chunk_i), {})
        frames = list(record.get("frames", []) or [])
        if not frames:
            return []
        if max_frames is None:
            max_frames = getattr(self, "qwen_memory_frames_per_chunk", 2)
        max_frames = max(1, int(max_frames))
        if len(frames) <= max_frames:
            return frames
        if max_frames == 1:
            if sample_mode == "first":
                return [frames[0]]
            if sample_mode == "last":
                return [frames[-1]]
            return [frames[len(frames) // 2]]
        indices = sorted(
            {
                int(round(i * (len(frames) - 1) / (max_frames - 1)))
                for i in range(max_frames)
            }
        )
        return [frames[i] for i in indices]

    def _build_qwen_chunk_payload(self, chunk_i, max_frames=None, sample_mode="middle"):
        chunk_i = int(chunk_i)
        record = self._qwen_memory_records.get(chunk_i)
        if record is None:
            return None
        frames = self._sample_qwen_frame_paths(
            chunk_i, max_frames=max_frames, sample_mode=sample_mode
        )
        if not frames:
            return None
        cluster_id = self._qwen_chunk_to_cluster.get(chunk_i, chunk_i)
        payload = {
            "chunk_i": chunk_i,
            "cluster_id": cluster_id,
            "class_id": cluster_id,
            "frames": frames,
        }
        score = record.get("visual_quality_score", None)
        if score is not None:
            payload["visual_quality_score"] = int(round(float(score)))
        return payload

    def _qwen_get_chunk_visual_quality_score(self, chunk_i):
        chunk_i = int(chunk_i)
        record = self._qwen_memory_records.get(chunk_i)
        if record is None:
            return None
        previous = record.get("visual_quality_response") or {}
        if (previous.get("result") or {}).get("skipped"):
            return None
        try:
            score = int(round(float(record.get("visual_quality_score"))))
            if 1 <= score <= 10:
                return score
        except Exception:
            pass
        payload = self._build_qwen_chunk_payload(
            chunk_i,
            max_frames=1,
            sample_mode="middle",
        )
        if payload is None:
            record["visual_quality_error"] = "missing_chunk_frames"
            return None
        response = self._qwen_memory_request(
            "visual_quality",
            {"generated_chunk": payload},
            max_new_tokens=64,
        )
        result = (response or {}).get("result", {}) if isinstance(response, dict) else {}
        score = result.get("visual_quality_score", None)
        if score is None:
            record["visual_quality_error"] = "missing_visual_quality_score"
            record["visual_quality_response"] = response
            return None
        score = max(1, min(10, int(round(float(score)))))
        record["visual_quality_score"] = int(score)
        record["visual_quality_response"] = response
        record["visual_quality_error"] = None
        return int(score)

    def _qwen_memory_update_min_visual_quality(self):
        try:
            return max(
                1,
                min(
                    10,
                    int(
                        getattr(
                            self,
                            "qwen_memory_update_min_vq",
                            os.getenv("QWEN_MEMORY_UPDATE_MIN_VQ", "6"),
                        )
                    ),
                ),
            )
        except Exception:
            return 6

    def _qwen_chunk_passes_update_visual_quality(self, chunk_i, min_vq=None):
        if min_vq is None:
            min_vq = self._qwen_memory_update_min_visual_quality()
        score = self._qwen_get_chunk_visual_quality_score(int(chunk_i))
        return score is not None and int(score) >= int(min_vq), score

    def _qwen_chunk_viewmats_np(self, chunk_i):
        record = self._qwen_memory_records.get(int(chunk_i), {})
        viewmats = record.get("viewmats", None)
        if viewmats is None:
            return None
        arr = np.asarray(viewmats, dtype=np.float32)
        # ``record_qwen_memory_chunk`` stores the normal per-chunk pose tensor
        # as (T, 4, 4).  Older callers may retain one or more leading batch
        # dimensions, so normalize every valid representation to a flat pose
        # sequence before the pairwise FoV calculation below.
        if arr.ndim >= 3 and arr.shape[-2:] == (4, 4):
            return arr.reshape((-1, 4, 4))
        return None

    def _qwen_overlap_device(self):
        device = getattr(self, "_execution_device", None)
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _qwen_points_local_for_overlap(self, device):
        points = getattr(self, "points_local", None)
        if points is None:
            points = generate_points_in_sphere(50000, 8.0).to(device)
            self.points_local = points
        return points.to(device)

    def _qwen_chunk_fov_overlap(self, query_chunk_i, candidate_chunk_i):
        query_viewmats = self._qwen_chunk_viewmats_np(query_chunk_i)
        candidate_viewmats = self._qwen_chunk_viewmats_np(candidate_chunk_i)
        if query_viewmats is None or candidate_viewmats is None:
            return 0.0
        if len(query_viewmats) == 0 or len(candidate_viewmats) == 0:
            return 0.0
        query_indices = sorted({0, len(query_viewmats) // 2, len(query_viewmats) - 1})
        candidate_indices = sorted(
            {0, len(candidate_viewmats) // 2, len(candidate_viewmats) - 1}
        )
        device = self._qwen_overlap_device()
        points_local = self._qwen_points_local_for_overlap(device)
        overlaps = []
        for query_idx in query_indices:
            for candidate_idx in candidate_indices:
                try:
                    overlaps.append(
                        float(
                            calculate_fov_overlap_similarity(
                                query_viewmats[int(query_idx)],
                                candidate_viewmats[int(candidate_idx)],
                                fov_h_deg=60.0,
                                fov_v_deg=35.0,
                                device=device,
                                points_local=points_local,
                            )
                        )
                    )
                except Exception:
                    continue
        return float(np.mean(overlaps)) if overlaps else 0.0

    def _qwen_apply_chunk_cluster_metadata(self, payload, cluster_id):
        if payload is None:
            return None
        payload["cluster_id"] = int(cluster_id)
        payload["class_id"] = int(cluster_id)
        return payload

    def _qwen_memory_descriptor_from_update(self, update_result, key, fallback=None):
        value = (update_result or {}).get(key, None)
        if value is None:
            return fallback
        return value

    def _qwen_create_memory_class(self, chunk_i, update_result=None, response=None):
        chunk_i = int(chunk_i)
        cluster_id = int(self._qwen_next_cluster_id)
        self._qwen_next_cluster_id += 1
        prototype_descriptor = self._qwen_memory_descriptor_from_update(
            update_result, "updated_prototype_descriptor", None
        )
        class_descriptor = self._qwen_memory_descriptor_from_update(
            update_result, "updated_class_descriptor", prototype_descriptor
        )
        self._qwen_memory_classes[cluster_id] = {
            "cluster_id": cluster_id,
            "prototype_chunk_i": chunk_i,
            "member_chunk_ids": [chunk_i],
            "prototype_status": "tentative",
            "prototype_descriptor": prototype_descriptor,
            "class_descriptor": class_descriptor,
        }
        self._qwen_chunk_to_cluster[chunk_i] = cluster_id
        self._qwen_active_memory_chunk_ids.add(chunk_i)
        record = self._qwen_memory_records.get(chunk_i, {})
        record["cluster_id"] = cluster_id
        record["active_memory"] = True
        record["qwen_memory_update_response"] = response
        return cluster_id

    def _qwen_add_chunk_to_memory_class(self, chunk_i, cluster_id, update_result=None):
        chunk_i = int(chunk_i)
        cluster_id = int(cluster_id)
        cls = self._qwen_memory_classes.get(cluster_id)
        if cls is None:
            return self._qwen_create_memory_class(chunk_i, update_result=update_result)
        members = list(cls.get("member_chunk_ids", []) or [])
        if chunk_i not in members:
            members.append(chunk_i)
        cls["member_chunk_ids"] = members
        updated_class_descriptor = (update_result or {}).get(
            "updated_class_descriptor", None
        )
        if updated_class_descriptor is not None:
            cls["class_descriptor"] = updated_class_descriptor
        updated_prototype_descriptor = (update_result or {}).get(
            "updated_prototype_descriptor", None
        )
        if updated_prototype_descriptor is not None:
            cls["prototype_descriptor"] = updated_prototype_descriptor
        self._qwen_chunk_to_cluster[chunk_i] = cluster_id
        self._qwen_active_memory_chunk_ids.add(chunk_i)
        record = self._qwen_memory_records.get(chunk_i, {})
        record["cluster_id"] = cluster_id
        record["active_memory"] = True
        return cluster_id

    def _qwen_mark_chunk_dropped(self, chunk_i, cluster_id, update_result=None):
        chunk_i = int(chunk_i)
        cluster_id = int(cluster_id) if cluster_id is not None else None
        self._qwen_chunk_to_cluster[chunk_i] = cluster_id
        self._qwen_active_memory_chunk_ids.discard(chunk_i)
        record = self._qwen_memory_records.get(chunk_i, {})
        record["cluster_id"] = cluster_id
        record["active_memory"] = False
        record["qwen_memory_drop_result"] = update_result

    def _qwen_build_candidate_class_payloads(self, current_chunk_i):
        current_chunk_i = int(current_chunk_i)
        score_items = getattr(self, "_qwen_update_candidate_scores_by_chunk", {}).get(
            current_chunk_i, None
        )
        score_by_chunk = {}
        score_by_cluster = {}
        if score_items:
            for item in score_items:
                try:
                    score_chunk_i = int(item.get("chunk_i"))
                    score_cluster_id = int(
                        self._qwen_chunk_to_cluster.get(
                            score_chunk_i, item.get("cluster_id", score_chunk_i)
                        )
                    )
                    score_overlap = float(item.get("overlap", 0.0) or 0.0)
                except Exception:
                    continue
                score_by_chunk[score_chunk_i] = score_overlap
                prev = score_by_cluster.get(score_cluster_id)
                if prev is None or score_overlap > float(prev.get("overlap", 0.0)):
                    score_by_cluster[score_cluster_id] = {
                        "overlap": score_overlap,
                        "chunk_i": score_chunk_i,
                        "source": item.get("select_source", "wan_fov_top"),
                    }
        candidate_classes = []
        update_min_vq = self._qwen_memory_update_min_visual_quality()
        for cluster_id, cls in sorted(self._qwen_memory_classes.items()):
            cluster_id = int(cluster_id)
            if score_by_cluster and cluster_id not in score_by_cluster:
                continue
            members = []
            for x in cls.get("member_chunk_ids", []) or []:
                member_i = int(x)
                if (
                    member_i == current_chunk_i
                    or member_i not in self._qwen_memory_records
                    or member_i not in self._qwen_active_memory_chunk_ids
                ):
                    continue
                passes_vq, _score = self._qwen_chunk_passes_update_visual_quality(
                    member_i, min_vq=update_min_vq
                )
                if not passes_vq:
                    continue
                members.append(member_i)
            if not members:
                continue
            member_overlap_scores = []
            for member_chunk_i in members:
                if int(member_chunk_i) in score_by_chunk:
                    overlap = float(score_by_chunk[int(member_chunk_i)])
                else:
                    overlap = float(
                        self._qwen_chunk_fov_overlap(current_chunk_i, member_chunk_i)
                    )
                member_overlap_scores.append(
                    {
                        "chunk_i": int(member_chunk_i),
                        "fov_overlap": overlap,
                    }
                )
            member_overlap_scores.sort(
                key=lambda x: (-float(x["fov_overlap"]), int(x["chunk_i"]))
            )
            prototype_chunk_i = int(cls.get("prototype_chunk_i", members[0]))
            if prototype_chunk_i not in members:
                prototype_chunk_i = members[0]
            prototype_fov_overlap = float(
                self._qwen_chunk_fov_overlap(current_chunk_i, prototype_chunk_i)
            )
            matched_score = score_by_cluster.get(cluster_id, {})
            matched_member_chunk_i = matched_score.get("chunk_i", None)
            matched_member_fov_overlap = float(matched_score.get("overlap", 0.0) or 0.0)
            max_member_overlap = (
                float(member_overlap_scores[0]["fov_overlap"])
                if member_overlap_scores
                else 0.0
            )
            top3 = [float(x["fov_overlap"]) for x in member_overlap_scores[:3]]
            top3_mean = float(sum(top3) / len(top3)) if top3 else 0.0
            class_fov_coverage = max_member_overlap
            selected_member_ids = [prototype_chunk_i]
            for item in member_overlap_scores:
                member_chunk_i = int(item["chunk_i"])
                if member_chunk_i not in selected_member_ids:
                    selected_member_ids.append(member_chunk_i)
                if len(selected_member_ids) >= int(
                    getattr(self, "qwen_memory_member_batch_size", 8)
                ):
                    break
            prototype_payload = self._qwen_apply_chunk_cluster_metadata(
                self._build_qwen_chunk_payload(
                    prototype_chunk_i,
                    max_frames=1,
                    sample_mode="middle",
                ),
                cluster_id,
            )
            member_payloads = []
            for member_chunk_i in selected_member_ids:
                payload = self._qwen_apply_chunk_cluster_metadata(
                    self._build_qwen_chunk_payload(
                        member_chunk_i,
                        max_frames=1,
                        sample_mode="middle",
                    ),
                    cluster_id,
                )
                if payload is not None:
                    member_payloads.append(payload)
            if prototype_payload is None:
                continue
            candidate_classes.append(
                {
                    "cluster_id": int(cluster_id),
                    "member_chunk_ids": members,
                    "prototype": prototype_payload,
                    "prototype_status": cls.get("prototype_status", "tentative"),
                    "member_chunks": member_payloads,
                    "fov_overlap": max(
                        float(prototype_fov_overlap),
                        max_member_overlap,
                        matched_member_fov_overlap,
                    ),
                    "prototype_fov_overlap": max(
                        float(prototype_fov_overlap), matched_member_fov_overlap
                    ),
                    "class_fov_coverage": float(class_fov_coverage),
                    "max_member_overlap": float(max_member_overlap),
                    "top3_member_overlap_mean": float(top3_mean),
                    "member_overlap_scores": member_overlap_scores[:8],
                    "matched_member_chunk_i": (
                        int(matched_member_chunk_i)
                        if matched_member_chunk_i is not None
                        else None
                    ),
                    "matched_member_fov_overlap": float(matched_member_fov_overlap),
                    "candidate_source": (
                        "select_raw_fov_top"
                        if score_by_cluster
                        else "update_chunk_pair_fov_top"
                    ),
                    "prototype_descriptor": cls.get("prototype_descriptor", None),
                    "class_descriptor": cls.get("class_descriptor", None),
                }
            )
        candidate_classes.sort(
            key=lambda x: (
                -float(x.get("fov_overlap", 0.0)),
                -float(x.get("prototype_fov_overlap", 0.0)),
                int(x.get("cluster_id", 10**9)),
            )
        )
        return candidate_classes[: int(getattr(self, "qwen_memory_update_fov_top_k", 6))]

    def _save_qwen_update_trace(self, chunk_i, trace_payload):
        root = self._qwen_frames_root()
        if not root:
            return None
        trace_dir = os.path.join(root.rstrip(os.sep), "qwen_memory_update_traces")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.abspath(
            os.path.join(trace_dir, f"chunk_{int(chunk_i):03d}_update.json")
        )
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return trace_path

    def _save_qwen_memory_state(self):
        root = self._qwen_frames_root()
        if not root:
            return None
        state_path = os.path.abspath(
            os.path.join(root.rstrip(os.sep), "qwen_memory_state.json")
        )
        classes = []
        for cluster_id, cls in sorted(self._qwen_memory_classes.items()):
            classes.append(
                {
                    "cluster_id": int(cluster_id),
                    "prototype_chunk_i": int(cls.get("prototype_chunk_i", -1)),
                    "member_chunk_ids": [int(x) for x in cls.get("member_chunk_ids", [])],
                    "active_member_chunk_ids": [
                        int(x)
                        for x in cls.get("member_chunk_ids", [])
                        if int(x) in self._qwen_active_memory_chunk_ids
                    ],
                    "prototype_status": cls.get("prototype_status", "tentative"),
                    "prototype_descriptor": cls.get("prototype_descriptor", None),
                    "class_descriptor": cls.get("class_descriptor", None),
                }
            )
        payload = {
            "classes": classes,
            "active_memory_chunk_ids": sorted(
                int(x) for x in self._qwen_active_memory_chunk_ids
            ),
            "chunk_to_cluster": {
                str(int(k)): (int(v) if v is not None else None)
                for k, v in sorted(self._qwen_chunk_to_cluster.items())
            },
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return state_path

    def _qwen_sparse_candidate_positions(self, member_count):
        positions = []
        for from_end in getattr(self, "qwen_memory_sparse_member_from_ends", (-2, -3)):
            pos = int(member_count) + int(from_end) if int(from_end) < 0 else int(from_end)
            if 0 < pos < int(member_count) and pos not in positions:
                positions.append(pos)
        if not positions and int(member_count) > 1:
            positions.append(1)
        return positions

    def _qwen_sparsify_memory_class_if_needed(self, cluster_id):
        if bool(getattr(self, "qwen_memory_disable_sparse", False)):
            return None
        cluster_id = int(cluster_id)
        cls = self._qwen_memory_classes.get(cluster_id)
        if cls is None:
            return None
        capacity = int(getattr(self, "qwen_memory_class_capacity", 7))
        members = [int(x) for x in cls.get("member_chunk_ids", []) or []]
        active_members = [x for x in members if x in self._qwen_active_memory_chunk_ids]
        if len(active_members) < capacity:
            return {
                "skipped": True,
                "skip_reason": "within_capacity",
                "cluster_id": cluster_id,
                "capacity": capacity,
                "active_member_count": len(active_members),
            }
        positions = self._qwen_sparse_candidate_positions(len(active_members))
        if not positions:
            return {
                "skipped": True,
                "skip_reason": "no_sparse_candidates",
                "cluster_id": cluster_id,
                "capacity": int(capacity),
                "active_member_count": len(active_members),
            }
        primary_pos = int(positions[0])
        primary_chunk_i = int(active_members[primary_pos])

        def _judge_sparse_position(pos, extra_removed_ids=None, branch="standard"):
            pos = int(pos)
            candidate_chunk_i = int(active_members[pos])
            scenario_removed_ids = []
            for removed_id in [candidate_chunk_i] + list(extra_removed_ids or []):
                removed_id = int(removed_id)
                if removed_id not in scenario_removed_ids:
                    scenario_removed_ids.append(removed_id)
            scenario_removed_set = set(scenario_removed_ids)
            remaining_ids = [
                x for x in active_members if int(x) not in scenario_removed_set
            ]
            if not remaining_ids:
                return None
            overlaps = [
                {
                    "chunk_i": int(other_chunk_i),
                    "fov_overlap": float(
                        self._qwen_chunk_fov_overlap(candidate_chunk_i, other_chunk_i)
                    ),
                }
                for other_chunk_i in remaining_ids
            ]
            overlaps.sort(key=lambda x: (-float(x["fov_overlap"]), int(x["chunk_i"])))
            coverage = float(overlaps[0]["fov_overlap"]) if overlaps else 0.0
            remaining_payload_ids = []
            prototype_chunk_i = int(cls.get("prototype_chunk_i", remaining_ids[0]))
            if prototype_chunk_i in remaining_ids:
                remaining_payload_ids.append(prototype_chunk_i)
            for item in overlaps:
                other_chunk_i = int(item["chunk_i"])
                if other_chunk_i not in remaining_payload_ids:
                    remaining_payload_ids.append(other_chunk_i)
                if len(remaining_payload_ids) >= int(
                    getattr(self, "qwen_memory_member_batch_size", 8)
                ):
                    break
            remaining_payloads = []
            for other_chunk_i in remaining_payload_ids:
                payload = self._qwen_apply_chunk_cluster_metadata(
                    self._build_qwen_chunk_payload(
                        other_chunk_i,
                        max_frames=1,
                        sample_mode="middle",
                    ),
                    cluster_id,
                )
                if payload is not None:
                    remaining_payloads.append(payload)
            candidate_payload = self._qwen_apply_chunk_cluster_metadata(
                self._build_qwen_chunk_payload(
                    candidate_chunk_i,
                    max_frames=1,
                    sample_mode="middle",
                ),
                cluster_id,
            )
            if candidate_payload is None:
                return None
            response = self._qwen_memory_request(
                "sparse",
                {
                    "cluster_id": int(cluster_id),
                    "candidate_chunk": candidate_payload,
                    "candidate_chunk_i": int(candidate_chunk_i),
                    "remaining_chunks": remaining_payloads,
                    "remaining_chunk_ids": [int(x) for x in remaining_ids],
                    "coverage_by_remaining_chunks": float(coverage),
                    "scenario_assumed_removed_chunk_ids": [
                        int(x) for x in scenario_removed_ids
                    ],
                },
                max_new_tokens=128,
            )
            result = (response or {}).get("result", {}) if isinstance(response, dict) else {}
            return {
                "chunk_i": int(candidate_chunk_i),
                "member_position": int(pos),
                "member_from_end": int(pos - len(active_members)),
                "branch": str(branch),
                "scenario_assumed_removed_chunk_ids": [
                    int(x) for x in scenario_removed_ids
                ],
                "coverage_by_remaining_chunks": float(coverage),
                "top_remaining_overlaps": overlaps[:5],
                "response": response,
                "result": result,
            }

        sparse_jobs = [
            {
                "position_order": int(order),
                "pos": int(pos),
                "extra_removed_ids": [],
                "branch": "standard",
            }
            for order, pos in enumerate(positions)
        ]
        if len(positions) >= 2:
            sparse_jobs.append(
                {
                    "position_order": 1,
                    "pos": int(positions[1]),
                    "extra_removed_ids": [primary_chunk_i],
                    "branch": "without_primary",
                }
            )
        candidate_scores = []
        max_workers = max(
            1,
            min(
                len(sparse_jobs),
                int(os.getenv("QWEN_MEMORY_SPARSE_PARALLEL_WORKERS", "3")),
            ),
        )
        if max_workers > 1 and len(sparse_jobs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {
                    executor.submit(
                        _judge_sparse_position,
                        job["pos"],
                        job["extra_removed_ids"],
                        job["branch"],
                    ): job
                    for job in sparse_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    item = future.result()
                    if item is None:
                        continue
                    item["position_order"] = int(job["position_order"])
                    candidate_scores.append(item)
        else:
            for job in sparse_jobs:
                item = _judge_sparse_position(
                    job["pos"], job["extra_removed_ids"], job["branch"]
                )
                if item is None:
                    continue
                item["position_order"] = int(job["position_order"])
                candidate_scores.append(item)
        branch_rank = {"standard": 0, "without_primary": 1}
        candidate_scores.sort(
            key=lambda item: (
                int(item.get("position_order", 10**9)),
                branch_rank.get(str(item.get("branch", "")), 9),
                int(item.get("member_position", 10**9)),
            )
        )

        removed_chunk_id = None
        removed_response = None
        decision_branch = None
        standard_scores_by_pos = {
            int(item["member_position"]): item
            for item in candidate_scores
            if str(item.get("branch", "standard")) == "standard"
        }
        for pos in positions:
            item = standard_scores_by_pos.get(int(pos))
            if item is None:
                continue
            result = item.get("result", {}) or {}
            if str(result.get("action", "keep")).strip().lower() == "remove":
                removed_chunk_id = int(item["chunk_i"])
                removed_response = item.get("response")
                decision_branch = str(item.get("branch", "standard"))
                break
        if removed_chunk_id is not None:
            self._qwen_active_memory_chunk_ids.discard(removed_chunk_id)
            record = self._qwen_memory_records.get(removed_chunk_id, {})
            record["active_memory"] = False
            record["qwen_sparse_removed"] = {
                "cluster_id": int(cluster_id),
                "response": removed_response,
            }
        return {
            "skipped": removed_chunk_id is None,
            "skip_reason": None if removed_chunk_id is not None else "qwen_kept_candidates",
            "cluster_id": int(cluster_id),
            "capacity": int(capacity),
            "active_member_count_before": len(active_members),
            "removed_chunk_i": removed_chunk_id,
            "removed_chunk_ids": [removed_chunk_id] if removed_chunk_id is not None else [],
            "selection_policy": "qwen_sparse_from_ends_parallel",
            "decision_branch": decision_branch,
            "candidate_scores": candidate_scores,
        }

    def _qwen_skip_memory_update(self, chunk_i, response):
        trace = {
            "chunk_i": int(chunk_i), "decision": None, "skipped": True,
            "reason": "invalid_model_output", "assigned_cluster_id": None,
            "matched_cluster_id": None, "sparsify_result": None,
            "update_response": response,
        }
        trace["trace_path"] = self._save_qwen_update_trace(chunk_i, trace)
        self._qwen_memory_records[chunk_i]["qwen_memory_update"] = trace
        self._qwen_log(f"[WAN Qwen Update Skip] chunk={chunk_i} reason=invalid_model_output")
        return trace

    def _update_qwen_memory_bank_for_chunk(self, chunk_i):
        chunk_i = int(chunk_i)
        current_payload = self._build_qwen_chunk_payload(
            chunk_i,
            max_frames=1,
            sample_mode="middle",
        )
        if current_payload is None:
            raise RuntimeError(f"missing qwen payload for chunk {chunk_i}")
        update_min_vq = self._qwen_memory_update_min_visual_quality()
        current_vq = current_payload.get("visual_quality_score", None)
        if current_vq is None:
            current_vq = self._qwen_get_chunk_visual_quality_score(chunk_i)
        if current_vq is None:
            return self._qwen_skip_memory_update(
                chunk_i, self._qwen_memory_records[chunk_i].get("visual_quality_response"))
        if current_vq is not None:
            current_vq = int(current_vq)
            current_payload["visual_quality_score"] = current_vq
        if current_vq is not None and int(current_vq) < int(update_min_vq):
            update_result = {
                "decision": "drop",
                "matched_cluster_id": None,
                "reason": (
                    f"visual_quality_score<{int(update_min_vq)}; "
                    "skip create/add to qwen memory"
                ),
                "visual_quality_score": int(current_vq),
                "update_min_vq": int(update_min_vq),
                "update_mode": "wan_qwen_quality_gate",
            }
            self._qwen_mark_chunk_dropped(
                chunk_i, None, update_result=update_result
            )
            record = self._qwen_memory_records.get(chunk_i, {})
            trace_payload = {
                "schema_version": 1,
                "runtime_update_flow": "wan_qwen_light_class_bank",
                "chunk_i": int(chunk_i),
                "decision": "drop",
                "matched_cluster_id": None,
                "assigned_cluster_id": None,
                "candidate_cluster_ids": [],
                "candidate_classes": [],
                "current_chunk": self._qwen_payload_image_ref(current_payload),
                "update_response": None,
                "update_result": update_result,
                "sparsify_result": None,
                "active_memory_chunk_ids": sorted(
                    int(x) for x in self._qwen_active_memory_chunk_ids
                ),
            }
            trace_path = self._save_qwen_update_trace(chunk_i, trace_payload)
            state_path = self._save_qwen_memory_state()
            trace_payload["trace_path"] = trace_path
            trace_payload["state_path"] = state_path
            record["qwen_memory_update"] = trace_payload
            return trace_payload
        candidate_classes = self._qwen_build_candidate_class_payloads(chunk_i)
        update_result = {
            "decision": "create",
            "matched_cluster_id": None,
            "reason": "no_existing_candidate_class",
            "update_mode": "wan_qwen_split_create",
        }
        create_responses = []
        matched_cluster_id = None
        if candidate_classes:
            # Match HY split5 update: judge each class with current/prototype only.
            def _judge_create_candidate(index, class_payload):
                cluster_id = int(class_payload["cluster_id"])
                prototype_chunk_i = class_payload.get("prototype", {}).get("chunk_i")
                try:
                    prototype_chunk_i = int(prototype_chunk_i)
                except Exception:
                    prototype_chunk_i = None
                prototype_payload = None
                if prototype_chunk_i is not None:
                    prototype_payload = self._qwen_apply_chunk_cluster_metadata(
                        self._build_qwen_chunk_payload(
                            prototype_chunk_i,
                            max_frames=1,
                            sample_mode="middle",
                        ),
                        cluster_id,
                    )
                if prototype_payload is None:
                    prototype_payload = dict(class_payload.get("prototype", {}) or {})
                    prototype_payload["cluster_id"] = cluster_id
                prototype_payload["fov_overlap"] = float(
                    class_payload.get("prototype_fov_overlap", 0.0) or 0.0
                )
                request_payload = {
                    "current_chunk": current_payload,
                    "prototype": prototype_payload,
                    "candidate_class": {
                        "cluster_id": cluster_id,
                        "prototype": prototype_payload,
                        "prototype_status": class_payload.get(
                            "prototype_status", "tentative"
                        ),
                        "prototype_fov_overlap": float(
                            class_payload.get("prototype_fov_overlap", 0.0) or 0.0
                        ),
                        "fov_overlap": float(
                            class_payload.get("fov_overlap", 0.0) or 0.0
                        ),
                    },
                    "cluster_id": cluster_id,
                    "class_id": cluster_id,
                    "prototype_fov_overlap": float(
                        class_payload.get("prototype_fov_overlap", 0.0) or 0.0
                    ),
                }
                try:
                    create_response = self._qwen_memory_request(
                        "create",
                        request_payload,
                        max_new_tokens=128,
                    )
                    result = (
                        create_response.get("result", {})
                        if isinstance(create_response, dict)
                        else {}
                    )
                    prototype_match = bool(result.get("prototype_match", False))
                    create_item = {
                        "candidate_index": int(index),
                        "cluster_id": cluster_id,
                        "prototype_chunk_i": prototype_chunk_i,
                        "prototype_fov_overlap": float(
                            class_payload.get("prototype_fov_overlap", 0.0) or 0.0
                        ),
                        "prototype_match": prototype_match,
                        "skipped": bool(result.get("skipped")),
                        "view_overlap": result.get("view_overlap", None),
                        "view_overlap_score": result.get("view_overlap_score", None),
                        "reason": result.get("reason", None),
                        "response": create_response,
                        "error": None,
                        "request_image_paths": {
                            "current_chunk": self._qwen_payload_image_ref(
                                current_payload
                            ),
                            "prototype": self._qwen_payload_image_ref(
                                prototype_payload
                            ),
                        },
                    }
                except Exception as exc:
                    if bool(getattr(self, "qwen_memory_strict", False)):
                        raise
                    create_item = {
                        "candidate_index": int(index),
                        "cluster_id": cluster_id,
                        "prototype_chunk_i": prototype_chunk_i,
                        "prototype_fov_overlap": float(
                            class_payload.get("prototype_fov_overlap", 0.0) or 0.0
                        ),
                        "prototype_match": False,
                        "view_overlap": "none",
                        "view_overlap_score": None,
                        "reason": f"create_error: {repr(exc)}",
                        "response": None,
                        "error": repr(exc),
                        "raw": str(getattr(exc, "raw", ""))[:600]
                        if getattr(exc, "raw", None) is not None
                        else None,
                    }
                return create_item

            max_workers = max(
                1,
                min(
                    len(candidate_classes),
                    int(os.getenv("QWEN_MEMORY_CREATE_PARALLEL_WORKERS", "4")),
                ),
            )
            if max_workers > 1 and len(candidate_classes) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            _judge_create_candidate,
                            index,
                            class_payload,
                        )
                        for index, class_payload in enumerate(candidate_classes)
                    ]
                    for future in as_completed(futures):
                        create_responses.append(future.result())
                create_responses.sort(
                    key=lambda item: int(item.get("candidate_index", 10**9))
                )
            else:
                for index, class_payload in enumerate(candidate_classes):
                    create_responses.append(
                        _judge_create_candidate(index, class_payload)
                    )
            if any(item.get("skipped") or item.get("error") for item in create_responses):
                return self._qwen_skip_memory_update(chunk_i, create_responses)
            matched_create = None
            for create_item in create_responses:
                if bool(create_item.get("prototype_match", False)):
                    matched_create = create_item
                    matched_cluster_id = int(create_item["cluster_id"])
                    break
            if matched_create is not None:
                update_result = {
                    "decision": "add",
                    "matched_cluster_id": int(matched_cluster_id),
                    "prototype_match": True,
                    "view_overlap": matched_create.get("view_overlap", None),
                    "view_overlap_score": matched_create.get(
                        "view_overlap_score", None
                    ),
                    "confidence": 1.0,
                    "reason": matched_create.get("reason", "prototype matched"),
                    "create_responses": create_responses,
                    "updated_prototype_descriptor": None,
                    "updated_class_descriptor": None,
                    "update_mode": "wan_qwen_split_create",
                }
            else:
                update_result = {
                    "decision": "create",
                    "matched_cluster_id": None,
                    "prototype_match": False,
                    "view_overlap": (
                        create_responses[0].get("view_overlap", "none")
                        if create_responses
                        else "none"
                    ),
                    "view_overlap_score": (
                        create_responses[0].get("view_overlap_score", None)
                        if create_responses
                        else None
                    ),
                    "confidence": 0.0,
                    "reason": (
                        create_responses[0].get("reason", "no prototype match")
                        if create_responses
                        else "no prototype match"
                    ),
                    "create_responses": create_responses,
                    "updated_prototype_descriptor": None,
                    "updated_class_descriptor": None,
                    "update_mode": "wan_qwen_split_create",
                }
        response = {
            "ok": True,
            "task": "wan_qwen_split_create",
            "result": update_result,
            "create_responses": create_responses,
        }
        decision = str(update_result.get("decision", "create")).strip().lower()
        if decision not in {"create", "add", "drop"}:
            decision = "create"
        try:
            matched_cluster_id = update_result.get("matched_cluster_id", matched_cluster_id)
            matched_cluster_id = (
                int(matched_cluster_id) if matched_cluster_id is not None else None
            )
        except Exception:
            matched_cluster_id = None
        candidate_by_cluster = {
            int(cls["cluster_id"]): cls for cls in candidate_classes
        }
        if matched_cluster_id not in candidate_by_cluster and decision in {"add", "drop"}:
            update_result["invalid_matched_cluster_id"] = matched_cluster_id
            decision = "create"
            matched_cluster_id = None
        if matched_cluster_id is not None:
            matched_class = candidate_by_cluster.get(matched_cluster_id, {})
            prototype_fov_overlap = float(
                matched_class.get("prototype_fov_overlap", 0.0)
            )
            class_fov_coverage = float(matched_class.get("class_fov_coverage", 0.0))
            update_result["computed_prototype_fov_overlap"] = prototype_fov_overlap
            update_result["computed_class_fov_coverage"] = class_fov_coverage
            update_result["qwen_decision_authoritative"] = True
        sparsify_result = None
        if decision == "create" or matched_cluster_id is None:
            assigned_cluster_id = self._qwen_create_memory_class(
                chunk_i, update_result=update_result, response=response
            )
            matched_cluster_id = None
            decision = "create"
        elif decision == "drop":
            assigned_cluster_id = int(matched_cluster_id)
            self._qwen_mark_chunk_dropped(
                chunk_i, assigned_cluster_id, update_result=update_result
            )
        else:
            assigned_cluster_id = int(matched_cluster_id)
            self._qwen_add_chunk_to_memory_class(
                chunk_i, assigned_cluster_id, update_result=update_result
            )
            sparsify_result = self._qwen_sparsify_memory_class_if_needed(
                assigned_cluster_id
            )
        record = self._qwen_memory_records.get(chunk_i, {})
        trace_payload = {
            "schema_version": 1,
            "runtime_update_flow": "wan_qwen_light_class_bank",
            "chunk_i": int(chunk_i),
            "decision": decision,
            "matched_cluster_id": matched_cluster_id,
            "assigned_cluster_id": (
                int(assigned_cluster_id) if assigned_cluster_id is not None else None
            ),
            "candidate_cluster_ids": [
                int(cls["cluster_id"]) for cls in candidate_classes
            ],
            "candidate_classes": candidate_classes,
            "current_chunk": self._qwen_payload_image_ref(current_payload),
            "update_response": response,
            "update_result": update_result,
            "sparsify_result": sparsify_result,
            "active_memory_chunk_ids": sorted(
                int(x) for x in self._qwen_active_memory_chunk_ids
            ),
        }
        trace_path = self._save_qwen_update_trace(chunk_i, trace_payload)
        state_path = self._save_qwen_memory_state()
        trace_payload["trace_path"] = trace_path
        trace_payload["state_path"] = state_path
        record["qwen_memory_update"] = trace_payload
        return trace_payload

    def _qwen_payload_image_ref(self, payload):
        if payload is None:
            return None
        result = {
            "chunk_i": int(payload["chunk_i"]),
            "cluster_id": payload.get("cluster_id", None),
            "visual_quality_score": payload.get("visual_quality_score", None),
            "frames": [os.path.abspath(str(x)) for x in payload.get("frames", [])],
        }
        if payload.get("option_label") is not None:
            result["option_label"] = str(payload["option_label"])
        return result

    def _format_qwen_target_motion_text(self, action_values):
        trans_names = {
            0: "none",
            1: "forward",
            2: "backward",
            3: "right",
            4: "left",
            5: "forward+right",
            6: "forward+left",
            7: "backward+right",
            8: "backward+left",
        }
        rot_names = {
            0: "none",
            1: "yaw_right",
            2: "yaw_left",
            3: "pitch_up",
            4: "pitch_down",
            5: "yaw_right+pitch_up",
            6: "yaw_right+pitch_down",
            7: "yaw_left+pitch_up",
            8: "yaw_left+pitch_down",
        }
        forward_units = 0.0
        right_units = 0.0
        up_units = 0.0
        yaw_degrees = 0.0
        pitch_degrees = 0.0
        for action_id in action_values or []:
            action_id = int(action_id)
            trans_parts = set(trans_names.get(action_id // 9, "none").split("+"))
            rot_parts = set(rot_names.get(action_id % 9, "none").split("+"))
            if "forward" in trans_parts:
                forward_units += 0.08
            if "backward" in trans_parts:
                forward_units -= 0.08
            if "right" in trans_parts:
                right_units += 0.08
            if "left" in trans_parts:
                right_units -= 0.08
            if "yaw_right" in rot_parts:
                yaw_degrees += 3.0
            if "yaw_left" in rot_parts:
                yaw_degrees -= 3.0
            if "pitch_up" in rot_parts:
                pitch_degrees += 3.0
            if "pitch_down" in rot_parts:
                pitch_degrees -= 3.0
        motion_tokens = []
        if abs(yaw_degrees) >= 5.0:
            motion_tokens.append("turn_right" if yaw_degrees > 0 else "turn_left")
        if abs(pitch_degrees) >= 5.0:
            motion_tokens.append("tilt_up" if pitch_degrees > 0 else "tilt_down")
        axes = {
            "move_right": right_units,
            "move_up": up_units,
            "move_forward": forward_units,
        }
        dominant_axis, dominant_value = max(axes.items(), key=lambda item: abs(item[1]))
        if abs(dominant_value) > 1e-6:
            if dominant_axis == "move_right":
                motion_tokens.append("move_right" if dominant_value > 0 else "move_left")
            elif dominant_axis == "move_up":
                motion_tokens.append("move_up" if dominant_value > 0 else "move_down")
            else:
                motion_tokens.append(
                    "move_forward" if dominant_value > 0 else "move_backward"
                )
        if not motion_tokens:
            motion_tokens.append("mostly_static")
        return (
            f"camera_motion={'+'.join(motion_tokens)}; "
            f"local_translation(right={right_units:.4f}, up={up_units:.4f}, forward={forward_units:.4f}); "
            f"local_rotation(yaw={yaw_degrees:.4f}deg, pitch={pitch_degrees:.4f}deg, roll=0.0000deg)"
        )

    def _qwen_fov_candidate_scores(
        self,
        viewmats_np,
        current_frame_idx,
        temporal_context_size,
        pred_latent_size,
        device,
    ):
        candidate_scores = []
        query_clip_indices = list(
            range(
                int(current_frame_idx),
                min(int(current_frame_idx) + int(pred_latent_size), len(viewmats_np)),
            )
        )
        history_stop = max(0, int(current_frame_idx) - int(temporal_context_size))
        for hist_idx in range(0, history_stop, int(pred_latent_size)):
            chunk_i = hist_idx // int(pred_latent_size)
            if chunk_i not in self._qwen_memory_records:
                continue
            if (
                bool(getattr(self, "qwen_memory_update_enable", False))
                and bool(getattr(self, "qwen_memory_select_active_only", True))
                and int(chunk_i) not in self._qwen_active_memory_chunk_ids
            ):
                continue
            comparisons = []
            for query_idx in query_clip_indices:
                for candidate_idx in (hist_idx, min(hist_idx + 2, len(viewmats_np) - 1)):
                    comparisons.append(
                        float(
                            calculate_fov_overlap_similarity(
                                viewmats_np[query_idx],
                                viewmats_np[candidate_idx],
                                fov_h_deg=60.0,
                                fov_v_deg=35.0,
                                device=device,
                                points_local=self.points_local,
                            )
                        )
                    )
            overlap = float(np.mean(comparisons)) if comparisons else 0.0
            cluster_id = self._qwen_chunk_to_cluster.get(chunk_i)
            if cluster_id is None:
                cluster_id = chunk_i
            candidate_scores.append(
                {
                    "chunk_i": int(chunk_i),
                    "cluster_id": int(cluster_id),
                    "overlap": overlap,
                    "active_memory": bool(
                        int(chunk_i) in self._qwen_active_memory_chunk_ids
                    ),
                    "select_source": "wan_fov_top",
                }
            )
        candidate_scores.sort(key=lambda x: (-float(x["overlap"]), int(x["chunk_i"])))
        return candidate_scores[: int(getattr(self, "qwen_memory_select_fov_top_k", 7))]

    def _candidate_payloads_from_scores(
        self,
        scores,
        role,
        max_payloads,
    ):
        payloads = []
        seen = set()
        min_vq = 6
        env_key = (
            "QWEN_MEMORY_SELECT_STABLE_MIN_VQ"
            if role == "stable_candidate"
            else "QWEN_MEMORY_SELECT_COVERAGE_MIN_VQ"
        )
        try:
            min_vq = int(os.getenv(env_key, str(min_vq)))
        except Exception:
            pass
        for item in scores:
            if len(payloads) >= int(max_payloads):
                break
            chunk_i = int(item["chunk_i"])
            if chunk_i in seen:
                continue
            seen.add(chunk_i)
            payload = self._build_qwen_chunk_payload(
                chunk_i, max_frames=1, sample_mode="middle"
            )
            if payload is None:
                continue
            if bool(
                getattr(self, "qwen_memory_select_filter_visual_quality", True)
            ):
                score = payload.get("visual_quality_score", None)
                if score is None:
                    score = self._qwen_get_chunk_visual_quality_score(chunk_i)
                if score is None or int(score) < min_vq:
                    continue
                payload["visual_quality_score"] = int(score)
            payload["fov_overlap"] = float(item.get("overlap", 0.0) or 0.0)
            payload["selection_role_hint"] = role
            payload["option_label"] = chr(ord("A") + len(payloads))
            payloads.append(payload)
        return payloads

    def _qwen_select_memory_chunks(
        self,
        current_chunk_i,
        local_chunk_ids,
        raw_top_memory_chunk_scores,
        current_chunk_action=None,
    ):
        local_context_chunk_ids = list(local_chunk_ids or [])[-1:]
        target_motion_text = self._format_qwen_target_motion_text(current_chunk_action)
        policy = str(
            getattr(self, "qwen_memory_select_policy", "coverage_only")
            or "coverage_only"
        ).strip().lower()
        if policy != "coverage_only":
            raise ValueError("The released controller uses coverage_only retrieval")
        if not local_context_chunk_ids:
            return None
        current_payload = self._build_qwen_chunk_payload(
            int(local_context_chunk_ids[-1]),
            max_frames=1,
            sample_mode="last",
        )
        if current_payload is None:
            return None
        coverage_payloads = self._candidate_payloads_from_scores(
            raw_top_memory_chunk_scores,
            "coverage_candidate",
            max_payloads=7,
        )
        coverage_response = None
        if coverage_payloads:
            coverage_response = self._qwen_memory_request(
                "select_coverage",
                {"current_chunk_i": int(current_chunk_i),
                 "target_next_chunk_id": int(current_chunk_i),
                 "current_chunk": current_payload,
                 "planned_camera_motion": target_motion_text,
                 "target_motion_text": target_motion_text,
                 "coverage_candidates": coverage_payloads,
                 "candidates": coverage_payloads}, max_new_tokens=128)
        coverage_by_option = {str(p["option_label"]): p for p in coverage_payloads}
        coverage_result = (
            coverage_response.get("result", {})
            if isinstance(coverage_response, dict)
            else {}
        )
        select_top_k = int(getattr(self, "qwen_memory_select_top_k", 3))
        coverage_limit = select_top_k if select_top_k > 0 else None
        final_payloads = []
        for option in coverage_result.get("selected_coverage", []) or []:
            payload = coverage_by_option.get(str(option))
            if payload is not None and all(
                int(x["chunk_i"]) != int(payload["chunk_i"]) for x in final_payloads
            ):
                final_payloads.append(payload)
            if coverage_limit is not None and len(final_payloads) >= coverage_limit:
                break
        if select_top_k > 0:
            final_payloads = final_payloads[:select_top_k]
        selected_ids = [int(payload["chunk_i"]) for payload in final_payloads]
        return {
            "memory_chunk_ids": selected_ids,
            "target_motion_text": target_motion_text,
            "selection_policy": policy,
            "coverage_response": coverage_response,
            "stable_response": None,
            "current_chunk": self._qwen_payload_image_ref(current_payload),
            "coverage_candidate_chunks": [
                self._qwen_payload_image_ref(payload) for payload in coverage_payloads
            ],
            "stable_candidate_chunks": [],
        }

    def _save_qwen_select_trace(self, chunk_i, trace_payload):
        trace_path = self._qwen_select_trace_path(chunk_i)
        if not trace_path:
            return None
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        tmp_path = f"{trace_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, trace_path)
        return trace_path

    def _qwen_select_trace_path(self, chunk_i):
        root = self._qwen_frames_root()
        if not root:
            return None
        trace_dir = os.path.join(root.rstrip("/"), "qwen_select_traces")
        return os.path.join(trace_dir, f"chunk_{int(chunk_i):03d}_select.json")

    def _qwen_select_debug_sheet_path(self, chunk_i):
        root = self._qwen_frames_root()
        if not root:
            return None
        sheet_dir = os.path.join(root.rstrip("/"), "qwen_select_debug_sheets")
        return os.path.join(sheet_dir, f"chunk_{int(chunk_i):03d}_select_debug.jpg")

    def _qwen_debug_font(self, size=14, bold=False):
        paths = (
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
            if bold
            else [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
        for path in paths:
            try:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _qwen_last_chunk_frame_path(self, chunk_i):
        record = self._qwen_memory_records.get(int(chunk_i), {})
        frames = list(record.get("frames", []) or [])
        if frames and os.path.isfile(str(frames[-1])):
            return str(frames[-1])
        root = self._qwen_frames_root()
        if not root:
            return None
        frame_dir = os.path.join(root, f"chunk_{int(chunk_i):03d}_frames")
        if not os.path.isdir(frame_dir):
            return None
        try:
            names = sorted(
                name
                for name in os.listdir(frame_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            )
        except Exception:
            return None
        if not names:
            mid_frame = record.get("mid_frame")
            if mid_frame and os.path.isfile(str(mid_frame)):
                return str(mid_frame)
            return None
        return os.path.join(frame_dir, names[-1])

    def _qwen_debug_wrap(self, text, width):
        import textwrap

        return textwrap.wrap(
            str(text or ""),
            width=width,
            break_long_words=False,
            replace_whitespace=False,
        ) or [""]

    def _qwen_debug_thumb(self, path, size):
        canvas = Image.new("RGB", size, (230, 230, 230))
        try:
            if path and os.path.exists(str(path)):
                image = Image.open(str(path)).convert("RGB")
                image.thumbnail(size, Image.Resampling.LANCZOS)
                canvas.paste(
                    image,
                    ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
                )
                return canvas
        except Exception:
            pass
        draw = ImageDraw.Draw(canvas)
        draw.text((12, size[1] // 2 - 8), "missing", fill=(80, 80, 80), font=self._qwen_debug_font(14))
        return canvas

    def _save_qwen_select_debug_sheet(self, chunk_i, trace_payload):
        raw = os.getenv("QWEN_MEMORY_SELECT_DEBUG_SHEET", "true")
        if str(raw).strip().lower() in {"0", "false", "no", "off"}:
            return None
        sheet_path = self._qwen_select_debug_sheet_path(chunk_i)
        if not sheet_path:
            return None

        try:
            qwen_info = trace_payload.get("qwen_info") or {}
            coverage_response = qwen_info.get("coverage_response") or {}
            coverage_result = coverage_response.get("result") or {}
            per_candidate = {
                str(item.get("option")): item
                for item in (coverage_result.get("per_candidate") or [])
                if item.get("option") is not None
            }
            selected_options = {
                str(option) for option in (coverage_result.get("selected_coverage") or [])
            }
            acceptable_options = {
                str(option) for option in (coverage_result.get("acceptable_coverage") or [])
            }
            stable_response = qwen_info.get("stable_response") or {}
            stable_result = stable_response.get("result") or {}
            stable_selected = str(stable_result.get("selected_stabilizer") or "")
            score_by_chunk = {}
            for item in trace_payload.get("raw_top_memory_chunk_scores") or []:
                try:
                    score_by_chunk[int(item.get("chunk_i"))] = item
                except Exception:
                    pass

            selected_chunks = {
                int(x) for x in (trace_payload.get("selected_memory_chunk_ids") or [])
            }
            qwen_selected_chunks = {
                int(x)
                for x in (trace_payload.get("qwen_selected_memory_chunk_ids") or [])
            }
            fov_fill_chunks = {
                int(x) for x in (trace_payload.get("fov_fill_memory_chunk_ids") or [])
            }
            local_chunk_ids = [int(x) for x in trace_payload.get("local_chunk_ids") or []]
            tiles = []
            displayed_candidate_chunks = set()
            for local_chunk_i in local_chunk_ids:
                tiles.append(
                    {
                        "kind": "local",
                        "image": self._qwen_last_chunk_frame_path(local_chunk_i),
                        "title": f"LOCAL chunk {local_chunk_i} last",
                        "lines": [
                            "last frame of adjacent context",
                            "included as temporal local",
                        ],
                        "selected": False,
                    }
                )

            def add_candidate_tiles(candidates, prefix, selected_option_set):
                for idx, payload in enumerate(candidates or []):
                    option = str(payload.get("option_label") or chr(ord("A") + idx))
                    try:
                        candidate_chunk_i = int(payload.get("chunk_i"))
                    except Exception:
                        continue
                    displayed_candidate_chunks.add(candidate_chunk_i)
                    result = per_candidate.get(option, {})
                    score = score_by_chunk.get(candidate_chunk_i, {})
                    frames = payload.get("frames") or []
                    is_selected = (
                        candidate_chunk_i in selected_chunks
                        or option in selected_option_set
                        or bool(result.get("selected"))
                        or (prefix == "S" and option == stable_selected)
                    )
                    lines = [
                        f"chunk={candidate_chunk_i} cid={payload.get('cluster_id')} vq={payload.get('visual_quality_score')}",
                        f"overlap={float(score.get('overlap', 0.0)):.3f}"
                        if score
                        else "overlap=n/a",
                        f"coverage={result.get('coverage_value', 'n/a')} align={result.get('motion_alignment', 'n/a')}",
                        f"role={result.get('coverage_role', 'n/a')}",
                    ]
                    title = f"{prefix}{option}: chunk {candidate_chunk_i}"
                    if candidate_chunk_i in fov_fill_chunks:
                        title += " FOV-FILL"
                    elif candidate_chunk_i in qwen_selected_chunks or option in selected_option_set:
                        title += " QWEN-SELECTED"
                    elif is_selected:
                        title += " SELECTED"
                    tiles.append(
                        {
                            "kind": "candidate",
                            "image": frames[0] if frames else None,
                            "title": title,
                            "lines": lines,
                            "selected": is_selected,
                            "acceptable": option in acceptable_options,
                        }
                    )

            add_candidate_tiles(
                qwen_info.get("coverage_candidate_chunks") or [],
                "",
                selected_options,
            )
            add_candidate_tiles(
                qwen_info.get("stable_candidate_chunks") or [],
                "S",
                {stable_selected} if stable_selected else set(),
            )
            for candidate_chunk_i in sorted(fov_fill_chunks - displayed_candidate_chunks):
                payload = self._build_qwen_chunk_payload(
                    candidate_chunk_i,
                    max_frames=1,
                    sample_mode="middle",
                )
                score = score_by_chunk.get(candidate_chunk_i, {})
                frames = payload.get("frames") if payload else []
                tiles.append(
                    {
                        "kind": "candidate",
                        "image": frames[0] if frames else None,
                        "title": f"chunk {candidate_chunk_i} FOV-FILL",
                        "lines": [
                            f"chunk={candidate_chunk_i} vq={payload.get('visual_quality_score') if payload else 'n/a'}",
                            f"overlap={float(score.get('overlap', 0.0)):.3f}"
                            if score
                            else "overlap=n/a",
                            "not selected by Qwen",
                            "added by FoV fallback",
                        ],
                        "selected": True,
                        "acceptable": False,
                    }
                )
            if not tiles:
                return None

            tile_w, image_h, label_h = 300, 170, 88
            tile_h = image_h + label_h
            cols, gap, margin, header_h = 3, 12, 18, 150
            rows = (len(tiles) + cols - 1) // cols
            width = margin * 2 + cols * tile_w + (cols - 1) * gap
            height = margin * 2 + header_h + rows * tile_h + (rows - 1) * gap
            sheet = Image.new("RGB", (width, height), (248, 248, 246))
            draw = ImageDraw.Draw(sheet)
            font = self._qwen_debug_font(18)
            small = self._qwen_debug_font(15)
            tiny = self._qwen_debug_font(13)
            bold = self._qwen_debug_font(18, bold=True)
            header_font = self._qwen_debug_font(24, bold=True)

            y = margin
            selected_text = ",".join(str(x) for x in sorted(selected_chunks)) or "none"
            qwen_selected_text = ",".join(sorted(selected_options)) or stable_selected or "none"
            fov_fill_text = ",".join(str(x) for x in sorted(fov_fill_chunks)) or "none"
            draw.text(
                (margin, y),
                f"Qwen select debug - target chunk {int(chunk_i):03d}",
                fill=(20, 20, 20),
                font=header_font,
            )
            y += 34
            draw.text(
                (margin, y),
                (
                    f"policy={trace_payload.get('selection_policy')} | local chunks={local_chunk_ids} | "
                    f"selected history chunks={selected_text} | qwen selected options={qwen_selected_text} | "
                    f"FoV fill chunks={fov_fill_text}"
                ),
                fill=(20, 20, 20),
                font=font,
            )
            y += 28
            for line in self._qwen_debug_wrap(
                "motion: " + str(qwen_info.get("target_motion_text") or ""),
                112,
            )[:3]:
                draw.text((margin, y), line, fill=(20, 20, 20), font=small)
                y += 20

            def draw_tile(x, y, tile):
                if tile["kind"] == "local":
                    border = (40, 88, 170)
                    fill = (246, 250, 255)
                elif tile.get("selected"):
                    border = (18, 136, 68)
                    fill = (246, 255, 248)
                else:
                    border = (185, 185, 185)
                    fill = (255, 255, 255)
                draw.rectangle(
                    [x, y, x + tile_w, y + tile_h],
                    fill=fill,
                    outline=border,
                    width=4 if tile["kind"] == "local" or tile.get("selected") else 1,
                )
                sheet.paste(
                    self._qwen_debug_thumb(tile.get("image"), (tile_w, image_h)),
                    (x, y),
                )
                label_y = y + image_h + 6
                draw.text((x + 8, label_y), str(tile["title"])[:42], fill=(20, 20, 20), font=bold)
                label_y += 24
                for line in tile["lines"][:4]:
                    draw.text((x + 8, label_y), str(line)[:58], fill=(20, 20, 20), font=tiny)
                    label_y += 17

            start_y = margin + header_h
            for idx, tile in enumerate(tiles):
                col = idx % cols
                row = idx // cols
                x = margin + col * (tile_w + gap)
                y = start_y + row * (tile_h + gap)
                draw_tile(x, y, tile)

            os.makedirs(os.path.dirname(sheet_path), exist_ok=True)
            tmp_path = f"{sheet_path}.tmp.{os.getpid()}"
            sheet.save(tmp_path, format="JPEG", quality=94)
            os.replace(tmp_path, sheet_path)
            return sheet_path
        except Exception as exc:
            self._qwen_log(
                f"[WAN Qwen Select DebugSheet Error] chunk={chunk_i} error={repr(exc)}"
            )
            return None

    def _wait_for_qwen_select_trace(self, chunk_i):
        trace_path = self._qwen_select_trace_path(chunk_i)
        if not trace_path:
            return None
        expected_run_token = str(os.getenv("WAN_QWEN_RUN_TOKEN", "")).strip()
        wait_sec = float(os.getenv("QWEN_MEMORY_SELECT_TRACE_WAIT_SEC", "600"))
        deadline = time.time() + max(1.0, wait_sec)
        last_error = None
        while time.time() < deadline:
            if os.path.exists(trace_path):
                try:
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace_payload = json.load(f)
                    if expected_run_token and str(
                        trace_payload.get("run_token", "")
                    ) != expected_run_token:
                        time.sleep(0.2)
                        continue
                    return trace_payload
                except Exception as exc:
                    last_error = exc
            time.sleep(0.2)
        detail = f"last_error={last_error!r}" if last_error is not None else "trace missing"
        raise TimeoutError(f"Timed out waiting for Qwen select trace {trace_path}; {detail}")

    def _qwen_temporal_local_frame_indices(
        self,
        current_frame_idx,
        chunk_size,
        local_context_chunks=None,
    ):
        if local_context_chunks is None:
            local_context_chunks = getattr(
                self,
                "qwen_memory_local_context_chunks",
                2,
            )
        temporal_context_size = max(0, int(chunk_size) * int(local_context_chunks))
        temporal_context_size = min(temporal_context_size, int(current_frame_idx))
        return list(
            range(
                max(0, int(current_frame_idx) - temporal_context_size),
                int(current_frame_idx),
            )
        )

    def _qwen_complete_memory_chunk_ids_by_fov(
        self,
        qwen_selected_chunk_ids,
        raw_top_memory_chunk_scores,
    ):
        select_top_k = max(0, int(getattr(self, "qwen_memory_select_top_k", 0)))
        qwen_selected = []
        for chunk_i in qwen_selected_chunk_ids or []:
            chunk_i = int(chunk_i)
            if chunk_i < 0 or chunk_i in qwen_selected:
                continue
            qwen_selected.append(chunk_i)
            if select_top_k > 0 and len(qwen_selected) >= select_top_k:
                break

        return qwen_selected, [], list(qwen_selected)

    def _wan_baseline_memory_frames(
        self,
        current_frame_idx,
        temporal_context_size=12,
        pred_latent_size=4,
    ):
        requested_top_k = max(1, int(getattr(self, "wan_memory_fov_top_k", 1)))
        available_top_k = max(
            0,
            (int(current_frame_idx) - int(temporal_context_size))
            // int(pred_latent_size),
        )
        selected_top_k = min(requested_top_k, available_top_k)
        return int(temporal_context_size) + selected_top_k * int(pred_latent_size)

    def _qwen_wan_max_selected_frames(self):
        raw = os.getenv(
            "QWEN_MEMORY_WAN_MAX_SELECTED_FRAMES",
            os.getenv("QWEN_MEMORY_MAX_SELECTED_FRAMES", "0"),
        )
        try:
            return max(0, int(str(raw).strip()))
        except Exception:
            return 0

    def _qwen_cap_selected_frame_indices(self, selected, local_frame_indices):
        max_frames = self._qwen_wan_max_selected_frames()
        selected = [int(x) for x in selected or []]
        if max_frames <= 0 or len(selected) <= max_frames:
            return selected, None

        local_set = {int(x) for x in local_frame_indices or []}
        local = [int(x) for x in selected if int(x) in local_set]
        extra = [int(x) for x in selected if int(x) not in local_set]
        if len(extra) >= max_frames:
            capped = extra[:max_frames]
        else:
            capped = extra + local[-(max_frames - len(extra)) :]
        return sorted(set(int(x) for x in capped)), max_frames

    def _select_qwen_memory_frame_indices(
        self,
        chunk_i,
        viewmats,
        action,
        current_frame_idx,
        context_window_length,
        chunk_size,
        n_latent,
        fallback_selected_frame_indices,
        device,
    ):
        if not bool(getattr(self, "qwen_memory_judge_enable", False)):
            return fallback_selected_frame_indices
        selected_container = [None]
        if self._is_rank0():
            try:
                local_frame_indices = self._qwen_temporal_local_frame_indices(
                    current_frame_idx=current_frame_idx,
                    chunk_size=chunk_size,
                )
                temporal_context_size = len(local_frame_indices)
                local_context_chunks = max(
                    1,
                    int(getattr(self, "qwen_memory_local_context_chunks", 2)),
                )
                local_chunk_ids = sorted(
                    {idx // int(chunk_size) for idx in local_frame_indices}
                )[-local_context_chunks:]
                viewmats_np = viewmats[0].detach().cpu().numpy()
                self._qwen_finalize_remote_eligible_memory_chunks(
                    current_chunk_i=int(chunk_i),
                    current_frame_idx=int(current_frame_idx),
                    temporal_context_size=temporal_context_size,
                    chunk_size=int(chunk_size),
                )
                raw_scores = self._qwen_fov_candidate_scores(
                    viewmats_np=viewmats_np,
                    current_frame_idx=int(current_frame_idx),
                    temporal_context_size=temporal_context_size,
                    pred_latent_size=int(chunk_size),
                    device=device,
                )
                self._qwen_update_candidate_scores_by_chunk[int(chunk_i)] = [
                    dict(item) for item in raw_scores
                ]
                current_action = (
                    action[
                        0,
                        int(current_frame_idx) : min(
                            int(current_frame_idx) + int(chunk_size), int(n_latent)
                        ),
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                qwen_info = None
                qwen_selected_chunk_ids = []
                if raw_scores:
                    qwen_info = self._qwen_select_memory_chunks(
                        current_chunk_i=int(chunk_i),
                        local_chunk_ids=local_chunk_ids,
                        raw_top_memory_chunk_scores=raw_scores,
                        current_chunk_action=current_action,
                    )
                    if qwen_info is not None:
                        qwen_selected_chunk_ids = list(
                            qwen_info.get("memory_chunk_ids", [])
                        )
                (
                    qwen_selected_chunk_ids,
                    fov_fill_chunk_ids,
                    selected_chunk_ids,
                ) = self._qwen_complete_memory_chunk_ids_by_fov(
                    qwen_selected_chunk_ids=qwen_selected_chunk_ids,
                    raw_top_memory_chunk_scores=raw_scores,
                )
                if qwen_info is not None:
                    qwen_info["qwen_selected_memory_chunk_ids"] = list(
                        qwen_selected_chunk_ids
                    )
                    qwen_info["fov_fill_memory_chunk_ids"] = list(
                        fov_fill_chunk_ids
                    )
                    qwen_info["memory_chunk_ids"] = list(selected_chunk_ids)
                selected = list(local_frame_indices)
                for selected_chunk_i in selected_chunk_ids:
                    start = int(selected_chunk_i) * int(chunk_size)
                    selected.extend(range(start, min(start + int(chunk_size), int(current_frame_idx))))
                selected = sorted(
                    {
                        int(idx)
                        for idx in selected
                        if 0 <= int(idx) < int(current_frame_idx)
                    }
                )
                selected_before_cap = list(selected)
                selected, selected_cap = self._qwen_cap_selected_frame_indices(
                    selected,
                    local_frame_indices,
                )
                trace_payload = {
                    "schema_version": 1,
                    "runtime_select_flow": "wan_qwen_split_select",
                    "run_token": str(os.getenv("WAN_QWEN_RUN_TOKEN", "")),
                    "selection_policy": getattr(
                        self, "qwen_memory_select_policy", "coverage_only"
                    ),
                    "chunk_i": int(chunk_i),
                    "current_frame_idx": int(current_frame_idx),
                    "local_context_chunks": int(local_context_chunks),
                    "local_chunk_ids": [int(x) for x in local_chunk_ids],
                    "local_frame_indices": [int(x) for x in local_frame_indices],
                    "raw_top_memory_chunk_scores": raw_scores,
                    "active_memory_chunk_ids": sorted(
                        int(x) for x in getattr(self, "_qwen_active_memory_chunk_ids", set())
                    ),
                    "qwen_selected_memory_chunk_ids": [
                        int(x) for x in qwen_selected_chunk_ids
                    ],
                    "fov_fill_memory_chunk_ids": [
                        int(x) for x in fov_fill_chunk_ids
                    ],
                    "fov_fill_min_chunks": int(
                        getattr(self, "qwen_memory_fov_fill_min_chunks", 0)
                    ),
                    "selected_memory_chunk_ids": [int(x) for x in selected_chunk_ids],
                    "selected_frame_indices": selected,
                    "selected_frame_indices_before_cap": selected_before_cap,
                    "selected_frame_cap": selected_cap,
                    "selected_frame_cap_policy": (
                        "preserve_selected_memory_then_recent_local"
                        if selected_cap
                        else None
                    ),
                    "allow_fallback": False,
                    "fallback_used": False,
                    "fallback_selected_frame_indices": [
                        int(x) for x in (fallback_selected_frame_indices or [])
                    ],
                    "qwen_info": qwen_info,
                }
                trace_path = self._save_qwen_select_trace(chunk_i, trace_payload)
                debug_sheet_path = self._save_qwen_select_debug_sheet(
                    chunk_i,
                    trace_payload,
                )
                self._qwen_log(
                    f"[WAN Qwen Select] chunk={chunk_i} local={local_chunk_ids} "
                    f"policy={getattr(self, 'qwen_memory_select_policy', 'coverage_only')} "
                    f"qwen_selected={qwen_selected_chunk_ids} "
                    f"fov_fill={fov_fill_chunk_ids} "
                    f"selected_chunks={selected_chunk_ids} frames={len(selected)} "
                    f"cap={selected_cap} "
                    f"trace={trace_path} debug_sheet={debug_sheet_path}"
                )
                selected_container[0] = selected
            except Exception as exc:
                error_payload = {
                    "schema_version": 1,
                    "runtime_select_flow": "wan_qwen_split_select",
                    "run_token": str(os.getenv("WAN_QWEN_RUN_TOKEN", "")),
                    "chunk_i": int(chunk_i), "qwen_error": repr(exc),
                }
                self._qwen_log(f"[WAN Qwen Select Error] chunk={chunk_i} error={repr(exc)}")
                self._save_qwen_select_trace(chunk_i, error_payload)
                selected_container[0] = {"__qwen_error__": repr(exc)}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            trace_path = self._qwen_select_trace_path(chunk_i)
            if trace_path:
                if not self._is_rank0():
                    try:
                        trace_payload = self._wait_for_qwen_select_trace(chunk_i)
                        if isinstance(trace_payload, dict) and trace_payload.get("qwen_error"):
                            selected_container[0] = {
                                "__qwen_error__": trace_payload["qwen_error"]
                            }
                        elif isinstance(trace_payload, dict):
                            selected_container[0] = trace_payload.get("selected_frame_indices")
                    except Exception as exc:
                        raise RuntimeError(
                            f"Qwen select trace synchronization failed: {exc!r}"
                        ) from exc
            else:
                torch.distributed.broadcast_object_list(selected_container, src=0)
        selected = selected_container[0]
        if isinstance(selected, dict) and "__qwen_error__" in selected:
            raise RuntimeError(
                f"Qwen select failed in strict mode: {selected['__qwen_error__']}"
            )
        if not selected:
            selected = self._qwen_temporal_local_frame_indices(
                current_frame_idx=current_frame_idx, chunk_size=chunk_size)
        selected = [int(x) for x in selected]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return selected

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}"
            )

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError(
                "`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def init_kv_cache(self):
        self._kv_cache = []
        self._kv_cache_neg = []
        transformer_num_layers = len(self.transformer.blocks)
        for i in range(transformer_num_layers):
            self._kv_cache.append({"k": None, "v": None})
            self._kv_cache_neg.append({"k": None, "v": None})

        self.use_kv_cache = True

    def resize_center_crop_to_tensor(self, img_path, target_size):
        """
        Read image -> Resize -> Center crop -> Convert to torch.tensor -> Normalize to [-1, 1]
        :param img_path: Path to the image
        :param target_size: (width, height)
        :return: torch.Tensor, shape [C, H, W], range [-1, 1]
        """
        img = Image.open(img_path).convert("RGB")
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # --- Step 1: proportional resize ---
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # --- Step 2: center crop ---
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))

        # --- Step 3: to numpy，and then tensor ---
        img_np = np.array(img).astype(np.float32)  # [H, W, C], 0~255
        img_np = img_np / 127.5 - 1.0  # normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]

        return img_tensor

    def init_decode_state(self, all_latents, chunk_i):
        """
        Initialize the decoding state and set all latents required for decoding.
        """
        self._decode_state["total_latents"] = all_latents.shape[2]  # all latent numbers
        self._decode_state["all_latents"] = all_latents
        self._decode_state["current_latent_idx"] = 0
        self._decode_state["chunk_i"] = chunk_i
        assert self._decode_state["par_vae_decode"]

        if self._decode_state["latents_mean"] is None:
            latents_mean = (
                torch.tensor(self.dist_vae.pipe.config.latents_mean)
                .view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1)
                .to(all_latents.device, all_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(
                self.dist_vae.pipe.config.latents_std
            ).view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1).to(
                all_latents.device, all_latents.dtype
            )

            self._decode_state["latents_mean"] = latents_mean
            self._decode_state["latents_std"] = latents_std

    def _pad_decode_latents_for_parallel_width(self, latents, num_gpus):
        original_width = latents.shape[4]
        remainder = original_width % num_gpus
        if remainder == 0:
            return latents, original_width

        pad_width = num_gpus - remainder
        pad_values = latents[..., -1:].expand(*latents.shape[:-1], pad_width)
        return torch.cat([latents, pad_values], dim=4), original_width

    def _crop_parallel_decode_video_width(self, video, original_latent_width):
        target_width = original_latent_width * self.vae_scale_factor_spatial
        if video.shape[4] > target_width:
            video = video[..., :target_width]
        return video

    def decode_next_latent(self, output_type="np"):
        """
        Decode the next latents and return the decoded frames.
        If all latents have been decoded, return None.
        """

        current_idx = self._decode_state["current_latent_idx"]
        total_latents = self._decode_state["total_latents"]

        if current_idx >= total_latents:
            return None  # all latents have been decoded

        # Get the current latent to be decoded.
        all_latents = self._decode_state["all_latents"]
        chunk_i = self._decode_state["chunk_i"]
        current_latent = all_latents[
            :, :, current_idx : current_idx + 1, :, :
        ]  # [B, C, 1, H, W]

        sp_world_size = self._decode_state["sp_world_size"]
        rank_in_sp_group = self._decode_state["rank_in_sp_group"]
        latents_mean = self._decode_state["latents_mean"]
        latents_std = self._decode_state["latents_std"]

        # Multi-GPU parallel VAE decoding
        current_latent = current_latent.to(self.dist_vae.pipe.dtype)
        # Ensure latents_mean and latents_std have the same type as current_latent
        latents_mean = latents_mean.to(current_latent.dtype)
        latents_std = latents_std.to(current_latent.dtype)
        current_latent = current_latent / latents_std + latents_mean

        local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]
        current_latent, original_latent_width = self._pad_decode_latents_for_parallel_width(
            current_latent, sp_world_size
        )

        decode_input = distribute_data_to_gpus(
            current_latent,
            dim=4,
            rank=rank_in_sp_group,
            local_rank=local_rank_in_sp_group_vae,
            num_gpus=sp_world_size,
            dtype=current_latent.dtype,
        )
        video = self.dist_vae.pipe.decode(
            decode_input,
            return_dict=False,
            is_first_chunk=(current_idx == 0 and chunk_i == 0),
        )
        video = gather_results(
            video[0],
            dim=4,
            world_size=sp_world_size,
        )
        video = self._crop_parallel_decode_video_width(video, original_latent_width)

        video = self.video_processor.postprocess_video(video, output_type=output_type)

        torch.cuda.synchronize()

        # Update the index
        self._decode_state["current_latent_idx"] += 1

        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        few_step=False,
        first_chunk_size=1,
        image_path=None,
        viewmats=None,
        Ks=None,
        action=None,
        use_memory=True,
        context_window_length=None,  # mean the context length for each chunk
        chunk_i: int = 0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor`
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if chunk_i == 0:
            # print("init run")
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                negative_prompt,
                height,
                width,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
                guidance_scale_2,
            )

            if num_frames % self.vae_scale_factor_temporal != 1:
                logger.warning(
                    "`num_frames - 1` has to be divisible by %d. "
                    "Rounding to the nearest number.",
                    self.vae_scale_factor_temporal,
                )
                num_frames = (
                    num_frames
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
            num_frames = max(num_frames, 1)

            if self.config.boundary_ratio is not None and guidance_scale_2 is None:
                guidance_scale_2 = guidance_scale

            self._guidance_scale = guidance_scale
            self._guidance_scale_2 = guidance_scale_2
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            device = self._execution_device

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            self.points_local = generate_points_in_sphere(50000, 8.0).to(device)

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            transformer_dtype = (
                self.transformer.dtype
                if self.transformer is not None
                else self.transformer_2.dtype
            )
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            # 4. Prepare timesteps
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])
            else:
                sigmas = self.scheduler.sigmas
                timesteps = torch.tensor(
                    [1000.0000, 960.0000, 888.8889, 727.2728, 0.0]
                ).to(sigmas.device)
                sigmas = timesteps / 1000.0

            # 5. Prepare latent variables
            num_channels_latents = (
                self.transformer.config.in_channels
                if self.transformer is not None
                else self.transformer_2.config.in_channels
            )
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

            self._num_timesteps = len(timesteps)
            chunk_size = CHUNK_SIZE
            n_latent = latents.shape[2]

            # set the noise level for the inference stage
            stabilization_level = 15

            sp_world_size = 1
            rank_in_sp_group = 0
            local_rank_in_sp_group = 0
            if self.parallel_run:
                sp_world_size = get_sp_world_size()
                rank_in_sp_group = get_sp_parallel_rank()
                local_rank_in_sp_group = get_sp_parallel_local_rank()

            first_image_condition = None
            if image_path:
                first_image = self.resize_center_crop_to_tensor(image_path, (width, height)) # (B C T H W)
                first_image = first_image.to(device).to(self.vae.dtype)
                first_image = first_image.unsqueeze(0).unsqueeze(2) # [B C T H W]
                first_image_condition = self.vae.encode(first_image, return_dict=False)[0].sample()
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                first_image_condition = (first_image_condition - latents_mean) * latents_std


            self.use_kv_cache = False
            self.init_kv_cache()

            self._decode_state = {
                "current_latent_idx": 0,
                "total_latents": None,
                "device": device,
                "transformer_dtype": transformer_dtype,
                "vae": self.vae,
                "video_processor": self.video_processor,
                "parallel_run": self.parallel_run,
                "par_vae_decode": self.par_vae_decode,
                "sp_world_size": sp_world_size,
                "rank_in_sp_group": rank_in_sp_group,
                "latents_mean": None,
                "latents_std": None,
                "is_initialized": False,
            }


            self.ctx = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "attention_kwargs": attention_kwargs,
                "transformer_dtype": (
                    self.transformer.dtype
                    if self.transformer is not None
                    else self.transformer_2.dtype
                ),
                # Denoising scheduler and state
                "timesteps": timesteps,
                "num_inference_steps": num_inference_steps,
                "latents": latents,
                "first_image_condition": first_image_condition,
                "chunk_size": chunk_size,
                "first_chunk_size": first_chunk_size,
                "n_latent": n_latent,
                "viewmats": torch.zeros(
                    1, n_latent, 4, 4, device=latents.device, dtype=latents.dtype
                ),
                "Ks": torch.zeros(
                    1, n_latent, 3, 3, device=latents.device, dtype=latents.dtype
                ),
                "action": torch.zeros(
                    1, n_latent, device=latents.device, dtype=latents.dtype
                ),
                "use_memory": use_memory,
                "context_window_length": context_window_length,
                "kv_cache": self._kv_cache,
                "kv_cache_neg": self._kv_cache_neg,
                "use_kv_cache": self.use_kv_cache,
                # Distributed related
                "sp_world_size": sp_world_size,
                "rank_in_sp_group": rank_in_sp_group,
                "local_rank_in_sp_group": local_rank_in_sp_group,
                # Other auxiliary parameters
                "device": device,
                "batch_size": batch_size,
                "few_step": few_step,
                "sigmas": sigmas if few_step else None,
                "stabilization_level": stabilization_level,
            }

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # ==========================================================
        if self.ctx is None and chunk_i > 0:
            raise ValueError(
                "When chunk_i > 0, the context self.ctx cannot be empty. Please ensure the call starts from chunk_i = 0."
            )
        prompt_embeds = self.ctx["prompt_embeds"]
        negative_prompt_embeds = self.ctx["negative_prompt_embeds"]
        guidance_scale = self.ctx["guidance_scale"]
        guidance_scale_2 = self.ctx["guidance_scale_2"]
        attention_kwargs = self.ctx["attention_kwargs"]
        transformer_dtype = self.ctx["transformer_dtype"]

        timesteps = self.ctx["timesteps"]
        num_inference_steps = self.ctx["num_inference_steps"]
        latents = self.ctx["latents"]
        first_image_condition = self.ctx["first_image_condition"]

        chunk_size = self.ctx["chunk_size"]
        first_chunk_size = self.ctx["first_chunk_size"]
        n_latent = self.ctx["n_latent"]
        stabilization_level = self.ctx["stabilization_level"]

        # Fill the poses of the current chunk into the corresponding positions.
        start_idx = chunk_i * chunk_size
        end_idx = start_idx + chunk_size

        self.ctx["viewmats"][:, start_idx:end_idx, ...] = viewmats
        self.ctx["Ks"][:, start_idx:end_idx, ...] = Ks
        self.ctx["action"][:, start_idx:end_idx, ...] = action

        viewmats = self.ctx["viewmats"]
        Ks = self.ctx["Ks"]
        action = self.ctx["action"]

        use_memory = self.ctx["use_memory"]
        context_window_length = self.ctx["context_window_length"]
        self._kv_cache = self.ctx["kv_cache"]
        self._kv_cache_neg = self.ctx["kv_cache_neg"]
        self.use_kv_cache = self.ctx["use_kv_cache"]

        sp_world_size_vae = self.ctx["sp_world_size"]
        rank_in_sp_group_vae = self.ctx["rank_in_sp_group"]
        local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]

        device = self.ctx["device"]
        batch_size = self.ctx["batch_size"]
        few_step = self.ctx["few_step"]
        sigmas = self.ctx["sigmas"]


        # Restore the attributes of the pipeline
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = len(timesteps)

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = first_chunk_size
            generate_latent_num = first_chunk_size
            if first_image_condition is not None:
                latents[:, :, :1] = first_image_condition
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
        else:
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])

            already_generate_num = chunk_i * chunk_size + first_chunk_size
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
            generate_latent_num = chunk_size

            # add memory selection here
            current_frame_idx = chunk_i * chunk_size  # the current frame index to generate
            if (
                use_memory
                and (current_frame_idx >= context_window_length)
                and current_frame_idx < n_latent
            ):
                if bool(getattr(self, "qwen_memory_judge_enable", False)):
                    fallback_selected_frame_indices = None
                    selected_frame_indices = self._select_qwen_memory_frame_indices(
                        chunk_i=chunk_i,
                        viewmats=viewmats,
                        action=action,
                        current_frame_idx=current_frame_idx,
                        context_window_length=context_window_length,
                        chunk_size=chunk_size,
                        n_latent=n_latent,
                        fallback_selected_frame_indices=fallback_selected_frame_indices,
                        device=device,
                    )
                else:
                    temporal_context_size = 12
                    pred_latent_size = 4
                    memory_frames = self._wan_baseline_memory_frames(
                        current_frame_idx=current_frame_idx,
                        temporal_context_size=temporal_context_size,
                        pred_latent_size=pred_latent_size,
                    )
                    selected_frame_indices = select_mem_frames_wan(
                        viewmats[0].cpu().detach().numpy(),
                        current_frame_idx,
                        memory_frames=memory_frames,
                        temporal_context_size=temporal_context_size,
                        pred_latent_size=pred_latent_size,
                        points_local=self.points_local,
                        device=self.device,
                    )
                    if self._is_rank0():
                        local_start = max(
                            0, int(current_frame_idx) - temporal_context_size
                        )
                        local_chunk_ids = sorted(
                            {
                                int(idx) // pred_latent_size
                                for idx in range(local_start, int(current_frame_idx))
                            }
                        )
                        history_chunk_ids = sorted(
                            {
                                int(idx) // pred_latent_size
                                for idx in selected_frame_indices
                                if int(idx) < local_start
                            }
                        )
                        self._qwen_log(
                            f"[WAN Baseline FoV Select] chunk={chunk_i} "
                            f"local={local_chunk_ids} "
                            f"requested_top_k={self.wan_memory_fov_top_k} "
                            f"selected_history_chunks={history_chunk_ids} "
                            f"frames={len(selected_frame_indices)}"
                        )

            elif use_memory:
                selected_frame_indices = list(range(0, current_frame_idx))

        with self.progress_bar(total=4) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue

                self._current_timestep = t

                current_model = self.transformer
                current_guidance_scale = guidance_scale

                if chunk_i > 0:  # context latent frame
                    t_now = torch.full((batch_size, chunk_size), t, device=device, dtype=timesteps.dtype)
                    t_ctx = torch.full((batch_size, first_chunk_size + (chunk_i - 1) * chunk_size),
                        stabilization_level - 1,
                        device=device,
                        dtype=timesteps.dtype,
                    )  # [B, F]
                    timestep = torch.cat([t_ctx, t_now], dim=1)  # [B, F+1]: predict the next frame in timestep t
                else:
                    if first_image_condition is not None:
                        t_now = torch.full((batch_size, chunk_size - 1), t, device=device, dtype=timesteps.dtype)
                        t_ctx = torch.full((batch_size, 1), stabilization_level - 1,
                            device=device,
                            dtype=timesteps.dtype,
                        )  # [B, F]
                        timestep = torch.cat([t_ctx, t_now], dim=1)
                    else:
                        timestep = torch.full((batch_size, first_chunk_size), t,
                            device=device,
                            dtype=timesteps.dtype,
                        )  # [B, 1]

                all_window_latent_num = latents_curr.shape[2]
                model_patch_size = getattr(current_model.config, "patch_size", (1, 2, 2))
                _, patch_h, patch_w = model_patch_size
                rope_tokens_per_latent = (latents_curr.shape[-2] // patch_h) * (latents_curr.shape[-1] // patch_w)

                ## kv cache
                if chunk_i > 0 and i == 0 and self.use_kv_cache:
                    latents_cache = latents_curr.clone()
                    latents_cache = latents_cache[:, :, selected_frame_indices]
                    t_cache = timestep[:, selected_frame_indices]

                    latent_model_cache = latents_cache.to(transformer_dtype)
                    # flatten the timestep for the transformer
                    timestep_cache = t_cache.flatten()
                    if action is not None:
                        action_cache = action[:, selected_frame_indices]
                        viewmats_cache = viewmats[:, selected_frame_indices]
                        Ks_cache = Ks[:, selected_frame_indices]

                    kv_start_rope = 0
                    kv_end_rope = len(selected_frame_indices)

                    torch.cuda.nvtx.range_push("kv_cache")
                    self._kv_cache = current_model(
                        hidden_states=latent_model_cache,
                        timestep=timestep_cache,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        current_start=kv_start_rope * rope_tokens_per_latent,
                        current_end=kv_end_rope * rope_tokens_per_latent,
                        kv_cache=self._kv_cache,
                        is_cache=True,
                        window_frames=n_latent,
                        viewmats=(viewmats_cache if viewmats is not None else None),
                        Ks=Ks_cache if Ks is not None else None,
                        action=action_cache if action is not None else None,
                    )

                torch.cuda.nvtx.range_pop()

                if not self.use_kv_cache:
                    generate_latent_num = all_window_latent_num

                if selected_frame_indices is not None:
                    now_window_latent_num = len(selected_frame_indices) + chunk_size
                else:
                    now_window_latent_num = chunk_size

                latent_model_input = latents_curr.clone()
                latent_model_input = latent_model_input[:, :, -generate_latent_num:].to(transformer_dtype)
                timestep = timestep[:, -generate_latent_num:]
                timestep = timestep.flatten()

                generate_rope_start = (now_window_latent_num - generate_latent_num) * rope_tokens_per_latent
                generate_rope_end = now_window_latent_num * rope_tokens_per_latent
                torch.cuda.nvtx.range_push("noise_pred")
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    current_start=generate_rope_start,
                    current_end=generate_rope_end,
                    kv_cache=self._kv_cache if self.use_kv_cache else None,
                    is_cache=False,
                    window_frames=n_latent,
                    viewmats=viewmats[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                    Ks=Ks[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                    action=action[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                )[0]

                torch.cuda.nvtx.range_pop()

                # compute the previous noisy sample x_t -> x_t-1
                if chunk_i == 0:
                    if not few_step:
                        latent_model_input = self.scheduler.step( noise_pred, t, latent_model_input, return_dict=False)[0]
                        if first_image_condition is not None:
                            latents_curr[:, :, -first_chunk_size + 1:] = latent_model_input[:, :, 1:]
                        else:
                            latents_curr[:, :, -first_chunk_size:] = latent_model_input
                    else:
                        sigma = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        dt = sigma_next - sigma
                        prev_sample = latent_model_input + dt * noise_pred
                        if first_image_condition is not None:
                            latents_curr[:, :, -first_chunk_size + 1:] = prev_sample[:, :, 1:]
                        else:
                            latents_curr[:, :, -first_chunk_size:] = prev_sample
                else:
                    noise_pred_chunk = noise_pred[:, :, -chunk_size:]  # [B, C, chunk_size, H, W
                    latents_curr_chunk = latent_model_input[:, :, -chunk_size:]  # [B, C, chunk_size, H, W]
                    if not few_step:
                        latent_curr_pred_chunk = self.scheduler.step(
                            noise_pred_chunk,
                            t,
                            latents_curr_chunk,
                            return_dict=False,
                        )[0]  # [B, C, chunk_size, H, W]
                    else:
                        sigma = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        dt = sigma_next - sigma
                        latent_curr_pred_chunk = latents_curr_chunk + dt * noise_pred_chunk

                    latents_curr[:, :, -chunk_size:] = latent_curr_pred_chunk  # update the latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents[:, :, :already_generate_num, :, :] = latents_curr
        # ==========================================================
        # Save the state updated during this call back to self.ctx
        self.ctx["latents"] = latents
        self.ctx["kv_cache"] = self._kv_cache
        self.ctx["kv_cache_neg"] = self._kv_cache_neg
        # ==========================================================

        re_latents = latents
        self._current_timestep = None
        latents = latents[:, :, start_idx:end_idx, :, :]

        # Automatically initialize the decoding state (for subsequent frame-by-frame decoding)
        # Note: Must be initialized here because the latents will be modified later
        if output_type == "latent":
            self.init_decode_state(latents, chunk_i)

        torch.cuda.synchronize()
        vae_decode_begin_time = time.time()
        if not output_type == "latent":
            if not self.par_vae_decode:
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video = self.vae.decode(
                    latents, return_dict=False, is_first_chunk=chunk_i == 0
                )[0]
            else:
                latents = latents.to(self.dist_vae.pipe.dtype)
                latents_mean = (
                    torch.tensor(self.dist_vae.pipe.config.latents_mean)
                    .view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(
                    self.dist_vae.pipe.config.latents_std
                ).view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean

                # Accelerated VAE
                latents, original_latent_width = self._pad_decode_latents_for_parallel_width(
                    latents, sp_world_size_vae
                )
                decode_input = distribute_data_to_gpus(
                    latents,
                    dim=4,
                    rank=rank_in_sp_group_vae,
                    local_rank=local_rank_in_sp_group_vae,
                    num_gpus=sp_world_size_vae,
                    dtype=latents.dtype,
                )
                video = self.dist_vae.pipe.decode(
                    decode_input, return_dict=False, is_first_chunk=chunk_i == 0
                )
                video = gather_results(
                    video[0],
                    dim=4,
                    world_size=sp_world_size_vae,
                )
                video = self._crop_parallel_decode_video_width(video, original_latent_width)
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )
        else:
            return None, None

        torch.cuda.synchronize()
        print("Vae decode time", time.time() - vae_decode_begin_time)
        torch.cuda.nvtx.range_pop()

        if not return_dict:
            return (
                video,
                re_latents,
            )

        return WanPipelineOutput(frames=video)
