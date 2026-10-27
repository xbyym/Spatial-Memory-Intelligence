"""Exact active prompt text and result normalization from the frozen reference."""
import json
import os
from pathlib import Path
from .qwen_results import normalize_visual_quality, normalize_create, normalize_sparse, normalize_coverage

PROMPTS = Path(__file__).with_name("qwen_prompts")
CREATE = (PROMPTS / 'hy15_qwen_memory_create_runtime_prompt.txt').read_text()
SPARSE = (PROMPTS / 'hy15_qwen_memory_sparse_runtime_prompt.txt').read_text()
COVERAGE = (PROMPTS / 'hy15_qwen_memory_select_coverage_motion_groups_lastframe_prompt.txt').read_text()
VISUAL_QUALITY = (PROMPTS / "visual_quality.txt").read_text().removesuffix("\n")

def add_chunk_images(content, title, chunk):
    if chunk.get('hide_metadata', False):
        content.append({'type': 'text', 'text': f'{title}:'})
        for path in chunk.get('frames', []):
            content.append({'type': 'image', 'url': str(path)})
        return
    chunk_i = chunk.get('chunk_i', chunk.get('id', 'unknown'))
    cluster_id = chunk.get('cluster_id', None)
    prototype_status = chunk.get('prototype_status', None)
    visual_quality_score = chunk.get('visual_quality_score', None)
    meta = [f'chunk={chunk_i}']
    if cluster_id is not None:
        meta.append(f'cluster={cluster_id}')
    if prototype_status is not None:
        meta.append(f'prototype_status={prototype_status}')
    if visual_quality_score is not None:
        meta.append(f'visual_quality_score={int(float(visual_quality_score))}')
    content.append({'type': 'text', 'text': f"{title} ({', '.join(meta)}), frames in temporal order:"})
    for path in chunk.get('frames', []):
        content.append({'type': 'image', 'url': str(path)})

def _json_compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def _normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'y', 'match', 'matched', '是'}:
        return True
    if text in {'false', '0', 'no', 'n', 'not_match', 'unmatched', '否'}:
        return False
    return bool(default)

def _normalize_visual_quality_result(result):
    return normalize_visual_quality(result)

def _create_score_to_legacy_overlap(score):
    if score <= 1:
        return 'none'
    if score <= 3:
        return 'weak'
    if score == 4:
        return 'medium'
    return 'strong'

def _normalize_create_result(result):
    return normalize_create(result)

def _normalize_sparse_result(result):
    return normalize_sparse(result)

def _normalize_select_coverage_compact_result(result, candidates):
    return normalize_coverage(result, candidates, max_selected=2)

def _build_visual_quality_content(payload):
    candidate = payload.get('generated_chunk', None) or payload.get('candidate', None) or payload.get('current_chunk', None) or {}
    content = [{'type': 'text', 'text': VISUAL_QUALITY}]
    add_chunk_images(content, 'Generated frame', candidate)
    return content

def _build_create_content(payload):
    current = payload.get('current_chunk', {}) or {}
    candidate_class = payload.get('candidate_class', {}) or {}
    prototype = payload.get('prototype', None) or candidate_class.get('prototype', {}) or {}
    class_id = payload.get('class_id', candidate_class.get('cluster_id', None))
    try:
        prototype_fov_overlap = float(payload.get('prototype_fov_overlap', candidate_class.get('prototype_fov_overlap', prototype.get('fov_overlap', 0.0))) or 0.0)
    except Exception:
        prototype_fov_overlap = 0.0
    content = [{'type': 'text', 'text': f"{CREATE}\n\nCurrent chunk_id={current.get('chunk_i', 'unknown')}\nCandidate class_id={class_id}\nPrototype chunk_id={prototype.get('chunk_i', 'unknown')}\nprototype_fov_overlap={prototype_fov_overlap:.4f}"}]
    add_chunk_images(content, 'Current middle frame', current)
    add_chunk_images(content, 'Prototype middle frame', prototype)
    return content

def _build_sparse_content(payload):
    candidate = payload.get('candidate_chunk', {}) or {}
    remaining = payload.get('remaining_chunks', []) or []
    if 'coverage_by_remaining_chunks' not in payload:
        raise ValueError('sparse payload missing coverage_by_remaining_chunks')
    try:
        coverage_by_remaining_chunks = float(payload['coverage_by_remaining_chunks'])
    except Exception as exc:
        raise ValueError(f"invalid sparse coverage_by_remaining_chunks: {payload.get('coverage_by_remaining_chunks')!r}") from exc
    content = [{'type': 'text', 'text': f"{SPARSE}\n\nclass_id={payload.get('cluster_id', payload.get('class_id', None))}\ncandidate_chunk_id={candidate.get('chunk_i', payload.get('candidate_chunk_i', 'unknown'))}\nremaining_chunk_ids={_json_compact(payload.get('remaining_chunk_ids', []))}\ncoverage_by_remaining_chunks={coverage_by_remaining_chunks:.4f}\nscenario_assumed_removed_chunk_ids={_json_compact(payload.get('scenario_assumed_removed_chunk_ids', []))}"}]
    for chunk in remaining:
        add_chunk_images(content, 'Remaining class middle frame', chunk)
    add_chunk_images(content, 'Candidate removal middle frame', candidate)
    return content

def _build_select_content(payload, instruction, candidate_key):
    current = payload.get('current_chunk', None) or (payload.get('local_chunks', []) or [{}])[-1] or {}
    candidates = payload.get(candidate_key, None) or payload.get('candidates', []) or []
    motion_text = payload.get('planned_camera_motion', payload.get('target_motion_text', ''))
    if candidate_key == 'coverage_candidates':
        target_next_chunk_id = payload.get('target_next_chunk_id', payload.get('current_chunk_i', 'unknown'))
        lines = [instruction, '']
        lines.extend([f"current_chunk_id: {current.get('chunk_i', 'unknown')}", f'target_next_chunk_id: {target_next_chunk_id}', f'Planned camera motion: {motion_text}', 'Candidates:'])
        for (idx, candidate) in enumerate(candidates):
            option = str(candidate.get('option_label', chr(ord('A') + idx)))
            visual_quality_score = candidate.get('visual_quality_score', None)
            lines.append(f"{option}: chunk_id={candidate.get('chunk_i')} visual_quality_score={visual_quality_score}")
        lines.append('Image: Current')
        for (idx, candidate) in enumerate(candidates):
            option = str(candidate.get('option_label', chr(ord('A') + idx)))
            lines.append(f'Image: Candidate-{option}')
    else:
        lines = [instruction, '', f"Current chunk_id={current.get('chunk_i', 'unknown')}", f'Planned camera motion: {motion_text}', 'Candidates:']
        for (idx, candidate) in enumerate(candidates):
            option = str(candidate.get('option_label', chr(ord('A') + idx)))
            visual_quality_score = candidate.get('visual_quality_score', None)
            class_id = candidate.get('class_id', candidate.get('cluster_id', None))
            class_text = 'null' if class_id is None else str(class_id)
            lines.append(f"- option={option}; chunk_id={candidate.get('chunk_i')}; class_id={class_text}; visual_quality_score={visual_quality_score}")
    content = [{'type': 'text', 'text': '\n'.join(lines)}]
    add_chunk_images(content, 'Current frame', {**current, 'hide_metadata': True})
    for (idx, candidate) in enumerate(candidates):
        option = str(candidate.get('option_label', chr(ord('A') + idx)))
        add_chunk_images(content, f'Candidate-{option}', {**candidate, 'hide_metadata': True})
    return content
