"""Validate model decisions without inventing a replacement memory operation."""
import json
import math
import re


def skipped_result(error):
    return dict(skipped=True, reason='invalid_model_output', error=str(error))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_result(raw):
    text = re.sub(r'^\s*<think>.*?</think>\s*', '', raw, flags=re.S).strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.S | re.I)
    if fence:
        text = fence.group(1)
    result = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError('model output must be a JSON object')
    return result


def _score(result, key, low, high):
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{key} must be a number')
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{key} must be in [{low}, {high}]')
    return int(round(value))


def normalize_visual_quality(result):
    return dict(visual_quality_score=_score(result, 'visual_quality_score', 1, 10))


def normalize_create(result):
    score = _score(result, 'view_overlap_score', 1, 6)
    matched = score >= 3
    for key in ('match', 'prototype_match'):
        if key in result and (not isinstance(result[key], bool) or result[key] != matched):
            raise ValueError(f'{key} contradicts view_overlap_score')
    overlap = 'none' if score <= 1 else 'weak' if score <= 3 else 'medium' if score == 4 else 'strong'
    return dict(prototype_match=matched, view_overlap_score=score,
                view_overlap=overlap, reason=str(result.get('reason', ''))[:240])


def normalize_sparse(result):
    choices = dict(action={'remove', 'keep'},
                   spatial_coverage={'none', 'weak', 'medium', 'strong', 'full'},
                   unique_useful_information={'none', 'small', 'clear'})
    normalized = {}
    for key, allowed in choices.items():
        value = result.get(key)
        if not isinstance(value, str) or value.strip().lower() not in allowed:
            raise ValueError(f'invalid or missing {key}')
        normalized[key] = value.strip().lower()
    return dict(normalized, reason=str(result.get('reason', ''))[:240])


def coverage_options(result, candidates, max_selected):
    """Return only explicit selections; an empty list is a valid abstention."""
    valid = [str(item.get('option_label', chr(65 + i))) for i, item in enumerate(candidates)]
    valid_set = set(valid)

    def labels(value, name):
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError(f'{name} must be a list of labels')
        if len(value) != len(set(value)) or not set(value) <= valid_set:
            raise ValueError(f'{name} contains duplicate or unknown labels')
        return value

    selected_key = 'selected_coverage' if 'selected_coverage' in result else 'selected'
    acceptable_key = 'acceptable_coverage' if 'acceptable_coverage' in result else 'acceptable'
    selected = labels(result.get(selected_key), selected_key)
    acceptable = labels(result.get(acceptable_key), acceptable_key)
    if not set(selected) <= set(acceptable) or len(selected) > max_selected:
        raise ValueError('selected must be a subset of acceptable within the selection cap')
    motion = result.get('motion')
    if not isinstance(motion, dict) or set(motion) != {'supports', 'neutral', 'opposes'}:
        raise ValueError('motion must contain supports, neutral, and opposes')
    flattened = [x for name in ('supports', 'neutral', 'opposes')
                 for x in labels(motion[name], f'motion.{name}')]
    if len(flattened) != len(valid) or set(flattened) != valid_set:
        raise ValueError('motion groups must partition all displayed candidates')
    return valid, acceptable, selected, motion


def normalize_coverage(result, candidates, max_selected):
    valid, acceptable, selected, motion = coverage_options(result, candidates, max_selected)
    alignment = {option: name for name, options in motion.items() for option in options}
    return dict(
        per_candidate=[dict(option=option, motion_alignment=alignment[option],
                            acceptable=option in acceptable, selected=option in selected,
                            evidence='') for option in valid],
        motion=motion,
        motion_compatible=[option for option in valid if alignment[option] != 'opposes'],
        acceptable_coverage=list(acceptable), selected_coverage=list(selected))


def decode_result(task, raw, candidates=(), max_selected=2):
    try:
        result = parse_result(raw)
        if task == 'select_coverage':
            return normalize_coverage(result, candidates, max_selected)
        return dict(visual_quality=normalize_visual_quality,
                    create=normalize_create, sparse=normalize_sparse)[task](result)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return skipped_result(exc)
