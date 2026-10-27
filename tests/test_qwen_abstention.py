"""CPU regressions for literal model choices and memory-operation abstention."""
import ast
import copy
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_results', ROOT / 'hyvideo/utils/qwen_results.py')
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)
CANDIDATES = [dict(option_label=label, chunk_i=i) for i, label in enumerate('ABCDEFG')]


def answer(selected, acceptable=None):
    return dict(motion=dict(supports=list('ABCDEFG'), neutral=[], opposes=[]),
                acceptable=list('ABCDEFG') if acceptable is None else acceptable, selected=selected)


def load_methods(path, class_name, names, **scope):
    """Exercise real runtime methods without importing GPU model dependencies."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    namespace = dict(os=os, json=json, Path=Path, ThreadPoolExecutor=ThreadPoolExecutor,
                     as_completed=as_completed, **scope)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return type(class_name, (), {n.name: namespace[n.name] for n in methods})


class DecisionTests(unittest.TestCase):
    def test_literal_choices_are_never_filled_or_reordered(self):
        for cap in (2, 3):
            for selected in ([], ['C'], ['C', 'A']):
                with self.subTest(cap=cap, selected=selected):
                    result = results.decode_result('select_coverage', json.dumps(answer(selected)), CANDIDATES, cap)
                    self.assertEqual(result['selected_coverage'], selected)

    def test_invalid_selections_are_skipped_whole_without_salvage(self):
        bad = [answer(['Z']), answer(['A', 'A']), answer('AB'),
               answer(['A'], []), answer(['A', 'B', 'C', 'D']), {},
               dict(answer(['A']), motion=dict(supports=['A'], neutral=[], opposes=[]))]
        for value in bad:
            with self.subTest(value=value):
                result = results.decode_result('select_coverage', json.dumps(value), CANDIDATES, 3)
                self.assertTrue(result['skipped'])
                self.assertNotIn('selected_coverage', result)

    def test_garbled_truncated_or_ambiguous_json_never_becomes_a_decision(self):
        for task in ('visual_quality', 'create', 'sparse', 'select_coverage'):
            for raw in ('garbled output', '{"action":"remove"', '[]', 'null',
                        '{} trailing garbage', '{"action":"keep","action":"remove"}'):
                with self.subTest(task=task, raw=raw):
                    self.assertTrue(results.decode_result(task, raw, CANDIDATES, 3)['skipped'])

    def test_invalid_operation_fields_have_no_default_score_or_action(self):
        for task, value in [
            ('visual_quality', {}), ('visual_quality', {'visual_quality_score': True}),
            ('visual_quality', {'visual_quality_score': float('nan')}),
            ('visual_quality', {'visual_quality_score': 12}),
            ('create', {'match': False}), ('create', {'view_overlap_score': 'garbage'}),
            ('create', {'view_overlap_score': 4, 'match': False}),
            ('sparse', {'action': 'remove'}),
            ('sparse', {'action': 'garbage', 'spatial_coverage': 'full', 'unique_useful_information': 'none'}),
        ]:
            with self.subTest(task=task, value=value):
                self.assertTrue(results.decode_result(task, json.dumps(value))['skipped'])

    def test_valid_operation_decisions_are_preserved(self):
        self.assertEqual(results.decode_result('visual_quality', '{"visual_quality_score":8}'),
                         {'visual_quality_score': 8})
        self.assertTrue(results.decode_result('create', '{"match":true,"view_overlap_score":4}')['prototype_match'])
        raw = '{"action":"remove","spatial_coverage":"full","unique_useful_information":"none"}'
        self.assertEqual(results.decode_result('sparse', raw)['action'], 'remove')


class RuntimeTests(unittest.TestCase):
    def test_current_profile_is_the_default_and_unknown_profiles_are_rejected(self):
        path = ROOT / 'scripts/qwen_judge_wan22.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'WAN22_PROMPT_PROFILE', 'HY15_PROMPT_PROFILE', '_memory_prompt_profile'}
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.FunctionDef) and node.name in names) or
                 (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
        scope = dict(os=os)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scope['_memory_prompt_profile'](), scope['WAN22_PROMPT_PROFILE'])
        with patch.dict(os.environ, {'QWEN_MEMORY_PROMPT_PROFILE': 'retired-profile'}, clear=True):
            with self.assertRaises(ValueError):
                scope['_memory_prompt_profile']()

    def test_wan_request_contains_only_current_visual_candidate_metadata(self):
        path = ROOT / 'scripts/qwen_judge_wan22.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'_build_select_content', 'add_chunk_images'}]
        scope = dict(os=os, _uses_wan22_compact_prompts=lambda: True)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        payload = dict(current_chunk=dict(chunk_i=1, frames=['current.png']), target_next_chunk_id=2,
                       coverage_candidates=[dict(chunk_i=0, option_label='A', frames=['candidate.png'],
                                                 unused_metadata='coordinate_secret')])
        content = scope['_build_select_content'](payload, 'Select from the images.', 'coverage_candidates')
        text = '\n'.join(item['text'] for item in content if item['type'] == 'text')
        self.assertIn('A: chunk_id=0', text)
        self.assertNotIn('coordinate_secret', text)
        self.assertEqual([item['url'] for item in content if item['type'] == 'image'],
                         ['current.png', 'candidate.png'])

    def test_wan_runtime_uses_one_attempt_and_preserve_raw_invalid_output(self):
        path = 'scripts/qwen_judge_wan22.py'
        scope = dict(_build_visual_quality_content=lambda p: [], _build_create_content=lambda p: [],
                     _build_sparse_content=lambda p: [], _build_select_content=lambda *a: [],
                     _active_select_coverage_instruction=lambda: '',
                     _decode_memory_result=lambda task, raw, payload: results.decode_result(task, raw, CANDIDATES, 3))
        for class_name in ('SpatialStackMemoryJudge',):
            cls = load_methods(path, class_name, ['judge'], **scope)
            for task in ('visual_quality', 'create', 'sparse', 'select_coverage'):
                obj = cls()
                obj._judge_via_update_worker = Mock(return_value=None)
                obj._generate_from_content = Mock(return_value=('garbled output', 0.1))
                obj._run = Mock(return_value='garbled output')
                with self.subTest(runtime=class_name, task=task):
                    raw, result = obj.judge(task, {}, 128)
                    self.assertEqual(raw, 'garbled output')
                    self.assertTrue(result['skipped'])
                    generator = obj._generate_from_content
                    self.assertEqual(generator.call_count, 1)

    def hy_memory(self, names):
        cls = load_methods('hyvideo/utils/qwen_memory.py', 'QwenMemory', names)
        obj = cls()
        obj.chunk_geometry_bank = {i: dict(cluster_id=0, visual_quality_score=8, removed=False)
                                   for i in range(3)}
        obj.members = [0, 1]
        obj.clusters = [dict(id=0, members=[0, 1], prototype=0, status='tentative')]
        obj.min_vq = 6
        obj.payload = lambda i, **kw: dict(chunk_i=i, cluster_id=0, visual_quality_score=8)
        obj.save_trace = Mock()
        return obj

    def test_hy_invalid_quality_does_not_admit_or_drop_memory(self):
        obj = self.hy_memory(['update'])
        obj.request = Mock(return_value=results.decode_result('visual_quality', 'garbage'))
        before = copy.deepcopy((obj.members, obj.clusters, obj.chunk_geometry_bank))
        obj.update(2)
        self.assertEqual((obj.members, obj.clusters, obj.chunk_geometry_bank), before)
        self.assertEqual(obj.request.call_count, 1)

    def test_hy_invalid_match_does_not_create_class(self):
        obj = self.hy_memory(['update'])
        obj.candidate_classes = lambda i: [0]
        obj.refresh_prototype = lambda cluster: cluster.update(status='dynamic')
        obj._compute_chunk_view_overlap = lambda *a: 0.5
        obj.execution_device = None
        obj.request = lambda task, *a: ({'visual_quality_score': 8} if task == 'visual_quality'
                                       else results.decode_result('create', '{}'))
        before = copy.deepcopy((obj.members, obj.clusters))
        obj.update(2)
        self.assertEqual((obj.members, obj.clusters), before)
        self.assertTrue(obj.save_trace.call_args.args[2]['skipped'])

    def test_hy_empty_or_invalid_selection_uses_only_existing_local_context(self):
        obj = self.hy_memory(['select'])
        obj.score_chunks = lambda *a, **kw: [dict(chunk_i=0, overlap=0.5)]
        obj.min_vq = 6
        obj._summarize_qwen_motion = lambda *a: {}
        obj._format_qwen_target_motion_text = lambda *a: ''
        view = Mock()
        for raw in ('garbage', json.dumps(dict(motion=dict(supports=['A'], neutral=[], opposes=[]), acceptable=['A'], selected=[]))):
            obj.request = lambda *a: results.decode_result('select_coverage', raw, [CANDIDATES[0]], 2)
            self.assertEqual(obj.select(3, view, None), list(range(4, 12)))
            self.assertEqual(obj.save_trace.call_args.args[2]['memory_chunk_ids'], [])

    def test_hy_invalid_sparse_decision_does_not_delete_memory(self):
        obj = self.hy_memory(['sparse'])
        obj.members = list(range(7))
        poses = Mock()
        poses.numpy.return_value = [None] * 4
        obj.chunk_geometry_bank = {i: dict(removed=False, cluster_id=0, worldplay_viewmats=poses)
                                   for i in obj.members}
        cluster = dict(id=0, members=obj.members[:], prototype=0, status='confirmed')
        obj.execution_device = None
        obj._compute_members_union_query_view_overlap = lambda *a: {'coverage': 1.0}
        obj.request = lambda *a: results.decode_result('sparse', 'garbled output')
        before = copy.deepcopy(cluster)
        obj.sparse(cluster)
        self.assertEqual(cluster, before)
        self.assertEqual(obj.members, list(range(7)))
        self.assertTrue(all(not record['removed'] for record in obj.chunk_geometry_bank.values()))

    def wan_memory(self, names):
        cls = load_methods('wan/inference/pipeline_wan_w_mem_relative_rope.py',
                           'WanPipeline', names)
        return cls()

    def test_wan_never_fills_empty_or_short_choices_from_fov(self):
        obj = self.wan_memory(['_qwen_complete_memory_chunk_ids_by_fov'])
        obj.qwen_memory_allow_fallback = True  # Even a stale object cannot enable replacement.
        obj.qwen_memory_fov_fill_min_chunks = 3
        obj.qwen_memory_select_top_k = 3
        for selected in ([], [2]):
            self.assertEqual(obj._qwen_complete_memory_chunk_ids_by_fov(selected, [dict(chunk_i=0)]),
                             (selected, [], selected))

    def test_wan_visual_candidates_need_no_coordinate_points_or_poses(self):
        obj = self.wan_memory(['_candidate_payloads_from_scores'])
        obj.qwen_memory_select_filter_visual_quality = False
        obj._build_qwen_chunk_payload = lambda i, **kw: dict(chunk_i=i, frames=[f'{i}.png'])
        scores = [dict(chunk_i=2, overlap=0.9), dict(chunk_i=0, overlap=0.8), dict(chunk_i=2, overlap=0.7)]
        candidates = obj._candidate_payloads_from_scores(scores, 'coverage_candidate', 7)
        self.assertEqual([item['chunk_i'] for item in candidates], [2, 0])
        self.assertEqual([item['option_label'] for item in candidates], ['A', 'B'])
        for item in candidates:
            self.assertEqual(set(item), {'chunk_i', 'frames', 'fov_overlap', 'selection_role_hint', 'option_label'})

    def test_wan_invalid_match_does_not_create_class(self):
        obj = self.wan_memory(['_update_qwen_memory_bank_for_chunk', '_qwen_skip_memory_update'])
        obj._qwen_memory_records = {2: dict(visual_quality_score=8)}
        obj._build_qwen_chunk_payload = lambda i, **kw: dict(chunk_i=i, frames=['frame'], visual_quality_score=8)
        obj._qwen_memory_update_min_visual_quality = lambda: 6
        obj._qwen_build_candidate_class_payloads = lambda i: [dict(cluster_id=0, prototype=dict(chunk_i=0))]
        obj._qwen_apply_chunk_cluster_metadata = lambda payload, cid: payload
        obj._qwen_memory_request = lambda *a, **kw: dict(result=results.decode_result('create', 'garbage'))
        obj._qwen_payload_image_ref = lambda payload: payload
        obj._save_qwen_update_trace = Mock(return_value=None)
        obj._qwen_log = Mock()
        obj._qwen_create_memory_class = Mock(side_effect=AssertionError('unexpected class creation'))
        result = obj._update_qwen_memory_bank_for_chunk(2)
        self.assertTrue(result['skipped'])
        self.assertIsNone(result['decision'])
        obj._qwen_create_memory_class.assert_not_called()

    def test_wan_invalid_sparse_decision_does_not_delete_memory(self):
        obj = self.wan_memory(['_qwen_sparsify_memory_class_if_needed', '_qwen_sparse_candidate_positions'])
        obj._qwen_memory_classes = {0: dict(member_chunk_ids=list(range(7)), prototype_chunk_i=0)}
        obj._qwen_active_memory_chunk_ids = set(range(7))
        obj._qwen_memory_records = {i: dict(active_memory=True) for i in range(7)}
        obj._qwen_chunk_fov_overlap = lambda *a: 1.0
        obj._build_qwen_chunk_payload = lambda i, **kw: dict(chunk_i=i, frames=['frame'])
        obj._qwen_apply_chunk_cluster_metadata = lambda payload, cid: payload
        obj._qwen_memory_request = lambda *a, **kw: dict(result=results.decode_result('sparse', 'garbled output'))
        result = obj._qwen_sparsify_memory_class_if_needed(0)
        self.assertEqual(result['removed_chunk_ids'], [])
        self.assertEqual(obj._qwen_active_memory_chunk_ids, set(range(7)))

    def test_wan_empty_or_invalid_choice_reaches_frames_without_fov_replacement(self):
        path = 'wan/inference/pipeline_wan_w_mem_relative_rope.py'
        cls = load_methods(path, 'WanPipeline',
                           ['_select_qwen_memory_frame_indices', '_qwen_select_memory_chunks',
                            '_qwen_complete_memory_chunk_ids_by_fov'],
                           torch=SimpleNamespace(distributed=SimpleNamespace(is_available=lambda: False)))
        obj = cls()
        obj.qwen_memory_judge_enable = True
        obj.qwen_memory_select_policy = 'coverage_only'
        obj.qwen_memory_select_top_k = 3
        obj._qwen_update_candidate_scores_by_chunk = {}
        obj._qwen_active_memory_chunk_ids = {0, 1}
        obj._is_rank0 = lambda: True
        obj._qwen_temporal_local_frame_indices = lambda **kw: list(range(8, 20))
        obj._qwen_finalize_remote_eligible_memory_chunks = lambda **kw: []
        obj._qwen_fov_candidate_scores = lambda **kw: [dict(chunk_i=0, overlap=1.0)]
        obj._format_qwen_target_motion_text = lambda *a: ''
        obj._build_qwen_chunk_payload = lambda i, **kw: dict(chunk_i=i, frames=['frame'])
        obj._candidate_payloads_from_scores = lambda *a, **kw: CANDIDATES
        obj._qwen_payload_image_ref = lambda payload: payload
        obj._qwen_cap_selected_frame_indices = lambda selected, local: (selected, None)
        obj._save_qwen_select_trace = Mock(return_value=None)
        obj._save_qwen_select_debug_sheet = Mock(return_value=None)
        obj._qwen_log = Mock()
        obj._qwen_compute_wan_fallback_frame_indices = Mock(side_effect=AssertionError('unexpected FoV replacement'))
        for raw in ('garbled output', json.dumps(answer([]))):
            obj._qwen_memory_request = lambda *a, **kw: dict(result=results.decode_result('select_coverage', raw, CANDIDATES, 3))
            selected = obj._select_qwen_memory_frame_indices(
                chunk_i=5, viewmats=MagicMock(), action=MagicMock(), current_frame_idx=20,
                context_window_length=16, chunk_size=4, n_latent=40,
                fallback_selected_frame_indices=[0, 1, 2, 3], device=None)
            self.assertEqual(selected, list(range(8, 20)))
            trace = obj._save_qwen_select_trace.call_args.args[1]
            self.assertEqual(trace['selected_memory_chunk_ids'], [])
            self.assertFalse(trace['fallback_used'])
        obj._qwen_compute_wan_fallback_frame_indices.assert_not_called()


if __name__ == '__main__':
    unittest.main()
