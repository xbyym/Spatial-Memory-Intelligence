"""CPU contracts for Base keyframes, rank gating, and unchanged SMI routing."""
import ast
from contextlib import nullcontext
import copy
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class Array:
    """Small CPU tensor adapter; VAE weights and CUDA are not used by these tests."""
    def __init__(self, data): self.data = np.asarray(data)
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Array(self.data.copy())
    def permute(self, *dims): return Array(self.data.transpose(dims))
    def clamp(self, low, high): return Array(np.clip(self.data, low, high))
    def to(self, dtype): return Array(self.data.astype(dtype))
    def float(self): return self.to(np.float32)
    def contiguous(self): return Array(np.ascontiguousarray(self.data))
    def numpy(self): return self.data
    def __len__(self): return len(self.data)
    def __getitem__(self, key): return Array(self.data[key])
    def __mul__(self, value): return Array(self.data * value)
    def __truediv__(self, value): return Array(self.data / value)
    def __add__(self, value): return Array(self.data + value)


def write_png(path, data):
    Image.fromarray(data).save(path)


def module_tree(path):
    return ast.parse((ROOT/path).read_text(encoding='utf-8'))


def pipeline_method():
    return next(n for n in ast.walk(module_tree('hyvideo/pipelines/worldplay_video_pipeline.py'))
                if isinstance(n, ast.FunctionDef) and n.name == 'ar_rollout')


def writer_class():
    tree = module_tree('hyvideo/utils/chunk_frames.py')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = dict(Path=Path, torch=SimpleNamespace(uint8=np.uint8),
                 imageio=SimpleNamespace(imwrite=write_png))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'chunk_frames', 'exec'), scope)
    return scope['ChunkFrameWriter']


def choose_recorder(qwen_factory, base_factory, rank):
    body = copy.deepcopy(pipeline_method().body)
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'positive_idx' for t in node.targets))
    body = body[:end] + [ast.Return(ast.Tuple(elts=[ast.Name('memory', ast.Load()),
                                                 ast.Name('chunk_recorder', ast.Load())], ctx=ast.Load()))]
    function = ast.parse('def choose(self, device):\n    pass').body[0]
    function.body = body
    scope = dict(os=os, get_rank=lambda: rank, QwenMemory=qwen_factory, ChunkFrameWriter=base_factory)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 'pipeline_recorder', 'exec'), scope)
    return scope['choose'](SimpleNamespace(init_kv_cache=lambda: None, points_local=None), 'cpu')


class BaseChunkFrameTests(unittest.TestCase):
    def test_base_pngs_match_existing_smi_writer_and_do_not_change_pixels(self):
        cls = next(n for n in module_tree('hyvideo/utils/qwen_memory.py').body
                   if isinstance(n, ast.ClassDef) and n.name == 'QwenMemory')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'add')
        scope = dict(torch=SimpleNamespace(uint8=np.uint8), imageio=SimpleNamespace(imwrite=write_png))
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'smi_add', 'exec'), scope)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for frames in [13, 16]:
                original = np.linspace(-0.1, 1.1, 3 * frames * 2 * 3, dtype=np.float32).reshape(3, frames, 2, 3)
                video = Array(original.copy())
                base = writer_class()(root/f'base-{frames}')
                smi_dir = root/f'smi-{frames}'
                smi_dir.mkdir()
                smi = SimpleNamespace(output_dir=smi_dir, chunk_geometry_bank={}, update=Mock())
                base.add(2, video)
                scope['add'](smi, 2, video, Array(np.zeros((4, 4, 4))), None)
                self.assertEqual(sorted(p.name for p in base.output_dir.iterdir()),
                                 ['chunk_002_last.png', 'chunk_002_mid.png'])
                for name in ['mid', 'last']:
                    filename = f'chunk_002_{name}.png'
                    self.assertEqual((base.output_dir/filename).read_bytes(), (smi_dir/filename).read_bytes())
                np.testing.assert_array_equal(video.data, original)

    def test_only_base_rank_zero_uses_the_new_writer(self):
        for enabled, rank in [(False, 0), (False, 1), (True, 0), (True, 1)]:
            with self.subTest(qwen=enabled, rank=rank):
                qwen, base = Mock(), Mock()
                env = {'QWEN_MEMORY_OUTPUT_DIR': 'output/memory'}
                if enabled: env['QWEN_MEMORY_JUDGE_URL'] = 'http://127.0.0.1/judge'
                with patch.dict(os.environ, env, clear=True): memory, recorder = choose_recorder(qwen, base, rank)
                if enabled and rank == 0:
                    self.assertIs(memory, qwen.return_value)
                    self.assertIs(recorder, memory)
                elif not enabled and rank == 0:
                    self.assertIsNone(memory)
                    self.assertIs(recorder, base.return_value)
                else:
                    self.assertIsNone(memory)
                    self.assertIsNone(recorder)
                self.assertEqual(qwen.call_count, int(enabled and rank == 0))
                self.assertEqual(base.call_count, int(not enabled and rank == 0))

    def test_base_saves_each_decoded_chunk_after_temporal_context_is_trimmed(self):
        guard = next(n for n in ast.walk(pipeline_method()) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'chunk_recorder is not None')
        with tempfile.TemporaryDirectory() as temp:
            recorder = writer_class()(temp)
            original = np.arange(32, dtype=np.float32).reshape(1, 1, 8, 2, 2)
            for chunk in [0, 1]:
                decoded = []
                def decode(latents, **kwargs):
                    decoded.append(latents.data.copy())
                    frames = (latents.data.shape[2] - 1) * 4 + 1
                    values = np.linspace(-1, 1, frames, dtype=np.float32)[None, None, :, None, None]
                    return (Array(np.broadcast_to(values, (1, 3, frames, 2, 2)).copy()),)
                scope = dict(chunk_recorder=recorder, chunk_i=chunk, start_idx=chunk*4, end_idx=(chunk+1)*4,
                             latents=Array(original), generator=None, viewmats=Array(np.zeros((1, 8, 4, 4))),
                             action=Array(np.zeros((1, 8))), torch=SimpleNamespace(autocast=lambda *a, **k: nullcontext()),
                             auto_offload_model=lambda *a, **k: nullcontext(),
                             self=SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=2, shift_factor=0.5), decode=decode),
                                                  vae_dtype='float32', vae_autocast_enabled=False,
                                                  execution_device='cpu', enable_offloading=False, chunk_latent_frames=4))
                exec(compile(ast.Module(body=[guard], type_ignores=[]), 'decode_chunk', 'exec'), scope)
                begin = max(0, chunk*4-2)
                np.testing.assert_array_equal(decoded[0], original[:, :, begin:(chunk+1)*4]/2+0.5)
                mid = np.array(Image.open(Path(temp)/f'chunk_{chunk:03d}_mid.png'))
                last = np.array(Image.open(Path(temp)/f'chunk_{chunk:03d}_last.png'))
                self.assertTrue((mid == (127 if chunk == 0 else 165)).all())
                self.assertTrue((last == 255).all())
                np.testing.assert_array_equal(scope['latents'].data, original)


if __name__ == '__main__':
    unittest.main()
