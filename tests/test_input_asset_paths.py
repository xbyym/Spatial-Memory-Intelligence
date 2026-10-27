"""Explicit input assets must never be replaced by a guessed image or pose."""
import ast
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def asset_resolver():
    source = ROOT/'scripts/generate_wan22.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'resolve_sample_path')
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['resolve_sample_path']


class InputAssetPathTests(unittest.TestCase):
    def test_relative_path_uses_manifest_directory_even_with_cwd_decoy(self):
        resolve = asset_resolver()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root/'inputs'
            inputs.mkdir()
            expected = inputs/'first_frame.png'
            expected.write_bytes(b'input image')
            (root/'first_frame.png').write_bytes(b'unrelated cwd image')
            previous = Path.cwd()
            try:
                os.chdir(root)
                result = resolve({'image_path': 'first_frame.png'}, 'image_path', inputs/'input.jsonl')
            finally:
                os.chdir(previous)
            self.assertEqual(result, str(expected.resolve()))

    def test_missing_explicit_image_is_not_replaced_by_name(self):
        resolve = asset_resolver()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            guessed = root/'model_inputs_832x480/images'
            guessed.mkdir(parents=True)
            (guessed/'scene.jpg').write_bytes(b'another image')
            with self.assertRaises(FileNotFoundError):
                resolve({'name': 'scene', 'image_path': 'missing.png'}, 'image_path', root/'input.jsonl')

    def test_absolute_image_and_pose_paths_are_preserved(self):
        resolve = asset_resolver()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for field, name in [('image_path', 'first_frame.png'), ('pose_json_path', 'poses.json'),
                                ('trajectory_npz_path', 'trajectory.npz')]:
                target = root/name
                target.write_bytes(b'provided asset')
                self.assertEqual(resolve({field: str(target)}, field, root/'inputs/input.jsonl'),
                                 str(target.resolve()))


if __name__ == '__main__':
    unittest.main()
