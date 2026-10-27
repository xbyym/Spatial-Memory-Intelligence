"""Check generic input handling and matched Base/Qwen launch configuration."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SingleInputInferenceTests(unittest.TestCase):
    def launch_plan(self, config, method, output, *extra, backbone='wan22'):
        command = [sys.executable, str(ROOT/'scripts/run_inference.py'),
                   '--backbone', backbone, '--input', str(config), '--method', method,
                   '--models', str(ROOT/'configs/models.example.json'), '--output', str(output), *extra]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((output/'running.json').exists())
        return json.loads((output/'launch_plan.json').read_text())

    def test_base_and_qwen_share_each_sample_input(self):
        with tempfile.TemporaryDirectory() as temp:
            for backbone, count in [('hy15', 8), ('wan22', 3)]:
                configs = sorted((ROOT/'examples'/backbone).glob('*/input.json'))
                self.assertEqual(len(configs), count)
                for config in configs:
                    sample = json.loads(config.read_text(encoding='utf-8'))
                    image = config.parent/sample['image_path']
                    expected_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                    for gpu_count in [4, 8]:
                        gpus = ','.join(map(str, range(gpu_count)))
                        base, qwen = [self.launch_plan(
                            config, method, Path(temp)/backbone/config.parent.name/str(gpu_count)/method,
                            '--gpus', gpus, backbone=backbone) for method in ['base', 'qwen']]
                        for key in ['seed', 'navigation', 'input_files', 'expected_raw_frames', 'geometry']:
                            self.assertEqual(base[key], qwen[key])
                        self.assertIsNone(base['judge_command'])
                        self.assertEqual(qwen['checkpoint_key'], backbone)
                        self.assertIsNotNone(qwen['judge_command'])
                        for p, mode in [(base, 'baseline'), (qwen, 'qwen')]:
                            cmd = p['generate_command']
                            self.assertEqual(p['input_files']['image_path']['sha256'], expected_hash)
                            self.assertEqual(p['seed'], sample['seed'])
                            self.assertEqual(p['generate_env']['CUDA_VISIBLE_DEVICES'], gpus)
                            self.assertIn(f'--nproc_per_node={gpu_count}', cmd)
                            if backbone == 'hy15':
                                for flag, field in [('image_path', 'image_path'), ('prompt', 'prompt'),
                                                    ('pose', 'pose'), ('video_length', 'num_frames'),
                                                    ('seed', 'seed')]:
                                    value = str(image) if field == 'image_path' else str(sample[field])
                                    self.assertEqual(cmd[cmd.index('--'+flag)+1], value)
                                self.assertEqual(p['expected_raw_frames'], sample['num_frames'])
                            else:
                                self.assertEqual(cmd[cmd.index('--mode')+1], mode)
                                self.assertEqual(p['navigation']['image_path'], str(image))
                                for field in ['prompt', 'navigation', 'chunk_length']:
                                    self.assertEqual(p['navigation'][field], sample[field])

    def test_arbitrary_input_outside_the_prepared_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'custom.png'
            image.write_bytes((ROOT/'examples/wan22/case1/first_frame.png').read_bytes())
            config = root/'my-camera-path.json'
            config.write_text(json.dumps({'name': 'my-scene', 'image_path': 'custom.png',
                'prompt': {'0': 'A scene'}, 'navigation': {'0': {'move': [0, 1], 'yaw': 0, 'pitch': 0}},
                'chunk_length': {'0': 1}, 'seed': 123}))
            plan = self.launch_plan(config, 'base', root/'output', '--gpus', '4,5,6,7')
            self.assertEqual(plan['case'], 'my-scene')
            self.assertEqual(plan['seed'], 123)
            self.assertEqual(plan['expected_raw_frames'], 13)
            self.assertEqual(plan['required_physical_gpu_indices'], ['4', '5', '6', '7'])
            self.assertEqual(Path(plan['navigation']['image_path']), image)

    def test_hy15_accepts_four_and_eight_gpus(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp)/'first_frame.png'
            image.write_bytes((ROOT/'examples/wan22/case1/first_frame.png').read_bytes())
            config = Path(temp)/'input.json'
            config.write_text(json.dumps(dict(image_path='first_frame.png', prompt='A scene',
                                              pose='w-3', num_frames=13, seed=1)))
            for count in [4, 8]:
                output = Path(temp)/str(count)
                result = subprocess.run([sys.executable, str(ROOT/'scripts/run_inference.py'),
                    '--backbone', 'hy15', '--input', str(config),
                    '--method', 'qwen',
                    '--models', str(ROOT/'configs/models.example.json'),
                    '--gpus', ','.join(map(str, range(count))), '--output', str(output)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads((output/'launch_plan.json').read_text())
                self.assertEqual(plan['inference_gpus'], count)
                self.assertEqual(len(plan['judge_devices']), count)

    def test_wan_eight_gpus_run_one_base_or_qwen_job(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in ['base', 'qwen']:
                output = Path(temp)/method
                plan = self.launch_plan(ROOT/'examples/wan22/case1/input.json', method,
                                        output, '--gpus', '0,1,2,3,4,5,6,7')
                self.assertIn('--nproc_per_node=8', plan['generate_command'])
                self.assertEqual(plan['inference_gpus'], 8)
                self.assertEqual(plan['generate_env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
                self.assertEqual(plan['required_physical_gpu_indices'], [str(i) for i in range(8)])
                if method == 'base':
                    self.assertIsNone(plan['judge_command'])
                else:
                    self.assertEqual(len(plan['judge_devices']), 8)
                    self.assertEqual(plan['judge_env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')


if __name__ == '__main__':
    unittest.main()
