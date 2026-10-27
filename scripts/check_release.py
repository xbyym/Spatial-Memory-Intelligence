"""Validate the current training/inference source and portable sample inputs."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(root, tracked=False):
    required = {
        'scripts/run_inference.py', 'scripts/train_qwen.py', 'scripts/generate_wan22.py',
        'scripts/qwen_judge_wan22.py', 'hyvideo/generate.py', 'hyvideo/qwen_judge.py',
        'vendor/SpatialStack/src/qwen_vl/train/train_qwen.py',
        'vendor/navigation/navigation_to_poses.py',
        'configs/inference.hy15.json', 'configs/inference.wan22.json',
        'configs/models.example.json', 'manifests/RELEASE_ASSETS.json',
        'LICENSE', 'NOTICE', 'licenses/HY-WorldPlay.txt', 'licenses/NOTICE',
        'hyvideo/utils/chunk_frames.py',
    }
    required.update(str(p.relative_to(root)).replace('\\', '/')
                    for p in (root/'hyvideo/utils/qwen_prompts').glob('*.txt'))
    required.update(str(p.relative_to(root)).replace('\\', '/')
                    for p in (root/'scripts/training').glob('*.txt'))
    errors = []
    samples = []
    for backbone in ('hy15', 'wan22'):
        for folder in sorted((root/'examples'/backbone).glob('*/')):
            path = folder/'input.json'
            frames = list(folder.glob('first_frame.*'))
            if not path.is_file():
                errors.append(f'Missing sample input: {path.relative_to(root)}')
                continue
            if len(frames) != 1:
                errors.append(f'Expected one first frame: {folder.relative_to(root)}')
                continue
            sample = json.loads(path.read_text(encoding='utf-8'))
            if sample.get('image_path') != frames[0].name:
                errors.append(f'Sample image must be its own first frame: {path.relative_to(root)}')
            fields = {'image_path', 'prompt', 'pose', 'num_frames', 'seed'} if backbone == 'hy15' else {'image_path', 'prompt', 'seed'}
            if fields - sample.keys():
                errors.append(f'Missing {backbone} input fields: {path.relative_to(root)}')
            samples.append(path)
    for path in samples:
        required.add(path.relative_to(root).as_posix())
        sample = json.loads(path.read_text(encoding='utf-8'))
        for field in ('image_path', 'pose_json_path', 'trajectory_npz_path'):
            if sample.get(field):
                asset = path.parent/sample[field]
                if Path(sample[field]).is_absolute() or not asset.is_file():
                    errors.append(f'Invalid portable {field}: {path}')
                else:
                    required.add(asset.relative_to(root).as_posix())
    for name in required:
        if not (root/name).is_file():
            errors.append(f'Missing source or input: {name}')
    files = set(subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()) if tracked else set()
    if tracked:
        errors.extend(f'Required file not tracked: {name}' for name in sorted(required-files))
    for path in root.rglob('*.py'):
        if any(part in {'.git', '.venv', '__pycache__', 'outputs'} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'Invalid Python source {path.relative_to(root)}: {exc}')
    return dict(passed=not errors, samples=len(samples), required_files=len(required), errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tracked', action='store_true')
    args = parser.parse_args()
    result = check(args.root.resolve(), args.tracked)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
