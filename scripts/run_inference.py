"""Run Base or SMI on one user-supplied input JSON, with CPU-only planning by default."""
import argparse
import json
import os
from pathlib import Path
import signal

from inference_launcher import (
    ROOT, STOP, absolute_inputs, completed_for_resume, execute, load_models,
    plan, read, save_plan, source_digest,
)


def expected_wan_frames(sample):
    if sample.get('requires_explicit_poses'):
        poses = read(sample['pose_json_path'])
        if list(poses) != [str(i) for i in range(len(poses))]:
            raise ValueError('Explicit pose keys must be ordered 0..N-1')
        latents = len(poses)
        if not latents or latents % 4:
            raise ValueError('Explicit pose count must be positive and divisible by four')
    else:
        navigation = sample['navigation']
        durations = sample['chunk_length']
        if not navigation or set(navigation) != set(durations):
            raise ValueError('Navigation and chunk_length must have matching action keys')
        counts = [int(float(durations[key]) * 4) for key in navigation]
        if any(n <= 0 for n in counts):
            raise ValueError('Each action duration must provide at least one latent frame')
        latents = sum(counts)
        latents += (-latents) % 4
    return (latents - 1) * 4 + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['hy15', 'wan22'], required=True)
    parser.add_argument('--input', type=Path, required=True, help='One generation input JSON')
    parser.add_argument('--input-root', type=Path, help='Root for relative asset paths; defaults to the input directory')
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--settings', type=Path, help='Optional backbone runtime settings JSON')
    parser.add_argument('--method', choices=['base', 'qwen', 'Base', 'SMI'], default='qwen')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for this one run')
    parser.add_argument('--port', type=int, default=18870)
    parser.add_argument('--geometry-seed', type=int, default=0)
    parser.add_argument('--geometry-points', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        gpu_group = [int(value.strip()) for value in args.gpus.split(',')]
        if len(gpu_group) not in (4, 8) or len(set(gpu_group)) != len(gpu_group) or min(gpu_group) < 0:
            raise ValueError('Use four or eight unique, nonnegative GPU indices')
        input_path = args.input.expanduser().resolve()
        input_root = (args.input_root or input_path.parent).expanduser().resolve()
        sample = absolute_inputs(read(input_path), input_root)
        sample.setdefault('name', input_path.stem)
        for field in ['image_path', 'pose_json_path', 'trajectory_npz_path']:
            if sample.get(field) and not Path(sample[field]).is_file():
                raise ValueError(f'Missing {field}: {sample[field]}')
        if args.backbone == 'hy15':
            if not all(key in sample for key in ['prompt', 'pose', 'num_frames', 'seed']):
                raise ValueError('HY1.5 inputs require prompt, pose, num_frames and seed')
            frames = int(sample['num_frames'])
        else:
            if not isinstance(sample.get('prompt'), dict):
                raise ValueError('Wan2.2 prompt must be a dictionary of action-indexed text')
            frames = expected_wan_frames(sample)
        if frames <= 0:
            raise ValueError('Expected frame count must be positive')
        settings = args.settings or ROOT/'configs'/f'inference.{args.backbone}.json'
        method = 'Base' if args.method.lower() == 'base' else 'SMI'
        output = args.output.expanduser().resolve()
        p = plan(input_path.parent, method, load_models(args.models), ROOT, ROOT,
                 output, args.port, gpu_group, geometry_seed=args.geometry_seed,
                 geometry_points=args.geometry_points, input_data=sample,
                 backbone={'hy15': 'HY1.5', 'wan22': 'Wan2.2'}[args.backbone],
                 environment=read(settings), expected_frames=frames)
        p['source_sha256'] = source_digest(ROOT)
        save_plan(p, args.execute, args.resume)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'method': method, 'input': str(input_path), 'output': str(output),
                     'gpus': gpu_group, 'expected_raw_frames': frames, 'execute': args.execute}, indent=2))
    if args.execute:
        if os.name != 'posix':
            parser.error('Actual distributed inference requires Linux with NVIDIA GPUs')
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        if not (args.resume and completed_for_resume(p)):
            execute(p)


if __name__ == '__main__':
    main()
