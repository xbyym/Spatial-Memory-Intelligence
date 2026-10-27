"""Compare two completed repetitions using decoded pixels and Qwen tokens."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def qwen_records(root):
    calls = Counter()
    outputs = defaultdict(set)
    for path in (root / "reproducibility").glob("qwen-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["input_sha256"]
            result = hashlib.sha256(json.dumps(row["output_tokens"]).encode()).hexdigest()
            calls[(key, result)] += 1
            outputs[key].add(result)
    return calls, {key: list(value) for key, value in outputs.items() if len(value) > 1}


def pixels(path):
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-pix_fmt", "rgb24", "-f", "framemd5", "-"], capture_output=True, text=True, check=True)
    return [line.split(",")[-1].strip() for line in result.stdout.splitlines() if not line.startswith("#")]


def compare(left, right):
    receipts = []
    plans = []
    for root in (left, right):
        if not (root / "finished.json").is_file():
            raise ValueError(f"Run is not complete: {root}")
        receipt = json.loads((root / "finished.json").read_text())
        if receipt.get("completed") is not True or not receipt.get("media"):
            raise ValueError(f"Missing successful completion/media evidence: {root}")
        plan = json.loads((root / "launch_plan.json").read_text())
        if not receipt.get("source_sha256") or receipt["source_sha256"] != plan.get("source_sha256"):
            raise ValueError(f"Completion source does not match the saved launch plan: {root}")
        receipts.append(receipt)
        plans.append(plan)
    for key in ("case", "method", "seed"):
        if key not in plans[0] or plans[0][key] != plans[1].get(key):
            raise ValueError(f"Cannot compare different or unspecified {key} values")
    geometry = []
    left_geometry = {p.name for p in (left / "reproducibility").glob("geometry-*.npy")}
    right_geometry = {p.name for p in (right / "reproducibility").glob("geometry-*.npy")}
    if not left_geometry or left_geometry != right_geometry:
        raise ValueError("Missing or different saved geometry fixture sets")
    for name in sorted(left_geometry):
        path = left / "reproducibility" / name
        peer = right / "reproducibility" / path.name
        geometry.append({"file": path.name, "equal": peer.exists() and path.read_bytes() == peer.read_bytes()})
    calls_a, conflicts_a = qwen_records(left)
    calls_b, conflicts_b = qwen_records(right)
    if plans[0]["method"].lower() == "smi" and (not calls_a or not calls_b):
        raise ValueError("Missing actual Qwen input/output audit records for an SMI run")
    media_a, media_b = [receipt["media"] for receipt in receipts]
    paths_a = sorted(row["path"] for row in media_a)
    paths_b = sorted(row["path"] for row in media_b)
    if paths_a != paths_b:
        raise ValueError("Runs have different output video sets")
    videos = []
    for relative in paths_a:
        a, b = pixels(left / relative), pixels(right / relative)
        if not a or not b:
            raise ValueError(f"No decoded frames in {relative}")
        different = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        videos.append({"path": relative, "left_frames": len(a), "right_frames": len(b),
                       "differing_frames": len(different) + abs(len(a) - len(b)),
                       "first_different_frame": different[0] if different else (min(len(a), len(b)) if len(a) != len(b) else None)})
    equal = (all(row["equal"] for row in geometry) and calls_a == calls_b
             and not conflicts_a and not conflicts_b and all(row["differing_frames"] == 0 for row in videos))
    return {"equal": equal, "case": plans[0]["case"], "method": plans[0]["method"],
            "source_equal": receipts[0]["source_sha256"] == receipts[1]["source_sha256"],
            "physical_gpu_groups_equal": plans[0].get("required_physical_gpu_indices") == plans[1].get("required_physical_gpu_indices"),
            "geometry": geometry, "videos": videos,
            "qwen_calls_left": sum(calls_a.values()), "qwen_calls_right": sum(calls_b.values()),
            "qwen_input_output_multiset_equal": calls_a == calls_b,
            "within_run_qwen_conflicts_left": conflicts_a,
            "within_run_qwen_conflicts_right": conflicts_b,
            "scope": "These two runs only; not a guarantee across hardware/software versions or historical videos."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
