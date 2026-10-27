"""Download SMI models, training data, and video generator prerequisites."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")


def release_assets(root, component, extract):
    from huggingface_hub import hf_hub_download
    manifest = json.loads((ROOT / "manifests/RELEASE_ASSETS.json").read_text())
    selected = [row for row in manifest["files"] if row["component"] == component]
    if not selected:
        raise ValueError(f"No assets listed for {component}")
    for row in selected:
        destination = root / row["path"]
        if not destination.is_file() or destination.stat().st_size != row["bytes"]:
            hf_hub_download(manifest["repo_id"], row["path"], repo_type=manifest["repo_type"],
                            revision=manifest["revision"], local_dir=root)
        check(destination, row["sha256"])
        print(f"Verified {row['path']}", flush=True)
    if extract and component == "data":
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_dataset.py"),
                        "--root", str(root / "datasets"), "--extract"], check=True)


def generators(root):
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as ms_download
    destination = root / "prerequisites"
    hy = destination / "HunyuanVideo-1.5"
    sources = [
        ("tencent/HunyuanVideo-1.5", "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038", hy,
         ["vae/*", "scheduler/*", "transformer/480p_i2v/*"]),
        ("tencent/HY-WorldPlay", "f4c29235647707b571479a69b569e4166f9f5bf8", destination / "HY-WorldPlay",
         ["ar_distilled_action_model/*", "wan_distilled_model/*", "wan_transformer/*", "License.txt"]),
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635", destination / "Wan2.2-TI2V-5B-Diffusers", None),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", hy / "text_encoder/llm", None),
        ("google/byt5-small", "68377bdc18a2ffec8a0533fef03b1c513a4dd49d", hy / "text_encoder/byt5-small", None),
        ("black-forest-labs/FLUX.1-Redux-dev", "c95859fbf7703ca4d6824b4da4407d7cd0434f81", hy / "vision_encoder/siglip", None),
    ]
    for repo, revision, path, patterns in sources:
        snapshot_download(repo, revision=revision, local_dir=path, allow_patterns=patterns)
    glyph = ms_download("AI-ModelScope/Glyph-SDXL-v2", revision="master",
                        cache_dir=str(destination / "modelscope-cache"))
    target = hy / "text_encoder/Glyph-SDXL-v2"
    shutil.copytree(glyph, target, dirs_exist_ok=True)
    # ModelScope's preserved receipt uses master, so verify the exact used bytes.
    check(target / "checkpoints/byt5_model.pt", "ca8c97c89136f767d4534449bbf3f25296d390574e0af1cc16f09774a901d6db")
    check(target / "checkpoints/byt5_mapper.pt", "6d5911cf07328d949eff29cc08ca7637dc4fe5312a8fa351ca4bec07d357b1c5")


def model_config(root):
    config = json.loads((ROOT / "configs/models.example.json").read_text())
    upstream = root / "prerequisites"
    config.update(hy15_backbone=str(upstream / "HunyuanVideo-1.5"),
                  hy15_action=str(upstream / "HY-WorldPlay/ar_distilled_action_model/model.safetensors"),
                  wan22_backbone=str(upstream / "Wan2.2-TI2V-5B-Diffusers"),
                  wan22_transformer=str(upstream / "HY-WorldPlay/wan_transformer"),
                  wan22_action=str(upstream / "HY-WorldPlay/wan_distilled_model/model.pt"))
    config["qwen_checkpoints"] = {key: str(root / "models" / Path(value).name)
                                  for key, value in config["qwen_checkpoints"].items()}
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["controllers", "data", "generators", "training-init", "all"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--models-config", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    components = ["controllers", "data", "generators", "training-init"] if args.component == "all" else [args.component]
    for component in components:
        if component in {"controllers", "data", "training-init"}:
            release_assets(root, component, args.extract)
        elif component == "generators":
            generators(root)
    if args.models_config:
        args.models_config.parent.mkdir(parents=True, exist_ok=True)
        args.models_config.write_text(json.dumps(model_config(root), indent=2) + "\n")


if __name__ == "__main__":
    main()
