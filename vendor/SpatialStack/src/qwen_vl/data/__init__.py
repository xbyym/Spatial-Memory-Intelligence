"""Portable registrations for the two released SMI datasets."""
import os
from pathlib import Path


def data_list(dataset_names):
    root = Path(os.environ["SMI_DATA_ROOT"]).expanduser().resolve()
    configs = []
    for name in dataset_names:
        if name not in {"smi_hy15_train", "smi_hy15_eval", "smi_wan22_train", "smi_wan22_eval"}:
            raise ValueError(f"Unknown released dataset: {name}")
        _, backbone, split = name.split("_")
        annotation = root / "annotations" / backbone / f"{split}.json"
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        configs.append({"annotation_path": str(annotation), "data_path": str(root),
                        "sampling_rate": 1.0, "dataset_name": name, "tag": "2d"})
    return configs
