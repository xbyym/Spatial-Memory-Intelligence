"""Regression checks for the geometry and deterministic inference controls."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieval = load_file("retrieval", ROOT / "hyvideo/utils/retrieval_context.py")


class ReproducibilityTests(unittest.TestCase):
    def test_flash_attention_receives_deterministic_backward_flag(self):
        from transformers.modeling_flash_attention_utils import _process_flash_attention_kwargs
        reproducibility = load_file("reproducibility", ROOT / "hyvideo/utils/reproducibility.py")
        with patch.dict(os.environ, {"SMI_SEED": "123", "FLASH_ATTENTION_DETERMINISTIC": "0"}):
            reproducibility.configure()
            kwargs = _process_flash_attention_kwargs(4, 4, True, supports_mapping={
                "dropout_p": True, "window_size": True, "deterministic": True,
                "softcap": True, "s_aux": False})
            self.assertIs(kwargs["deterministic"], True)

    def test_geometry_is_independent_of_model_rng_and_does_not_consume_it(self):
        with patch.dict(os.environ, {"SMI_GEOMETRY_SEED": "123"}, clear=True):
            torch.manual_seed(19)
            before = torch.get_rng_state().clone()
            first = retrieval.generate_points_in_sphere(50000, 8.0)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            torch.rand(12345)
            second = retrieval.generate_points_in_sphere(50000, 8.0)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue((first.norm(dim=1) <= 8.00001).all())
            self.assertFalse(torch.equal(first, retrieval.generate_points_in_sphere(50000, 8.0, seed=124)))

    def test_saved_fixture_replays_exactly_and_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SMI_REPRO_DIR": directory}, clear=True):
                points = retrieval.generate_points_in_sphere(50, 8.0)
            fixture = str(Path(directory) / "geometry-50-8.npy")
            with patch.dict(os.environ, {"SMI_GEOMETRY_POINTS": fixture}, clear=True):
                self.assertTrue(torch.equal(points, retrieval.generate_points_in_sphere(50, 8.0, seed=999)))
                with self.assertRaises(ValueError):
                    retrieval.generate_points_in_sphere(51, 8.0)



if __name__ == "__main__":
    unittest.main()
