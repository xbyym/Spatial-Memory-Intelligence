"""Incomplete evidence must never produce a successful repetition claim."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).resolve().parents[1] / "scripts/compare_repro_runs.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = [Path(self.temporary.name) / name for name in ("a", "b")]
        for root in self.runs:
            audit = root / "reproducibility"
            audit.mkdir(parents=True)
            (audit / "geometry-50000-8.npy").write_bytes(b"identical fixture")
            (audit / "qwen-1.jsonl").write_text(json.dumps({
                "input_sha256": "input", "output_tokens": [[1, 2, 3]]}) + "\n")
            (root / "finished.json").write_text(json.dumps({
                "completed": True, "source_sha256": "source", "media": [{"path": "video.mp4"}]}))
            (root / "launch_plan.json").write_text(json.dumps({
                "case": "0", "method": "SMI", "seed": 10001,
                "source_sha256": "source", "required_physical_gpu_indices": ["0", "1", "2", "3"]}))

    def test_complete_matching_evidence(self):
        with patch.object(comparison, "pixels", return_value=["frame0", "frame1"]):
            self.assertTrue(comparison.compare(*self.runs)["equal"])

    def test_absent_qwen_audits_are_not_equality(self):
        for root in self.runs:
            (root / "reproducibility/qwen-1.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "Qwen"):
            comparison.compare(*self.runs)

    def test_extra_geometry_fixture_is_rejected(self):
        (self.runs[1] / "reproducibility/geometry-extra.npy").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "fixture sets"):
            comparison.compare(*self.runs)

    def test_incomplete_receipt_is_rejected(self):
        path = self.runs[0] / "finished.json"
        receipt = json.loads(path.read_text())
        receipt["completed"] = False
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "completion"):
            comparison.compare(*self.runs)

    def test_empty_decoded_video_is_not_equality(self):
        with patch.object(comparison, "pixels", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No decoded frames"):
                comparison.compare(*self.runs)


if __name__ == "__main__":
    unittest.main()
