"""Protect expensive completed runs from stale input reuse or evidence overwrite."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/inference_launcher.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunEvidenceTests(unittest.TestCase):
    def test_replaced_image_bytes_cannot_resume_under_the_same_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "first_frame.png"
            image.write_bytes(b"original image")
            plan = {"output": str(root / "run"), "source_sha256": "source",
                    "input_files": runner.input_fingerprints({"image_path": str(image)})}
            runner.save_plan(plan)
            saved = (root / "run/launch_plan.json").read_bytes()
            image.write_bytes(b"modified image")
            changed = dict(plan, input_files=runner.input_fingerprints({"image_path": str(image)}))
            with self.assertRaisesRegex(ValueError, "input bytes"):
                runner.save_plan(changed, execute=True, resume=True)
            self.assertEqual(saved, (root / "run/launch_plan.json").read_bytes())

    def test_planning_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"output": directory, "source_sha256": "original"}
            runner.save_plan(plan)
            path = Path(directory) / "launch_plan.json"
            saved = path.read_bytes()
            runner.save_plan(copy.deepcopy(plan))
            with self.assertRaises(ValueError):
                runner.save_plan(dict(plan, source_sha256="changed"))
            self.assertEqual(saved, path.read_bytes())

    def test_orphaned_run_and_failed_attempt_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"output": directory, "source_sha256": "source"}
            runner.write(root / "failed.json", {"error": "original failure"})
            with self.assertRaisesRegex(ValueError, "no launch plan"):
                runner.save_plan(plan)
            self.assertFalse((root / "launch_plan.json").exists())
            runner.write(root / "launch_plan.json", plan)
            with self.assertRaisesRegex(ValueError, "Failed attempt"):
                runner.save_plan(plan, execute=True, resume=True)
            self.assertEqual(runner.read(root / "failed.json")["error"], "original failure")

    def test_resume_rejects_false_or_incomplete_completion_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"output": directory, "source_sha256": "source"}
            self.assertFalse(runner.completed_for_resume(plan))
            valid = {"completed": True, "media": [{"path": "raw/gen.mp4"}], "source_sha256": "source"}
            for bad in [{}, dict(valid, completed=False), dict(valid, media=[]),
                        dict(valid, source_sha256="other")]:
                runner.write(Path(directory) / "finished.json", bad)
                with self.assertRaises(ValueError):
                    runner.completed_for_resume(plan)
            runner.write(Path(directory) / "finished.json", valid)
            self.assertTrue(runner.completed_for_resume(plan))


if __name__ == "__main__":
    unittest.main()
