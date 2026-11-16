"""Recovery planning tests with synthetic metadata; no training or network."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("recovery",Path(__file__).with_name("resume_m5_overnight.py"))
recovery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)

class RecoveryTests(unittest.TestCase):
    def job(self,batch,name):
        directory=batch/"suite"/f"{name}_seed5"
        return {"id":name,"seed":5,"command":["python3","-m","src.retail.train_deep",
                "--episodes","300","--output",str(directory)]}

    def test_skip_finished_resume_partial_and_start_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp)
            done=self.job(batch,"V0");partial=self.job(batch,"S0");fresh=self.job(batch,"V1")
            d=batch/"suite"/"V0_seed5";d.mkdir(parents=True);(d/"manifest.json").write_text("{}")
            d=batch/"suite"/"S0_seed5";d.mkdir();(d/"checkpoint.pt").write_bytes(b"fixture")
            pending,records,completed=recovery.recovery_plan(batch,{"jobs":[done,partial,fresh]},
                lambda _:{"completed_episodes":190,"real_steps":69350,"horizon":365})
            self.assertEqual(completed,["V0_seed5"])
            self.assertEqual(len(pending),2)
            self.assertIn("--resume",pending[0]["command"])
            self.assertNotIn("--resume",pending[1]["command"])
            self.assertEqual([r["action"] for r in records],["resume","start"])
            self.assertEqual(records[0]["completed_episodes"],190)
            self.assertNotIn("--resume",partial["command"])

    def test_existing_directory_without_checkpoint_never_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);job=self.job(batch,"S0")
            (batch/"suite"/"S0_seed5").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"lacks a checkpoint"):
                recovery.recovery_plan(batch,{"jobs":[job]})

    def test_completed_checkpoint_without_manifest_requires_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);job=self.job(batch,"S0");d=batch/"suite"/"S0_seed5"
            d.mkdir(parents=True);(d/"checkpoint.pt").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError,"needs review"):
                recovery.recovery_plan(batch,{"jobs":[job]},lambda _:{
                    "completed_episodes":300,"real_steps":109500,"horizon":365})

    def test_invalid_checkpoint_progress_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);job=self.job(batch,"S0");d=batch/"suite"/"S0_seed5"
            d.mkdir(parents=True);(d/"checkpoint.pt").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError,"inconsistent"):
                recovery.recovery_plan(batch,{"jobs":[job]},lambda _:{
                    "completed_episodes":190,"real_steps":123,"horizon":365})

    def test_wrong_output_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);job=self.job(batch,"S0");job["command"][-1]=str(batch/"elsewhere")
            with self.assertRaisesRegex(ValueError,"path mismatch"):
                recovery.recovery_plan(batch,{"jobs":[job]})

    def test_active_training_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/"batch";proc=root/"proc";p=proc/"99999999"
            p.mkdir(parents=True)
            (p/"cmdline").write_bytes(("\0".join(["python3","-m","src.retail.train_deep",
                "--output",str(batch/"suite"/"S0_seed5")])+"\0").encode())
            with self.assertRaisesRegex(RuntimeError,"still active"):
                recovery.assert_idle(batch,proc)
            (p/"cmdline").write_bytes(b"unrelated\0")
            recovery.assert_idle(batch,proc)

if __name__=="__main__":
    unittest.main()
