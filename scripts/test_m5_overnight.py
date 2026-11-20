"""Offline orchestration tests: no network, GPU or actual training."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

spec=importlib.util.spec_from_file_location("night",Path(__file__).with_name("m5_overnight.py"))
night=importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)

class OvernightTests(unittest.TestCase):
    def test_failed_job_does_not_cancel_later_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            jobs=[{"id":"V0","seed":3,"command":["fake"]},
                  {"id":"V1","seed":3,"command":["fake"]}]
            with patch.object(night,"run_command",side_effect=[
                {"exit_code":1,"timed_out":False},{"exit_code":0,"timed_out":False}]), contextlib.redirect_stdout(io.StringIO()):
                state=night.execute_jobs(jobs,output,60)
            self.assertEqual(state["status"],"completed_with_errors")
            self.assertEqual(len(state["jobs"]),2)
            self.assertEqual(state["pending_jobs"],0)

    def test_exhausted_budget_never_starts_a_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(night,"run_command") as run:
                state=night.execute_jobs([{"id":"V0","seed":3,"command":["fake"]}],Path(tmp),0)
            run.assert_not_called()
            self.assertEqual(state["status"],"budget_exhausted")
            self.assertEqual(state["pending_jobs"],1)

    def test_timeout_escalates_and_reaps_process(self):
        process=Mock(pid=123)
        process.wait.side_effect=[subprocess.TimeoutExpired("fake",1),
                                  subprocess.TimeoutExpired("fake",20),-9]
        process.returncode=-9
        fake_os=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(night.subprocess,"Popen",return_value=process),patch.object(night,"os",fake_os),patch.object(night,"signal",SimpleNamespace(SIGTERM=15,SIGKILL=9)):
                result=night.run_command(["fake"],Path(tmp)/"log",1)
        self.assertTrue(result["timed_out"])
        self.assertEqual(fake_os.killpg.call_count,2)
        self.assertEqual(process.wait.call_count,3)

    def test_prepare_only_freezes_complete_development_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/"pilot";pilot.mkdir()
            audit={"manifest":{"item_id":"FOODS_3_090","sha256":{name:"hash" for name in night.FILES}}}
            (pilot/"source-audit.json").write_text(json.dumps(audit))
            config=night.RetailConfig(1050,1768,253,450,758,.0128,1.28,(1.5,)*3)
            (pilot/"assumptions.json").write_text(json.dumps(config.as_dict()))
            output=root/"batch"
            with patch.object(night,"os",SimpleNamespace(name="posix")),patch.object(night,"fingerprint",return_value="hash"),patch.object(night.subprocess,"check_output",return_value="fixture-commit"),patch.object(night,"run_command") as run,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(night.main(["--pilot",str(pilot),"--data-dir",str(root),"--output",str(output)]),0)
            run.assert_not_called()
            plan=json.loads((output/"suite"/"suite.json").read_text())
            self.assertEqual(len(plan["jobs"]),25)
            self.assertEqual(plan["real_steps_per_trained_run"],109500)
            self.assertEqual({j["seed"] for j in plan["jobs"]},{None,3,4,5})
            self.assertFalse(plan["test_included"])
            self.assertEqual(json.loads((output/"assumptions.json").read_text()),config.as_dict() | {"elasticities":[1.5]*3,"price_multipliers":list(config.price_multipliers),"target_days":list(config.target_days)})

if __name__=="__main__":
    unittest.main()
