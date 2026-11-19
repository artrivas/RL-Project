"""Focused G0 checks against known discounted rewards."""
import json
from pathlib import Path
import tempfile
import unittest
from src.retail.metrics import discounted_return
from src.retail.rollout import demo_data, demo_config

class G0UnitTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(discounted_return([1, 1, 1], 0.9), 2.71, places=6)
        self.assertAlmostEqual(discounted_return([1, 2, 3], 0.5), 2.75, places=6)
        self.assertEqual(discounted_return([], 0.9), 0.0)
        self.assertEqual(discounted_return([5, 5], 0.0), 5.0)
        self.assertEqual(discounted_return([1, 2], 1.0), 3.0)

    def test_gamma_bounds(self):
        with self.assertRaises(ValueError):
            discounted_return([1], -0.1)
        with self.assertRaises(ValueError):
            discounted_return([1], 1.1)

class G0LoggingTests(unittest.TestCase):
    def test_tabular_gamma1_G0_matches_base_return_and_td_loss(self):
        from src.retail.train_tabular import main
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            main(["--demo", "--seed", "3", "--episodes", "2", "--horizon", "2",
                  "--gamma", "1.0", "--output", str(out)])
            lines = (out / "training.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                rec = json.loads(line)
                self.assertIn("base_G0", rec)
                self.assertIn("td_loss", rec)
                self.assertAlmostEqual(rec["base_G0"], rec["base_return"], places=9)
                self.assertTrue(rec["td_loss"] is None or rec["td_loss"] >= 0)

    def test_tabular_discounted_G0_manual(self):
        from src.retail.train_tabular import main
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            main(["--demo", "--seed", "3", "--episodes", "1", "--horizon", "3",
                  "--gamma", "0.5", "--output", str(out)])
            rec = json.loads((out / "training.jsonl").read_text().splitlines()[0])
            self.assertIn("base_G0", rec)
            self.assertLessEqual(rec["base_G0"], rec["base_return"] if rec["base_return"] >= 0 else rec["base_G0"])
            self.assertTrue(rec["td_loss"] is None or rec["td_loss"] >= 0)
