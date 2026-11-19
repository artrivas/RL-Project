"""Setup checks must distinguish absent data, wrong M5 release and valid headers."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.retail.doctor import SCHEMAS, inspect_data, main


class DataPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_inputs(self, last_day=1941):
        for name, columns in SCHEMAS.items():
            header = sorted(columns)
            if name == "sales_train_evaluation.csv":
                header += [f"d_{day}" for day in range(1, last_day + 1)]
            self.write_csv(name, header)

    def write_csv(self, name, header, row=True):
        with (self.root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            if row:
                writer.writerow(["0"] * len(header))

    def invoke(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(["--data-dir", str(self.root), "--skip-dependencies", *arguments])
        return result, json.loads(output.getvalue())

    def test_absent_data_is_explicit_but_optional_for_environment_setup(self):
        result, report = self.invoke()
        self.assertEqual(result, 0)
        self.assertFalse(report["data_headers_ready"])
        self.assertFalse(report["dependencies_checked"])
        self.assertEqual([entry["status"] for entry in report["data"]], ["missing"] * 3)

    def test_required_data_fails_when_missing(self):
        self.assertEqual(self.invoke("--require-data")[0], 1)

    def test_correct_schema_does_not_claim_complete_audit(self):
        self.write_inputs()
        result, report = self.invoke("--require-data", "--report", str(self.root / "report.json"))
        self.assertEqual(result, 0)
        self.assertTrue(report["data_headers_ready"])
        self.assertFalse(report["full_data_audit_completed"])
        self.assertFalse(report["retail_training_implemented"])
        self.assertEqual(json.loads((self.root / "report.json").read_text(encoding="utf-8")), report)

    def test_shorter_sales_history_rejected(self):
        self.write_inputs(last_day=1913)
        self.assertEqual(self.invoke()[0], 1)

    def test_extra_sales_day_rejected(self):
        self.write_inputs(last_day=1942)
        self.assertEqual(self.invoke()[0], 1)

    def test_calendar_missing_join_key_rejected(self):
        self.write_inputs()
        self.write_csv("calendar.csv", sorted(SCHEMAS["calendar.csv"] - {"wm_yr_wk"}))
        self.assertEqual(self.invoke()[0], 1)

    def test_duplicate_columns_rejected(self):
        self.write_inputs()
        self.write_csv("sell_prices.csv", sorted(SCHEMAS["sell_prices.csv"]) + ["sell_price"])
        self.assertEqual(self.invoke()[0], 1)

    def test_header_only_file_rejected(self):
        self.write_inputs()
        self.write_csv("sell_prices.csv", sorted(SCHEMAS["sell_prices.csv"]), row=False)
        self.assertEqual(self.invoke()[0], 1)

    def test_validation_release_not_silently_used(self):
        (self.root / "sales_train_validation.csv").write_text("id\nexample\n", encoding="utf-8")
        sales = inspect_data(self.root)[0]
        self.assertEqual(sales["status"], "missing")
        self.assertIn("1913-day", sales["errors"][0])


if __name__ == "__main__":
    unittest.main()
