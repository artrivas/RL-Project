import csv
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from src.retail.config import RetailConfig
from src.retail.data import RetailData, load_m5
from src.retail.env import RetailEnv, allocate
from src.retail.baselines import HistoricalPriceBaseStock
from src.retail.metrics import summarize, relative_stockout_reduction
from src.retail.rollout import demo_data, demo_config, main

class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = demo_data(), demo_config()

    def test_integer_rationing_and_ties(self):
        self.assertEqual(allocate((5, 5, 5), 5), (2, 2, 1))
        self.assertEqual(allocate((0, 3, 0), 10), (0, 3, 0))
        self.assertEqual(allocate((0, 0, 0), 0), (0, 0, 0))

    def test_allocation_properties(self):
        for a in range(6):
            for b in range(6):
                for c in range(6):
                    for available in range(12):
                        requests = (a, b, c)
                        actual = allocate(requests, available)
                        self.assertEqual(sum(actual), min(available, sum(requests)))
                        self.assertTrue(all(0 <= x <= r for x, r in zip(actual, requests)))

    def test_joint_conservation_capacities_and_one_day_lead_time(self):
        env = RetailEnv(self.data, self.config, 29, 90)
        obs = env.reset()
        policy = HistoricalPriceBaseStock(self.config)
        previous = None
        while obs is not None:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            self.assertEqual(sum(info["stock_before"])+info["regional_before"],
                             sum(info["stock_after"])+info["regional_after"]+
                             sum(info["transit"])+sum(info["sales"]))
            self.assertTrue(all(s+t <= self.config.store_capacity for s, t in
                                zip(info["stock_after"], info["transit"])))
            self.assertGreaterEqual(info["regional_after"], 0)
            self.assertAlmostEqual(reward, sum(info["local_rewards"]))
            self.assertFalse(truncated)
            if previous is not None:
                self.assertEqual(info["arrivals"], previous["dispatched"])
                self.assertEqual(info["regional_before"],
                                 previous["regional_after"]+info["external_supply"])
                self.assertEqual(info["stock_before"],
                                 tuple(s+t for s,t in zip(previous["stock_after"],previous["transit"])))
            previous = info
        self.assertTrue(terminated)
        with self.assertRaises(RuntimeError):
            env.step((0, 0, 0))

    def test_orders_cannot_satisfy_todays_demand(self):
        config = replace(self.config, initial_store_stock=0)
        env = RetailEnv(self.data, config, 29, 30)
        obs = env.reset()
        obs, _, _, _, info = env.step(HistoricalPriceBaseStock(config).act(obs))
        self.assertEqual(info["sales"], (0, 0, 0))
        self.assertGreater(sum(info["dispatched"]), 0)
        self.assertEqual(tuple(o["stock"] for o in obs), info["dispatched"])

    def test_current_and_future_sales_do_not_change_observation(self):
        changed = replace(self.data, sales=tuple(s[:28]+(1000,)*(len(s)-28) for s in self.data.sales))
        a, b = RetailEnv(self.data,self.config,29,30), RetailEnv(changed,self.config,29,30)
        self.assertEqual(a.reset(), b.reset())

    def test_forecast_uses_realized_sales_after_reset(self):
        config = replace(self.config, initial_store_stock=0, history_days=1)
        env = RetailEnv(self.data, config, 29, 30)
        env.reset()
        obs, _, _, _, _ = env.step((0, 0, 0))
        self.assertEqual(tuple(o["forecast"] for o in obs), (0,0,0))

    def test_profit_accounting(self):
        config = replace(self.config, initial_store_stock=2, regional_supply=0)
        env = RetailEnv(self.data, config,29,29)
        env.reset()
        action = config.actions.index((1.0, 0))
        _, reward, terminal, _, info = env.step((action,)*3)
        expected = sum(2*self.data.prices[i][28]-config.shortage_cost*(info["demand"][i]-2) for i in range(3))
        self.assertAlmostEqual(reward, expected)
        self.assertTrue(terminal)

    def test_price_elasticity_direction(self):
        demands = []
        for multiplier in (0.9, 1.0, 1.1):
            env = RetailEnv(self.data, self.config, 29, 29)
            env.reset()
            action = self.config.actions.index((multiplier,0))
            demands.append(sum(env.step((action,)*3)[4]["demand"]))
        self.assertGreaterEqual(demands[0],demands[1])
        self.assertGreaterEqual(demands[1],demands[2])

    def test_invalid_actions_do_not_mutate_state(self):
        env = RetailEnv(self.data,self.config,29,30)
        before = env.reset()
        for actions in ((0,0), (0,0,15), (0,0,True), (-1,0,0)):
            with self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(before,env.observation())

    def test_config_rejects_invalid_assumptions(self):
        for values in ({"regional_supply":-1},{"initial_store_stock":81},
                       {"elasticities":(1,2)},{"holding_cost":float("nan")},
                       {"price_multipliers":(0,1)},{"history_days":0}):
            with self.assertRaises(ValueError):
                replace(self.config,**values)

    def test_zero_demand_metrics_and_reduction(self):
        row = {"demand":(0,0,0),"sales":(0,0,0),"lost":(0,0,0),"base_reward":0}
        result = summarize([row])
        self.assertIsNone(result["stockout_rate"])
        self.assertIsNone(result["fill_rate"])
        self.assertIsNone(relative_stockout_reduction(0,0))
        self.assertAlmostEqual(relative_stockout_reduction(.1,.2),.5)

    def test_rollout_outputs_reproducible_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a","b"):
                main(["--demo","--policy","random","--seed","3","--output",str(root/name)])
            self.assertEqual((root/"a/manifest.json").read_bytes(),(root/"b/manifest.json").read_bytes())
            self.assertEqual((root/"a/transitions.jsonl").read_bytes(),(root/"b/transitions.jsonl").read_bytes())
            report = json.loads((root/"a/manifest.json").read_text())
            self.assertTrue(report["data"]["not_m5_results"])
            with self.assertRaises(FileExistsError):
                main(["--demo","--output",str(root/"a")])

class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fields = ["item_id","store_id"]+[f"d_{d}" for d in range(1,1942)]
        with (self.root/"sales_train_evaluation.csv").open("w",newline="") as f:
            writer=csv.writer(f); writer.writerow(fields)
            for item, train, future in (("A",2,1),("B",1,10000)):
                for store in ("CA_1","CA_2","CA_3"):
                    writer.writerow([item,store]+[train]*1211+[future]*730)
        with (self.root/"calendar.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["d","date","wm_yr_wk","snap_CA","event_name_1","event_name_2"])
            for d in range(1,1942):
                writer.writerow([f"d_{d}",date(2011,1,29)+timedelta(days=d-1),(d-1)//7,0,"",""])
        with (self.root/"sell_prices.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["item_id","store_id","wm_yr_wk","sell_price"])
            for item in ("A","B"):
                for store in ("CA_1","CA_2","CA_3"):
                    for week in range(278):
                        writer.writerow([item,store,week,2.5])

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_ignores_future_and_records_hashes(self):
        data=load_m5(self.root)
        self.assertEqual(data.item_id,"A")
        self.assertEqual(data.days,1941)
        self.assertEqual(data.sales[0][1211],1)
        self.assertEqual(len(data.manifest["sha256"]),3)
        self.assertEqual(data.manifest["selection"],"train_sales_desc_item_id_asc")

    def test_explicit_item_selection(self):
        data=load_m5(self.root,"B")
        self.assertEqual(data.item_id,"B")
        self.assertEqual(data.manifest["selection"],"explicit")

    def test_duplicate_prices_fail(self):
        with (self.root/"sell_prices.csv").open("a") as f:
            f.write("A,CA_1,0,2.5\n")
        with self.assertRaisesRegex(ValueError,"duplicate price"):
            load_m5(self.root)

    def test_missing_price_fails_without_future_backfill(self):
        path=self.root/"sell_prices.csv"
        text=path.read_text().replace("A,CA_1,0,2.5\n","")
        path.write_text(text)
        with self.assertRaisesRegex(ValueError,"missing price"):
            load_m5(self.root)

    def test_duplicate_sales_fail(self):
        path=self.root/"sales_train_evaluation.csv"
        row=path.read_text().splitlines()[1]
        with path.open("a") as f:f.write(row+"\n")
        with self.assertRaisesRegex(ValueError,"duplicate sales"):
            load_m5(self.root)

    def test_nonfinite_price_fails(self):
        path=self.root/"sell_prices.csv"
        path.write_text(path.read_text().replace("A,CA_1,0,2.5","A,CA_1,0,nan"))
        with self.assertRaisesRegex(ValueError,"invalid price"):
            load_m5(self.root)

    def test_invalid_calendar_alignment_fails(self):
        path=self.root/"calendar.csv"
        path.write_text(path.read_text().replace("2011-01-29","2010-01-29"))
        with self.assertRaisesRegex(ValueError,"unexpected date"):
            load_m5(self.root)

    def test_invalid_held_out_sales_fails_validation(self):
        path=self.root/"sales_train_evaluation.csv"
        lines=path.read_text().splitlines()
        row=lines[1].split(",");row[-1]="-1";lines[1]=",".join(row)
        path.write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError,"negative sales"):
            load_m5(self.root)

if __name__ == "__main__":
    unittest.main()
