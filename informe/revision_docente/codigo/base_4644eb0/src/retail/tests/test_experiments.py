from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.retail.features import FeatureEncoder
from src.retail.shaping import potential,shaped_rewards
from src.retail.deep_config import DeepConfig
from src.retail.env import RetailEnv
from src.retail.rollout import demo_config,demo_data
from src.retail.analysis import summary,time_to_threshold,aggregate
from src.retail.suite import main as suite_main

class ProtocolTests(unittest.TestCase):
    def test_encoder_fit_does_not_see_future_prices(self):
        data,config=demo_data(),demo_config()
        changed=replace(data,prices=tuple(s[:60]+(9999.0,)*30 for s in data.prices))
        a,b=FeatureEncoder.fit(data,config,60),FeatureEncoder.fit(changed,config,60)
        self.assertEqual(a.state_dict(),b.state_dict())
        obs=RetailEnv(data,config,29,30).reset()
        self.assertEqual(a.encode(obs).shape,(3,a.size))
        self.assertTrue(np.isfinite(a.encode(obs)).all())

    def test_pbrs_telescopes_with_true_terminal_zero(self):
        env=RetailEnv(demo_data(),demo_config(),29,33)
        obs=env.reset()
        initial=potential(obs)
        discounted_bonus=0
        gamma=.9
        for t in range(5):
            nxt,base,terminal,_,info=env.step((8,8,8))
            rewards,bonus=shaped_rewards(base,info["local_rewards"],obs,nxt,terminal,gamma,"pbrs",2.0)
            self.assertAlmostEqual(rewards[0]-base,bonus)
            discounted_bonus+=gamma**t*bonus
            obs=nxt
        self.assertAlmostEqual(discounted_bonus,-2*initial)

    def test_rollout_cut_preserves_next_potential(self):
        env=RetailEnv(demo_data(),demo_config(),29,35)
        obs=env.reset()
        nxt,base,terminal,_,info=env.step((0,0,0))
        _,bonus=shaped_rewards(base,info["local_rewards"],obs,nxt,False,.9,"pbrs")
        self.assertAlmostEqual(bonus,.9*potential(nxt)-potential(obs))
        self.assertFalse(terminal)

    def test_naive_bonus_changes_objective(self):
        obs=RetailEnv(demo_data(),demo_config(),29,30).reset()
        hoarded=tuple(dict(o,stock=80) for o in obs)
        _,a=shaped_rewards(0,(0,0,0),obs,None,True,.99,"naive")
        _,b=shaped_rewards(0,(0,0,0),hoarded,None,True,.99,"naive")
        self.assertGreater(b,a)

    def test_selfish_and_regional_rewards_remain_distinct(self):
        obs=RetailEnv(demo_data(),demo_config(),29,30).reset()
        rewards,_=shaped_rewards(6,(1,2,3),obs,None,True,.99,reward_mode="local",scale=2)
        self.assertEqual(rewards,[.5,1,1.5])
        with self.assertRaises(ValueError):
            shaped_rewards(6,(1,2,3),obs,None,True,.99,"pbrs",reward_mode="local")

    def test_deep_settings_reject_invalid_combinations(self):
        for fields in ({"algorithm":"dqn"},{"algorithm":"ppo","dyna_steps":1},
                       {"reward_mode":"local","shaping":"pbrs"},{"gamma":float("nan")},
                       {"replay_capacity":1},{"batch_size":0},{"seed":-1}):
            with self.assertRaises(ValueError):
                DeepConfig(**fields)

    def test_model_transition_uses_exact_accounting_without_sales_access(self):
        from types import SimpleNamespace
        data,config=demo_data(),demo_config()
        env=RetailEnv(data,config,29,30)
        env.reset()
        state=env.snapshot()
        # No sales field: any attempt to access reference demand must fail.
        no_sales=SimpleNamespace(days=data.days,calendar=data.calendar,prices=data.prices)
        imagined=RetailEnv.from_snapshot(no_sales,config,state)
        _,_,_,_,info=imagined.step_model((8,8,8),(100,0,3))
        self.assertEqual(info["demand"],(100,0,3))
        self.assertEqual(info["sales"],(20,0,3))
        self.assertEqual(env.snapshot(),state)

    def test_model_and_real_transition_match_for_identical_demand(self):
        data,config=demo_data(),demo_config()
        real=RetailEnv(data,config,29,30)
        real.reset()
        model=RetailEnv.from_snapshot(data,config,real.snapshot())
        real_result=real.step((8,8,8))
        model_result=model.step_model((8,8,8),real_result[4]["demand"])
        self.assertEqual(real_result,model_result)

    def test_threshold_negative_returns_and_censoring(self):
        history=[{"real_steps":10,"validation":{"base_return":-95}},
                 {"real_steps":20,"validation":{"base_return":-59}}]
        self.assertEqual(time_to_threshold(history,-100,-50),20)
        self.assertIsNone(time_to_threshold(history,-100,0))
        with self.assertRaises(ValueError):
            time_to_threshold(history,-50,-100)
        self.assertAlmostEqual(summary([1,3])["std"],math.sqrt(2))
        self.assertIsNone(summary([1])["std"])

    def test_suite_preparation_is_complete_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"suite"
            suite_main(["--demo","--episodes","2","--output",str(root)])
            plan=json.loads((root/"suite.json").read_text())
            self.assertEqual(len(plan["jobs"]),25)
            self.assertFalse(plan["test_included"])
            self.assertFalse((root/"V0_seed0").exists())
            report,curves=aggregate(root)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["missing_runs"]),25)
            self.assertEqual(curves,[])

    def test_analysis_rejects_mixed_environment_assumptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan={"jobs":[{"id":"H0","seed":None},{"id":"V0","seed":0}],"real_steps_per_trained_run":4}
            (root/"suite.json").write_text(json.dumps(plan))
            for name,capacity in (("H0",80),("V0_seed0",90)):
                path=root/name
                path.mkdir()
                manifest={"data":{"source":"synthetic_test_fixture"},"config":{"capacity":capacity},
                          "range":[61,90],"metrics":{"base_return":1,"stockout_rate":0.1,"fill_rate":0.9,"lost_units":1},
                          "seed":0,"real_steps":4}
                (path/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"cannot compare"):
                aggregate(root)
