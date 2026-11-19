from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.retail.env import RetailEnv
from src.retail.rollout import demo_data,demo_config
from src.retail.tabular import CooperativeQLearning,RetailDiscretizer

class TabularTests(unittest.TestCase):
    def test_state_ids_in_range_and_remaining_time_encoded(self):
        disc=RetailDiscretizer()
        obs=RetailEnv(demo_data(),demo_config(),29,90).reset()[0]
        a=disc(obs)
        self.assertTrue(0<=a<disc.n_states)
        obs=dict(obs,days_remaining=1)
        self.assertNotEqual(a,disc(obs))

    def test_terminal_target_does_not_bootstrap(self):
        env=RetailEnv(demo_data(),demo_config(),29,29)
        obs=env.reset()
        policy=CooperativeQLearning(demo_config(),seed=3,alpha=1.0)
        for agent in policy.agents:
            agent.Q[:]=100.0
        actions=policy.act(obs)
        nxt,reward,terminal,_,_=env.step(actions)
        policy.observe(obs,actions,reward,nxt,terminal)
        for i,agent in enumerate(policy.agents):
            self.assertEqual(agent.Q[policy.discretizer(obs[i]),actions[i]],reward)

    def test_greedy_evaluation_freezes_tables_and_checkpoint_roundtrip(self):
        policy=CooperativeQLearning(demo_config(),seed=3)
        before=[a.Q.copy() for a in policy.agents]
        obs=RetailEnv(demo_data(),demo_config(),29,29).reset()
        self.assertEqual(policy.act(obs),policy.act(obs))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"tables.npz"
            policy.save(path)
            with np.load(path,allow_pickle=False) as saved:
                for i,agent in enumerate(policy.agents):
                    np.testing.assert_array_equal(before[i],agent.Q)
                    np.testing.assert_array_equal(saved[f"store_{i}"],agent.Q)

    def test_training_entrypoint_saves_artifacts_without_test_evaluation(self):
        import json
        from src.retail.train_tabular import main
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"run"
            main(["--demo","--seed","3","--episodes","2","--horizon","2","--output",str(output)])
            report=json.loads((output/"manifest.json").read_text())
            self.assertEqual(report["real_steps"],4)
            self.assertFalse(report["test_evaluated"])
            self.assertTrue((output/"q_tables.npz").is_file())
            self.assertEqual(len((output/"training.jsonl").read_text().splitlines()),2)

    def test_frozen_table_evaluation_matches_training_report(self):
        import json
        from src.retail.train_tabular import main as train
        from src.retail.evaluate_tabular import main as evaluate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train(["--demo","--seed","3","--episodes","1","--horizon","2","--output",str(root/"run")])
            evaluate(["--run",str(root/"run"),"--split","validation","--output",str(root/"eval")])
            trained=json.loads((root/"run/manifest.json").read_text())
            evaluated=json.loads((root/"eval/evaluation.json").read_text())
            self.assertEqual(trained["validation"],evaluated["metrics"])
            self.assertEqual(evaluated["learning_updates"],0)
