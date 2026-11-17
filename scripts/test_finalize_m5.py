import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from finalize_m5 import (NAMES, SEEDS, aggregate_test, audit, hacking_example,
                         learning_diagnostics, rollout, verify_checkpoint)
from src.retail.rollout import demo_config, demo_data
from src.retail.baselines import HistoricalPriceBaseStock

class ClosureTests(unittest.TestCase):
    def test_rejects_incomplete_or_duplicate_matrix_before_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            suite=Path(d)
            (suite/"suite.json").write_text(json.dumps({"jobs":[{"id":"H0","seed":None}],
                                                      "real_steps_per_trained_run":109500}))
            with self.assertRaisesRegex(ValueError,"complete fixed matrix"):
                audit(suite)

    def test_checkpoint_cannot_silently_use_earlier_episode(self):
        manifest={"settings":{"seed":3},"validation":{"base_return":2}}
        payload={"metadata":{"settings":{"seed":3}},"completed_episodes":299,"real_steps":109500,
                 "history":[{"validation":{"base_return":2}}]}
        with self.assertRaisesRegex(ValueError,"final 300"):
            verify_checkpoint(payload,manifest)
        payload["completed_episodes"]=300
        verify_checkpoint(payload,manifest)
        payload["history"][-1]["validation"]["base_return"]=1
        with self.assertRaisesRegex(ValueError,"validation disagree"):
            verify_checkpoint(payload,manifest)

    def test_metadata_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"metadata differs"):
            verify_checkpoint({"metadata":{"settings":{"seed":4}}},{"settings":{"seed":3}})

    def test_frozen_rollout_accounting_and_terminal_potential(self):
        data,config=demo_data(),demo_config()
        result=rollout(data,config,HistoricalPriceBaseStock(config).act,(61,90),.99)
        m=result["metrics"]
        self.assertAlmostEqual(m["base_return"],m["revenue"]-m["holding_cost"]-m["shortage_cost"])
        self.assertEqual(result["learning_updates"],0)
        self.assertEqual(len(result["daily"]),30)
        self.assertAlmostEqual(result["pbrs_telescoping_error"],0)
        expected=sum(.99**i*r["base_reward"] for i,r in enumerate(result["daily"]))
        self.assertAlmostEqual(m["discounted_base_return"],expected)

    def test_model_predictions_are_made_before_transition(self):
        data,config=demo_data(),demo_config()
        class Predictor:
            def predict(self,obs,actions):
                return (0,0,0)
        result=rollout(data,config,HistoricalPriceBaseStock(config).act,(61,90),.99,Predictor())
        errors=[d for r in result["daily"] for d in r["demand"]]
        self.assertAlmostEqual(result["model"]["demand_mae"],sum(errors)/len(errors))
        self.assertEqual(result["model"]["n_store_days"],90)

    def test_paired_targets_and_zero_denominator(self):
        results=[]
        for name in NAMES:
            for seed in ([None] if name=="H0" else SEEDS):
                results.append({"id":name,"seed":seed,"name":name if seed is None else f"{name}_seed{seed}",
                                "metrics":{"base_return":100 if name=="H0" else 110,
                                           "stockout_rate":.10 if name=="H0" else .20 if name=="S0" else .05,
                                           "fill_rate":.9,"lost_units":3}})
        report=aggregate_test(results)
        v0=[r for r in report["comparisons"] if r["experiment"]=="V0"]
        self.assertTrue(all(r["proxy_joint_target_met"] for r in v0))
        self.assertEqual(report["groups"]["H0"]["base_return"]["std"],None)
        self.assertEqual(len(report["paired_ablations"]),24)
        results[0]["metrics"]["stockout_rate"]=0
        report=aggregate_test(results)
        self.assertTrue(all(r["stockout_reduction_vs_H0"] is None for r in report["comparisons"]))
        self.assertFalse(any(r["proxy_joint_target_met"] for r in report["comparisons"]))

    def test_hacking_counterexample_separates_changed_objective(self):
        example=hacking_example()
        a,b=example["returns"]
        self.assertGreater(a["none"],b["none"])
        self.assertLess(a["naive"],b["naive"])
        self.assertAlmostEqual(a["pbrs"]-a["none"],b["pbrs"]-b["none"])

    def test_censored_runs_remain_in_learning_table(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"V0_seed3"
            folder.mkdir()
            history=[{"episode":i,"real_steps":365*i,"base_return":i,"synthetic_steps":0,
                      "training_seconds":1,"validation":{"base_return":10}} for i in range(1,31)]
            (folder/"training.json").write_text(json.dumps(history))
            rows=learning_diagnostics([{"id":"V0","name":"V0_seed3","seed":3}],Path(d),[],100)
            self.assertTrue(rows[0]["right_censored"])
            self.assertIsNone(rows[0]["steps_to_80pct"])
            self.assertEqual(rows[0]["budget"],109500)


    def test_report_config_accepts_numeric_equivalence_without_rewriting(self):
        from finalize_m5 import ROOT,validate_report_config
        expected=json.loads((ROOT/"docker/m5_final_assumptions.json").read_text())
        expected["price_multipliers"][2]=1
        actual=copy.deepcopy(expected)
        actual["price_multipliers"]=[.9,.95,1.0,1.05,1.1]
        actual["target_days"]=(0.0,3.0,7.0)
        before=copy.deepcopy(actual)
        self.assertNotEqual(json.dumps(actual),json.dumps(expected))
        validate_report_config(actual,expected,"H0")
        self.assertEqual(actual,before)
        self.assertIs(type(actual["price_multipliers"][2]),float)

    def test_report_config_rejects_real_changes_and_invalid_types(self):
        from finalize_m5 import ROOT,validate_report_config
        expected=json.loads((ROOT/"docker/m5_final_assumptions.json").read_text())
        for field,value in (
            ("holding_cost",expected["holding_cost"]+1e-12),
            ("store_capacity",expected["store_capacity"]+1),
            ("price_multipliers",[.95,.9,1.,1.05,1.1]),
            ("elasticities",[1.4,1.5,1.5]),
        ):
            actual=copy.deepcopy(expected)
            actual[field]=value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,"H0:.*"+field):
                    validate_report_config(actual,expected,"H0")
        for field,value in (("store_capacity",1050.0),("holding_cost","0.0128"),
                            ("holding_cost",float("nan")),("price_multipliers",[.9,.95,True,1.05,1.1])):
            actual=copy.deepcopy(expected)
            actual[field]=value
            with self.subTest(field=field,value=value):
                with self.assertRaises(ValueError):
                    validate_report_config(actual,expected,"H0")

    def test_full_audit_rejects_changed_budget_and_assumptions(self):
        from finalize_m5 import ROOT,canonical
        from src.retail.data import fingerprint
        from src.retail.deep_config import DeepConfig
        from types import SimpleNamespace
        from m5_pilot import assumptions
        # Use the actual pilot constructor, not a copy of the report JSON.
        means=(73.31709331131296,29.26094137076796,149.88521882741537)
        data_fixture=SimpleNamespace(sales=tuple((v,)*1211 for v in means),
                                     prices=((1.28,)*1211,)*3)
        config=assumptions(data_fixture).as_dict()
        self.assertIs(type(config["price_multipliers"][2]),float)
        code={p.name:fingerprint(p) for p in (ROOT/"src/retail").glob("*.py")}
        # Match train_tabular: paths relative to src, including its P1 dependency.
        tabular_code={p.relative_to(ROOT/"src").as_posix():fingerprint(p)
                      for p in [*sorted((ROOT/"src/retail").glob("*.py")), ROOT/"src/agents.py"]}
        data={"source":"M5","item_id":"FOODS_3_090","sha256":{"fixture":"hash"}}
        metrics={"base_return":1,"stockout_rate":.1,"fill_rate":.9,"lost_units":2}
        with tempfile.TemporaryDirectory() as d:
            suite=Path(d)
            (suite/"settings").mkdir()
            jobs=[]
            for name in NAMES:
                for seed in ([None] if name=="H0" else SEEDS):
                    run=name if seed is None else f"{name}_seed{seed}"
                    jobs.append({"id":name,"seed":seed})
                    folder=suite/run
                    folder.mkdir()
                    manifest={"config":config,"data":data,
                              "code_sha256":tabular_code if name=="T0" else code,"range":[1212,1576]}
                    if name=="H0":
                        manifest["metrics"]=metrics
                    else:
                        settings=DeepConfig(seed=seed,horizon=365).as_dict()
                        manifest.update(seed=seed,settings=settings,validation=metrics,validation_range=[1212,1576],
                                        completed_episodes=300,real_steps=109500)
                        history=[{"episode":i,"real_steps":i*365} for i in range(1,301)]
                        (folder/"training.json").write_text(json.dumps(history))
                        (folder/("q_tables.npz" if name=="T0" else "checkpoint.pt")).write_bytes(b"fixture")
                        if name!="T0":
                            (suite/"settings"/(run+".json")).write_text(json.dumps(settings))
                    (folder/"manifest.json").write_text(json.dumps(manifest))
            (suite/"suite.json").write_text(json.dumps({"jobs":jobs,"real_steps_per_trained_run":109500}))
            records,report,curves=audit(suite)
            self.assertEqual(len(records),25)
            self.assertTrue(report["complete"])
            # A hash mismatch or missing dependency must still fail closed.
            tabular_manifest=suite/"T0_seed3"/"manifest.json"
            original_tabular=json.loads(tabular_manifest.read_text())
            for key,digest,message in (
                ("agents.py","0"*64,"Code fingerprint mismatch: T0_seed3: agents.py"),
                ("missing.py","0"*64,"Missing fingerprinted source: T0_seed3: missing.py"),
            ):
                changed=copy.deepcopy(original_tabular)
                changed["code_sha256"][key]=digest
                tabular_manifest.write_text(json.dumps(changed))
                with self.subTest(source=key):
                    with self.assertRaisesRegex(ValueError,message):
                        audit(suite)
            tabular_manifest.write_text(json.dumps(original_tabular))
            bad=suite/"V3_seed5"/"manifest.json"
            original=json.loads(bad.read_text())
            changed=copy.deepcopy(original)
            changed["real_steps"]=100
            bad.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"budgets differ"):
                audit(suite)
            changed=copy.deepcopy(original)
            changed["config"]["elasticities"]=[.5,.5,.5]
            bad.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"different data"):
                audit(suite)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),"Report rendering runs on Pod")
    def test_delivery_generates_tables_figures_latex_notebook_and_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            report_smoke(directory)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch validation runs on Pod")
    def test_deep_inference_matches_existing_evaluator_and_preserves_weights(self):
        import torch
        from src.retail.deep import DDQN,PPO
        from src.retail.features import FeatureEncoder
        from src.retail.deep_config import DeepConfig
        from src.retail.train_deep import evaluate_policy
        data,config=demo_data(),demo_config()
        encoder=FeatureEncoder.fit(data,config,60)
        for algorithm,klass in (("ddqn",DDQN),("ppo",PPO)):
            torch.manual_seed(3)
            settings=DeepConfig(algorithm=algorithm)
            agent=klass(encoder.size,len(config.actions),settings,"cpu")
            nets=agent.online if algorithm=="ddqn" else agent.actors
            before=[p.detach().clone() for p in nets.parameters()]
            act=(lambda obs:agent.act(encoder.encode(obs))) if algorithm=="ddqn" else (
                lambda obs:agent.act(encoder.encode(obs),deterministic=True)[0])
            observed=rollout(data,config,act,(61,90),.99)
            expected=evaluate_policy(agent,encoder,data,config,(61,90),algorithm)
            for key,value in expected.items():
                self.assertEqual(observed["metrics"][key],value)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,nets.parameters())))


def report_smoke(directory):
    import sys, tempfile, json
    from pathlib import Path
    from unittest.mock import patch
    sys.path.insert(0,str(Path('scripts').resolve()))
    from finalize_m5 import NAMES,SEEDS,rollout,aggregate_test,learning_diagnostics
    from m5_delivery import deliver
    from src.retail.rollout import demo_data,demo_config
    from src.retail.baselines import HistoricalPriceBaseStock
    from src.retail.analysis import summary
    data,config=demo_data(),demo_config()
    class Predictor:
        def predict(self,obs,actions): return (0,0,0)
    smoke=Path(directory)
    smoke.mkdir(parents=True,exist_ok=True)
    suite=smoke/'suite'
    suite.mkdir(exist_ok=True)
    out=smoke/'output'
    out.mkdir(exist_ok=True)
    records=[]; results=[]; curves=[]
    for name in NAMES:
        for seed in ([None] if name=='H0' else SEEDS):
            run=name if seed is None else f'{name}_seed{seed}'
            result=rollout(data,config,HistoricalPriceBaseStock(config).act,(61,90),.99,
                           Predictor() if name in ('V2','V3') else None)
            result.update(id=name,seed=seed,name=run,cpu_validation_return_delta=0)
            results.append(result); records.append({'id':name,'seed':seed,'name':run})
            folder=suite/run; folder.mkdir(exist_ok=True)
            if name!='H0':
                history=[]
                for episode in range(1,301):
                    row={'episode':episode,'real_steps':365*episode,'base_return':100+episode+(seed or 0),
                         'training_seconds':1,'synthetic_steps':0,'losses':{'td_loss':1/(episode+1)},
                         'model_loss':1/(episode+1) if name in ('V2','V3') else None}
                    if name=='P0': row['losses']={'policy_loss':-.01,'value_loss':3/(episode+1)}
                    if episode%10==0:
                        row['validation']=result['metrics']
                        curves.append({'experiment':name,'seed':seed,'real_steps':365*episode,
                                       'base_return':result['metrics']['base_return'],'stockout_rate':result['metrics']['stockout_rate'],
                                       'synthetic_steps':0})
                    history.append(row)
                (folder/'training.json').write_text(json.dumps(history))
    test=aggregate_test(results)
    validation={'groups':test['groups']}
    diagnostics=learning_diagnostics(records,suite,curves,1500)
    with patch('m5_delivery.shutil.which',return_value=None):
        deliver(out,suite,records,validation,test,curves,diagnostics,results,
                {'threshold_reference':1500,'source_commit':'test-fixture-only'})
    tex=(out/'informe_p2.tex').read_text(encoding='utf8')
    assert '@@' not in tex
    assert len(list(out.glob('*.png'))) == 5
    assert len(list(__import__('csv').DictReader((out/'daily_store.csv').open())))==25*30*3
    assert len(json.loads((out/'analisis_p2.ipynb').read_text())['cells'])==4
    

if __name__=="__main__":
    unittest.main()
