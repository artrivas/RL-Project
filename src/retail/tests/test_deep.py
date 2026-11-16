"""These tests MUST execute (not skip) before any M5 campaign is launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.retail.deep import DDQN,PPO,double_q_target,gae,clipped_policy_loss
    from src.retail.deep_config import DeepConfig
    from src.retail.features import FeatureEncoder
    from src.retail.world_model import DemandModel
    from src.retail.train_deep import run
    from src.retail.checkpoint import load_checkpoint
    from src.retail.rollout import demo_data,demo_config
    from src.retail.env import RetailEnv

@unittest.skipUnless(HAS_TORCH,"PyTorch unavailable: deep numerical validation is pending")
class DeepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def assert_nested_equal(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.assert_nested_equal(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested_equal(x,y)
        else:
            self.assertEqual(a,b)

    def test_double_q_selection_differs_from_target_max_and_masks_terminal(self):
        actual=double_q_target(torch.tensor([[1.,4.],[1.,4.]]),
                               torch.tensor([[10.,2.],[10.,2.]]),
                               torch.tensor([1.,1.]),torch.tensor([0.,1.]),.9)
        torch.testing.assert_close(actual,torch.tensor([2.8,1.]))

    def test_gae_terminal_and_rollout_cut(self):
        rewards=torch.ones(2,3)
        values=torch.full((2,3),.5)
        next_values=torch.full((2,3),.5)
        terminals=torch.tensor([[0.,0.,0.],[1.,1.,1.]])
        adv,returns=gae(rewards,values,next_values,terminals,.9,1.)
        torch.testing.assert_close(returns[0],torch.full((3,),1.9))
        torch.testing.assert_close(returns[1],torch.ones(3))
        _,cut_returns=gae(rewards,values,next_values,torch.zeros(2,3),.9,1.)
        torch.testing.assert_close(cut_returns[1],torch.full((3,),1.45))

    def test_ppo_clipping_signs(self):
        loss=clipped_policy_loss(torch.tensor([2.,.5]).log(),torch.zeros(2),torch.tensor([1.,-1.]),.2)
        self.assertAlmostEqual(loss.item(),-.2,places=6)

    def test_ddqn_updates_online_but_not_target_before_interval(self):
        c=DeepConfig(hidden=8,batch_size=2,warmup_steps=2,target_interval=99)
        agent=DDQN(4,3,c)
        original=[p.detach().clone() for p in agent.online.parameters()]
        target=[p.detach().clone() for p in agent.target.parameters()]
        for _ in range(2):
            agent.remember(np.ones((3,4),dtype=np.float32),(0,1,2),(1,2,3),
                           np.zeros((3,4),dtype=np.float32),True)
        self.assertTrue(np.isfinite(agent.learn()["td_loss"]))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original,agent.online.parameters())))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(target,agent.target.parameters())))

    def test_ppo_update_is_finite_and_changes_parameters(self):
        c=DeepConfig(algorithm="ppo",hidden=8,batch_size=2,ppo_epochs=2)
        agent=PPO(4,3,c)
        original=[p.detach().clone() for p in agent.parameters]
        for t in range(4):
            obs=np.full((3,4),t/4,dtype=np.float32)
            actions,logp,values=agent.act(obs)
            agent.remember(obs,actions,(1.,2.,3.),obs,t==3,logp,values)
        losses=agent.learn()
        self.assertTrue(all(np.isfinite(x) for x in losses.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original,agent.parameters)))
        self.assertEqual(len(agent.buffer),0)

    def test_dyna_never_adds_synthetic_data_to_real_buffers(self):
        data,config=demo_data(),demo_config()
        settings=DeepConfig(hidden=8,batch_size=1,model_warmup=1,dyna_steps=2)
        encoder=FeatureEncoder.fit(data,config,60)
        agent=DDQN(encoder.size,len(config.actions),settings)
        model=DemandModel(encoder,len(config.actions),settings)
        env=RetailEnv(data,config,29,30)
        obs=env.reset(); snapshot=env.snapshot()
        actions=agent.act(encoder.encode(obs))
        nxt,reward,done,_,info=env.step(actions)
        agent.remember(encoder.encode(obs),actions,[reward]*3,encoder.encode(nxt),done)
        model.observe_real(snapshot,obs,actions,info["demand"])
        self.assertTrue(np.isfinite(model.learn()))
        model.plan(agent,data,config)
        self.assertEqual(len(model.real),1)
        self.assertEqual(len(agent.replay),1)
        self.assertEqual(model.synthetic_steps,2)

    def test_episode_boundary_resume_matches_uninterrupted_training(self):
        data,config=demo_data(),demo_config()
        variants=[{"algorithm":"ddqn"},{"algorithm":"ppo"},
                  {"algorithm":"ddqn","dyna_steps":1,"shaping":"pbrs"}]
        for variant in variants:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                settings=DeepConfig(seed=3,horizon=2,hidden=8,batch_size=2,warmup_steps=2,
                                    model_warmup=2,ppo_epochs=1,rollout_steps=2,**variant)
                root=Path(directory)
                full=run(data,config,settings,root/"full",2,eval_every=1)
                run(data,config,settings,root/"resume",1,eval_every=1)
                resumed=run(data,config,settings,root/"resume",2,eval_every=1,
                            resume=root/"resume/checkpoint.pt")
                a,b=load_checkpoint(root/"full/checkpoint.pt"),load_checkpoint(root/"resume/checkpoint.pt")
                self.assert_nested_equal(a["agent"],b["agent"])
                self.assert_nested_equal(a["model"],b["model"])
                self.assert_nested_equal(a["rng"],b["rng"])
                self.assertEqual(full["validation"],resumed["validation"])
                self.assertFalse(resumed["test_evaluated"])

    def test_frozen_evaluation_is_reproducible(self):
        from src.retail.train_deep import evaluate_policy
        data,config=demo_data(),demo_config()
        encoder=FeatureEncoder.fit(data,config,60)
        for algorithm,klass in (("ddqn",DDQN),("ppo",PPO)):
            agent=klass(encoder.size,len(config.actions),DeepConfig(algorithm=algorithm,hidden=8))
            before={k:v.clone() for k,v in (agent.online if algorithm=="ddqn" else agent.actors).state_dict().items()}
            a=evaluate_policy(agent,encoder,data,config,(61,65),algorithm)
            b=evaluate_policy(agent,encoder,data,config,(61,65),algorithm)
            self.assertEqual(a,b)
            self.assert_nested_equal(before,(agent.online if algorithm=="ddqn" else agent.actors).state_dict())
