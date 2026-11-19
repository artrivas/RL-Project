"""Reconstrucción de T0 con las mismas decisiones de entrenamiento y registro de G0."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

from calibrar_m5 import BASE, HASHES, HERE, ITEM, load_m5, RetailConfig, RetailEnv
from src.retail.tabular import CooperativeQLearning
from src.retail.metrics import summarize

def run(args):
    data = load_m5(args.datos, ITEM)
    if data.manifest["sha256"] != HASHES:
        raise ValueError("Huellas de M5 inesperadas")
    cfg = RetailConfig(**json.loads((BASE/"docker/m5_final_assumptions.json").read_text()))
    with (args.archivo/"training_curves.csv").open() as f:
        old = {(int(r["seed"]), int(r["episode"])):float(r["base_return"])
               for r in csv.DictReader(f) if r["experiment"] == "T0"}
    evidence = args.salida/"evidencia"
    demo = args.salida/"demo"
    demo.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    (demo/"m5_producto.json").write_text(json.dumps(asdict(data),separators=(",",":")),encoding="utf-8")
    stats = []
    curve = []
    for seed in (3,4,5):
        print(f"Reproduciendo T0, semilla {seed}", flush=True)
        policy = CooperativeQLearning(cfg,seed)
        rng = random.Random(seed+300)
        for episode in range(300):
            start = rng.randint(cfg.history_days+1,1211-365+1)
            env = RetailEnv(data,cfg,start,start+364)
            obs = env.reset()
            total = g0 = 0.0
            for day in range(365):
                actions = policy.act(obs,1.0-0.95*episode/299)
                nxt,reward,terminal,_,_ = env.step(actions)
                policy.observe(obs,actions,reward/100.0,nxt,terminal)
                total += reward
                g0 += (0.99**day)*reward
                obs = nxt
            diff = abs(total-old[seed,episode+1])
            curve.append((seed,episode+1,total,g0,diff))
        env = RetailEnv(data,cfg,1577,1941)
        obs = env.reset()
        rows = []
        while obs is not None:
            obs,_,_,_,row = env.step(policy.act(obs))
            rows.append(row)
        current = summarize(rows)
        archived = json.loads((args.archivo/"test"/f"T0_seed{seed}.json").read_text())["metrics"]
        stats.append(dict(semilla=seed,retorno_test=current["base_return"],
            diferencia_test=current["base_return"]-archived["base_return"],
            diferencia_quiebres=current["stockout_rate"]-archived["stockout_rate"],
            maxima_diferencia_episodica=max(r[-1] for r in curve if r[0]==seed)))
        if seed == 3:
            policy.save(demo/"t0_semilla3.npz")
        print(stats[-1],flush=True)
    with (evidence/"g0_tabular.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["semilla","episodio","retorno_base","g0","diferencia_absoluta_archivo"])
        writer.writerows(curve)
    (evidence/"verificacion_tabular.json").write_text(json.dumps(stats,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos",type=Path,default=HERE.parent/"notebooks/data/m5/raw")
    p.add_argument("--archivo",type=Path,default=HERE.parent/"informe/entrega_p2_revisada/resultados_originales")
    p.add_argument("--salida",type=Path,default=HERE.parent/"informe/revision_docente")
    run(p.parse_args())
