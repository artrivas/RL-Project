"""Recalcula las metas de test y la sensibilidad de validación con los archivos de esta entrega."""
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT/"codigo"/"base_4644eb0"
sys.path.insert(0,str(BASE))
from src.retail.config import RetailConfig
from src.retail.env import RetailEnv
from src.retail.data import RetailData
from src.retail.metrics import summarize
from src.retail.tabular import CooperativeQLearning

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_rows(name,rows):
    with (ROOT/"evidencia"/name).open("w",newline="",encoding="utf-8") as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
def simulate(data,cfg,span,act,elasticities):
    env=RetailEnv(data,cfg,*span)
    obs=env.reset()
    rows=[]
    while obs is not None:
        actions=act(obs)
        day=env.day
        demand=[math.floor(data.sales[i][day-1]*cfg.actions[a][0]**(-elasticities[i])+0.5)
                for i,a in enumerate(actions)]
        obs,_,_,_,info=env.step_model(actions,demand)
        rows.append(info)
    return summarize(rows)
def main():
    data=RetailData(**read(ROOT/"demo/m5_producto.json"))
    cfg=RetailConfig(**read(BASE/"docker/m5_final_assumptions.json"))
    cal=read(ROOT/"evidencia/calibracion_m5.json")
    assert cal["hashes"]==data.manifest["sha256"]
    t0=CooperativeQLearning(cfg,3)
    with np.load(ROOT/"demo/t0_semilla3.npz",allow_pickle=False) as qs:
        for i,agent in enumerate(t0.agents):
            agent.Q[:]=qs[f"store_{i}"]
    pol={"T0_semilla3":t0.act}
    for q in ("0.9","0.95"):
        acts=tuple(cfg.actions.index((1.,k)) for k in cal["H0"][q]["dias_objetivo"])
        pol["H0_"+q]=lambda obs,a=acts:a
    signed=[cal["tiendas"][s]["elasticidad_asociativa"] for s in ("CA_1","CA_2","CA_3")]
    validations=[]
    for model,pars in (("campaña",cfg.elasticities),("asociación por tienda",signed)):
        for name,act in pol.items():
            metrics=simulate(data,cfg,(1212,1576),act,pars)
            validations.append(dict(modelo=model,politica=name,retorno=metrics["base_return"],
                tasa_quiebres=metrics["stockout_rate"],servicio_unidades=metrics["fill_rate"]))
    write_rows("sensibilidad_validacion.csv",validations)
    archived={p.stem:read(p)["metrics"] for p in (ROOT/"campana/test").glob("*.json")}
    if len(archived)!=25: raise ValueError("Se esperaban 25 trayectorias")
    checks=[]
    for h0 in ("H0_0.9","H0_0.95"):
        h=simulate(data,cfg,(1577,1941),pol[h0],cfg.elasticities)
        if h0.endswith("95"):
            for k in ("base_return","stockout_rate"): assert math.isclose(h[k],archived["H0"][k])
        for name,m in sorted(archived.items()):
            if name=="H0":continue
            selfish=archived["S0_seed"+name.split("seed")[1]]
            gh=m["base_return"]-h["base_return"]
            rh=1-m["stockout_rate"]/h["stockout_rate"]
            rs=1-m["stockout_rate"]/selfish["stockout_rate"]
            checks.append(dict(referencia=h0,corrida=name,retorno_H0=h["base_return"],
                quiebres_H0=h["stockout_rate"],ganancia=gh,reduccion_H0=rh,reduccion_S0=rs,
                cumple=gh>0 and rh>=.3 and rs>=.3))
    write_rows("meta_por_corrida.csv",checks)
if __name__=="__main__":main()
