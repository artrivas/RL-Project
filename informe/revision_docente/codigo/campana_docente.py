"""Campaña posterior con elasticidades asociativas de M5. Lanzar en el contenedor."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid

import torch
from finalize_m5 import audit, build_policy, csv_rows, read, rollout, run_name, save
from src.retail.checkpoint import load_checkpoint
from src.retail.config import RetailConfig, STORES
from src.retail.data import fingerprint, load_m5, SPLITS
from graficar_g0 import main as graficar
from finalizar_docente import entregar

ROOT = Path("/opt/retail")
OUT = Path("/salidas")
REVIEW = Path("/revision")
SUITE = OUT/"campana_calibrada"/"suite"
CONFIG = ROOT/"docker/m5_final_assumptions.json"


def train_job(job):
    name = run_name(job)
    folder = SUITE/name
    command = list(job["command"])
    if (folder/"manifest.json").is_file():
        print(f"{name}: se conserva la corrida terminada",flush=True)
        return
    checkpoint = folder/"checkpoint.pt"
    if checkpoint.exists():
        payload = load_checkpoint(checkpoint,"cpu")
        done = payload["completed_episodes"]
        del payload
        if 0 < done < 300:
            command.extend(["--resume",str(checkpoint)])
    if folder.exists() and "--resume" not in command:
        folder.rename(folder.with_name(f"{name}.interrumpido-{uuid.uuid4().hex[:8]}"))
    print(f"{name}: entrenamiento",flush=True)
    log = OUT/(name+".log")
    with log.open("a", encoding="utf8") as handle:
        subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print(f"{name}: completado",flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cal = read(OUT/"calibracion.json")
    data = load_m5(Path("/datos"),cal["producto"])
    if data.manifest["sha256"] != cal["hashes"] or cal["dias_ajuste"] != [1,1211]:
        raise ValueError("La calibración y los datos no coinciden")
    settings = read(REVIEW/"codigo/base_4644eb0/docker/m5_final_assumptions.json")
    settings["elasticities"] = [cal["tiendas"][s]["elasticidad_asociativa"] for s in STORES]
    cfg = RetailConfig(**settings)
    settings = cfg.as_dict()
    save(CONFIG,settings)
    lock = OUT/"protocolo_calibrado.json"
    sources = sorted((ROOT/"src").rglob("*.py"))
    protocol = dict(modelo="asociativo_por_tienda",entrenamiento=[1,1211],semillas=[3,4,5],
                    episodios=300,regla_checkpoint="último episodio",configuracion=settings,
                    huellas_datos=cal["hashes"],prueba="último año ya examinado",
                    codigo={str(p.relative_to(ROOT)):fingerprint(p) for p in sources},
                    ajustes=fingerprint(ROOT/"docker/m5_final_settings.json"),pytorch=str(torch.__version__))
    # Normalizar listas y tuplas como en el protocolo JSON persistido.
    protocol=json.loads(json.dumps(protocol))
    if lock.exists() and read(lock)!=protocol:
        raise ValueError("La salida corresponde a otra configuración; use una nueva carpeta de salida")
    save(lock,protocol)
    if not (SUITE/"suite.json").is_file():
        if SUITE.exists():
            SUITE.rename(SUITE.with_name("suite.interrumpida-"+uuid.uuid4().hex[:8]))
        subprocess.run([sys.executable,"-m","src.retail.suite","--data-dir","/datos",
            "--config",str(CONFIG),"--item-id",data.item_id,"--settings",
            str(ROOT/"docker/m5_final_settings.json"),"--seeds","3","4","5","--episodes","300",
            "--device","cpu","--output",str(SUITE)],cwd=ROOT,check=True)
    jobs=read(SUITE/"suite.json")["jobs"]
    workers=max(1,min(25,int(os.environ.get("M5_WORKERS","3"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(train_job,jobs))
    records,_,_=audit(SUITE)
    save(OUT/"auditoria_corridas.json",[{k:r[k] for k in ("name","manifest_sha256","history_sha256","checkpoint_sha256") if k in r} for r in records])
    g_rows=[]
    for r in records:
        if r["name"]=="H0":continue
        for x in read(SUITE/r["name"]/"training.json"):
            g=x["discounted_base_return"]
            if not math.isfinite(g): raise ValueError("G0 no finito")
            g_rows.append({"corrida":r["name"],"episodio":x["episode"],"g0":g,"retorno_base":x["base_return"]})
    if len(g_rows)!=7200:raise ValueError("Se esperaban 7200 retornos episódicos")
    csv_rows(OUT/"g0_calibrado.csv",g_rows)
    graficar(SUITE,OUT)
    metrics={}
    with torch.inference_mode():
        for split in ("validation","test"):
            results={}
            for r in records:
                if r["name"]=="H0":continue
                run_cfg,act,_,gamma=build_policy(r,SUITE)
                results[r["name"]]=rollout(data,run_cfg,act,SPLITS[split],gamma)["metrics"]
            for level in ("0.9","0.95"):
                actions=tuple(cfg.actions.index((1.0,k)) for k in cal["H0"][level]["dias_objetivo"])
                results["H0_"+level]=rollout(data,cfg,lambda obs:actions,SPLITS[split],.99)["metrics"]
            metrics[split]=results
    save(OUT/"metricas_calibradas.json",metrics)
    entregar(OUT,REVIEW)
    print("CIERRE COMPLETO: salidas_cierre/P2_final.zip",flush=True)

if __name__=="__main__":
    main()

