"""Calibración preliminar de H0 y elasticidad en M5. Ejecutar desde cualquier directorio."""
import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "base_4644eb0"
sys.path.insert(0, str(BASE))
from src.retail.config import RetailConfig, STORES
from src.retail.data import SPLITS, load_m5, fingerprint
from src.retail.env import RetailEnv
from src.retail.metrics import summarize

ITEM = "FOODS_3_090"
HASHES = {
    "sales_train_evaluation.csv": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
    "calendar.csv": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
    "sell_prices.csv": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
}

def obtain(root, download):
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = root / name
        if path.exists():
            if fingerprint(path) != expected:
                raise ValueError(f"SHA-256 distinto para {path}")
            continue
        if not download:
            raise FileNotFoundError(f"Falta {path}. Usar --descargar para obtener M5.")
        print(f"Descargando {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            candidate = Path(tmp) / name
            url = f"https://zenodo.org/records/10203108/files/{name}?download=1"
            with urllib.request.urlopen(url, timeout=120) as response, candidate.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            if fingerprint(candidate) != expected:
                raise ValueError(f"Descarga con huella distinta para {name}")
            if path.exists():
                raise FileExistsError(path)
            os.replace(candidate, path)


def design(calendar, prices):
    t = np.arange(len(prices))
    controls = [np.ones(len(t)), t / 365, np.sin(2*np.pi*t/365.25),
                np.cos(2*np.pi*t/365.25)]
    controls.extend([[r["weekday"] == k for r in calendar] for k in range(1, 7)])
    controls.extend([[r[key] for r in calendar] for key in ("snap", "event")])
    return np.column_stack([*controls, np.log(prices)])


def price_fit(x, y):
    # Drop calendar controls without independent variation. Keep price last
    # so a price perfectly confounded with calendar is explicitly rejected.
    columns = []
    for i in range(x.shape[1]):
        indices = [*columns, i]
        if np.linalg.matrix_rank(x[:, indices]) == len(indices):
            columns.append(i)
    if columns[-1] != x.shape[1]-1 or not np.any(y):
        return None
    a = x[:, columns]
    b = np.zeros(len(columns))
    b[0] = np.log(np.mean(y))
    for _ in range(80):
        z = a @ b
        mu = np.exp(z)
        grad = a.T @ (mu-y)/len(y)
        if np.max(np.abs(grad)) < 1e-8 * max(1.0, np.mean(y)):
            full = np.zeros(x.shape[1])
            full[columns] = b
            return full
        hessian = (a.T*mu) @ a / len(y)
        try:
            delta = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return None
        loss = np.mean(mu-y*z)
        for power in range(25):
            trial = b - delta/(2**power)
            eta = a @ trial
            if eta.max() < 100 and np.mean(np.exp(eta)-y*eta) <= loss:
                b = trial
                break
        else:
            return None
    return None


def store_profile(units, prices, calendar, targets):
    y, p = np.asarray(units), np.asarray(prices)
    x = design(calendar, p)
    coeff = price_fit(x, y)
    windows = []
    for end in (665, 847, 1029):
        fitted = price_fit(x[:end], y[:end])
        if fitted is not None:
            pred = np.exp(x[end:end+182] @ fitted)
            windows.append({"fin_ajuste": end, "dias_evaluados": len(pred),
                            "elasticidad_asociativa": float(-fitted[-1]),
                            "mae_ventas": float(np.abs(pred-y[end:end+182]).mean())})
    history = 28
    forecasts = np.convolve(y, np.ones(history)/history, mode="valid")[:-2]
    need = y[history:-1] + y[history+1:]
    # Two selling days cover review today and arrival tomorrow.
    cover = {k: float(np.mean(np.ceil(k*forecasts) >= need)) for k in targets}
    selected = {str(q): next((k for k in sorted(targets) if cover[k]>=q), None)
                for q in (.8,.9,.95)}
    return {"dias": len(y), "semanas": len({r["week"] for r in calendar}),
            "niveles_precio": len(set(p)), "cambios_precio": int(np.count_nonzero(np.diff(p))),
            "precio_min": float(p.min()), "precio_max": float(p.max()),
            "elasticidad_asociativa": None if coeff is None else float(-coeff[-1]),
            "validacion_temporal_interna": windows,
            "cobertura_empirica_dos_dias": cover, "objetivo_por_nivel_servicio": selected}


def validate_h0(data, config, targets):
    start, end = SPLITS["validation"]
    env = RetailEnv(data, config, start, end)
    actions = [config.actions.index((1.0, k)) for k in targets]
    env.reset()
    rows = []
    while not env.done:
        *_, row = env.step(actions)
        rows.append(row)
    result = summarize(rows)
    result["ingresos_m5_aproximados"] = sum(sum(data.sales[i][t]*data.prices[i][t]
        for i in range(3)) for t in range(start-1,end))
    result["mae_ventas_por_tienda"] = {s: float(np.mean([
        abs(r["sales"][i]-data.sales[i][r["day"]-1]) for r in rows])) for i,s in enumerate(STORES)}
    revenues = sum(data.prices[i][row["day"]-1]*q for row in rows for i,q in enumerate(row["sales"]))
    inventory_days = sum(sum(row["stock_after"]) for row in rows)
    shortages = sum(sum(row["lost"]) for row in rows)
    result["sensibilidad_costes"] = [
        {"factor_almacenamiento": h, "factor_penalizacion": c,
         "retorno": revenues - h*config.holding_cost*inventory_days - c*config.shortage_cost*shortages}
        for h in (0.5, 1.0, 2.0) for c in (0.5, 1.0, 2.0)
    ]
    return result


def calibrate(data, config):
    n = SPLITS["train"][1]
    profiles = {s: store_profile(data.sales[i][:n], data.prices[i][:n], data.calendar[:n],
                                config.target_days) for i,s in enumerate(STORES)}
    results = {}
    for q in ("0.8", "0.9", "0.95"):
        targets = [v["objetivo_por_nivel_servicio"][q] for v in profiles.values()]
        results[q] = {"dias_objetivo": targets, "validacion": None if None in targets
                      else validate_h0(data, config, targets)}
    return {"producto": data.item_id, "hashes": data.manifest["sha256"], "dias_ajuste": [1,n],
            "nivel_servicio_principal": .9, "tiendas": profiles, "H0": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=HERE.parent/"notebooks/data/m5/raw")
    parser.add_argument("--descargar", action="store_true")
    parser.add_argument("--salida", type=Path, default=HERE/"calibracion_m5.json")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error(f"La salida ya existe: {args.salida}")
    obtain(args.datos, args.descargar)
    data = load_m5(args.datos, ITEM)
    cfg = json.loads((BASE/"docker/m5_final_assumptions.json").read_text())
    report = calibrate(data, RetailConfig(**cfg))
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    with args.salida.open("x", encoding="utf-8") as f:
        f.write(payload+"\n")
    print(args.salida)

if __name__ == "__main__":
    main()
