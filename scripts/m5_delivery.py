"""Reporting only. Never trains, selects checkpoints, or modifies source runs."""
import csv
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from collections import defaultdict

NAMES = ("H0","T0","V0","P0","S0","V1","V2","V3","N0")
LABELS = {"H0":"Precio histórico + base-stock (proxy)", "T0":"Q-Learning tabular",
          "V0":"DDQN cooperativo", "P0":"PPO cooperativo", "S0":"DDQN egoísta",
          "V1":"DDQN + PBRS", "V2":"DDQN + Dyna", "V3":"DDQN + PBRS + Dyna",
          "N0":"DDQN + bono ingenuo"}

def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def write_csv(path, rows):
    if rows:
        with Path(path).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

def bands(ax, rows, name, key="base_return", label=None):
    pts=defaultdict(list)
    for r in rows:
        if r["experiment"]==name:
            pts[r["real_steps"]].append(r[key])
    x=sorted(pts)
    if not x:
        return
    mean=[statistics.mean(pts[t]) for t in x]
    std=[statistics.stdev(pts[t]) if len(pts[t])>1 else 0 for t in x]
    line,=ax.plot(x,mean,label=label or name,linewidth=1.4)
    ax.fill_between(x,[m-s for m,s in zip(mean,std)],[m+s for m,s in zip(mean,std)],
                    color=line.get_color(),alpha=.15)

def figures(out, suite, records, validation, test, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes=plt.subplots(3,1,figsize=(7,9),sharex=True)
    for ax,names,title in zip(axes,[("T0","V0","P0"),("V0","V1","N0"),("V0","V2","V3")],
                              ["Familias","Shaping","Modelo aprendido"]):
        for name in names:
            bands(ax,curves,name)
        ax.axhline(validation["groups"]["H0"]["base_return"]["mean"],color="black",ls="--",label="H0")
        ax.set(ylabel="Retorno base en validación",title=title)
        ax.legend(ncol=4)
    axes[-1].set_xlabel("Interacciones reales con el simulador")
    fig.tight_layout()
    fig.savefig(out/"validation_returns.png",dpi=180)
    plt.close(fig)
    histories,losses=[],[]
    for record in records:
        if record["id"]=="H0":
            continue
        rows=json.loads((suite/record["name"]/"training.json").read_text())
        for row in rows:
            histories.append({"experiment":record["id"],"seed":record["seed"],"episode":row["episode"],
                              "real_steps":row["real_steps"],"base_return":row["base_return"]})
            values=dict(row.get("losses",{}))
            values.update({k:row[k] for k in ("model_loss","synthetic_td_loss") if row.get(k) is not None})
            for key,value in values.items():
                losses.append({"experiment":record["id"],"seed":record["seed"],"real_steps":row["real_steps"],
                               "metric":key,"value":value})
    write_csv(out/"training_curves.csv",histories)
    write_csv(out/"loss_curves.csv",losses)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in NAMES[1:]:
        bands(ax,histories,name)
    ax.set(xlabel="Interacciones reales",ylabel="Suma de recompensas base (sin descuento)",
           title="Entrenamiento: ventanas anuales variables, media ±1σ")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out/"training_returns.png",dpi=180)
    plt.close(fig)
    loss_keys=sorted({(r["experiment"],r["metric"]) for r in losses})
    if loss_keys:
        fig,axes=plt.subplots(math.ceil(len(loss_keys)/3),3,figsize=(11,2.2*math.ceil(len(loss_keys)/3)),squeeze=False)
        for ax,(name,metric) in zip(axes.flat,loss_keys):
            bands(ax,[r for r in losses if r["metric"]==metric],name,"value")
            ax.set(title=f"{name}: {metric}",xlabel="Pasos reales",ylabel="Pérdida")
        for ax in list(axes.flat)[len(loss_keys):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out/"losses.png",dpi=150)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,5))
    for ax,metric,names,title in [
        (axes[0,0],"td_loss",("V0","V1","V2","V3","S0","N0"),"DDQN: Huber TD"),
        (axes[0,1],"policy_loss",("P0",),"PPO: pérdida de política"),
        (axes[1,0],"value_loss",("P0",),"PPO: pérdida de valor"),
        (axes[1,1],"model_loss",("V2","V3"),"Modelo: MSE de log(1+demanda)")]:
        for name in names:
            bands(ax,[r for r in losses if r["metric"]==metric],name,"value")
        ax.set(title=title,xlabel="Pasos reales",ylabel="Pérdida")
        if ax.lines:
            ax.legend(fontsize=7,ncol=3)
    fig.tight_layout()
    fig.savefig(out/"losses_summary.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,scale,label in [(axes[0],"base_return",1,"Retorno base"),
                                (axes[1],"stockout_rate",100,"Tienda-días con quiebre (%)")]:
        means=[test["groups"][n][key]["mean"]*scale for n in NAMES]
        std=[(test["groups"][n][key]["std"] or 0)*scale for n in NAMES]
        ax.bar(NAMES,means,yerr=std,capsize=3,color=["#737d85"]+["#197c96"]*8)
        ax.set(ylabel=label,title="Test: media ±1σ; H0 determinista")
    fig.tight_layout()
    fig.savefig(out/"test_results.png",dpi=180)
    plt.close(fig)

def demo(out, result):
    # JSON is local generated numeric data, with fixed names; no external scripts/assets.
    payload=json.dumps(result["daily"],allow_nan=False).replace("</","<\\/")
    page="""<!doctype html><html lang="es"><meta charset="utf-8">
<title>Demo P2: tres tiendas y almacén regional</title>
<style>body{font:18px system-ui;background:#edf4f5;color:#183039;max-width:1100px;margin:30px auto;padding:20px}
button,input{font:inherit}button{padding:10px 20px;background:#006d77;color:white;border:0;border-radius:6px}
table{width:100%;border-collapse:collapse;background:white}td,th{text-align:right;padding:12px;border-bottom:1px solid #ddd}
th:first-child,td:first-child{text-align:left}#timeline{width:70%}.card{background:white;padding:20px;margin:18px 0;border-radius:12px}
small{color:#455b63}h1{font-size:30px}</style>
<h1>Inventario compartido: CA_1, CA_2 y CA_3</h1>
<p>Replay de PPO, semilla 3, política congelada en test. Simulación basada en M5; no representa una operación real de Walmart.</p>
<button id="play">Reproducir</button> <input id="timeline" type="range" min="0" max="364" value="0">
<div class="card"><strong id="date"></strong><p id="central"></p><p id="reward"></p></div>
<table><thead><tr><th>Tienda</th><th>Precio</th><th>Stock inicial</th><th>Pedido</th><th>Despacho</th><th>Demanda</th><th>Ventas</th><th>Perdidas</th><th>Stock final</th></tr></thead><tbody id="stores"></tbody></table>
<p><small>Los despachos llegan al día siguiente. El beneficio mostrado resta almacenamiento y penalización de faltantes; no incluye costes de compra desconocidos. La demanda responde a elasticidad supuesta de 1.5.</small></p>
<script>const rows=__DATA__;let timer=null;const slider=document.querySelector('#timeline');slider.max=rows.length-1;
const money=x=>Number(x).toFixed(2);function show(){const i=Number(slider.value),r=rows[i];
document.querySelector('#date').textContent=r.date+' | día '+(i+1)+' / '+rows.length;
document.querySelector('#central').textContent='Centro: '+r.regional_before+' → '+r.regional_after+' unidades. Suministro externo: '+r.external_supply;
document.querySelector('#reward').textContent='Beneficio proxy del día: '+money(r.base_reward)+' | Acumulado: '+money(rows.slice(0,i+1).reduce((s,x)=>s+x.base_reward,0));
document.querySelector('#stores').innerHTML=['CA_1','CA_2','CA_3'].map((n,j)=>'<tr><td>'+n+'</td>'+[money(r.prices[j]),r.stock_before[j],r.requests[j],r.dispatched[j],r.demand[j],r.sales[j],r.lost[j],r.stock_after[j]].map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('');}
slider.oninput=show;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Reproducir';return}
document.querySelector('#play').textContent='Pausar';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%rows.length;show()},350)};show();</script></html>"""
    (out/"demo.html").write_text(page.replace("__DATA__",payload),encoding="utf-8")

def latex_number(value):
    return "--" if value is None else f"{value:,.1f}".replace(",","\\,")

def deliver(out,suite,records,validation,test,curves,diagnostics,results,protocol):
    figures(out,suite,records,validation,test,curves)
    table=[]
    for split,report in (("validation",validation),("test",test)):
        for name in NAMES:
            g=report["groups"][name]
            table.append({"split":split,"experiment":name,"method":LABELS[name],
                          "n":g["base_return"]["n"],"return_mean":g["base_return"]["mean"],
                          "return_std":g["base_return"]["std"],"stockout_mean":g["stockout_rate"]["mean"],
                          "stockout_std":g["stockout_rate"]["std"],"fill_mean":g["fill_rate"]["mean"],
                          "lost_mean":g["lost_units"]["mean"]})
    write_csv(out/"results_table.csv",table)
    model_rows=[{"run":r["name"],**r["model"]} for r in results if r["model"]]
    write_csv(out/"model_test_errors.csv",model_rows)
    daily=[]
    for r in results:
        for day in r["daily"]:
            for i,store in enumerate(("CA_1","CA_2","CA_3")):
                daily.append({"run":r["name"],"date":day["date"],"day":day["day"],"store":store,
                              **{key:day[key][i] for key in ("actions","prices","stock_before","requests",
                                                         "dispatched","demand","sales","lost","stock_after","local_rewards")}})
    write_csv(out/"daily_store.csv",daily)
    demo(out,next(r for r in results if r["name"]=="P0_seed3"))
    h0=test["groups"]["H0"]["base_return"]["mean"]
    best=max(NAMES[1:],key=lambda n:validation["groups"][n]["base_return"]["mean"])
    final=test["groups"][best]["base_return"]["mean"]
    joint=sum(r["proxy_joint_target_met"] for r in test["comparisons"])
    cpu_deltas=[{"run":r["name"],"return_delta":r["cpu_validation_return_delta"]} for r in results]
    write_csv(out/"cpu_validation_check.csv",cpu_deltas)
    readiness={
        "evaluation_complete":True, "n_learned_policies":24, "n_deterministic_references":1,
        "training_seeds":[3,4,5], "independent_confirmatory_training":False,
        "held_out_test_range":[1577,1941], "new_training_steps":0,
        "proxy_joint_target_runs":joint, "historical_target":"not identifiable in M5 inputs",
        "elasticity":"assumed 1.5, not estimated; requirement gap",
        "hacking_evidence":"two-step synthetic counterexample; N0 empirical outcome is separate",
        "asymptotic_convergence":"not demonstrated; report final finite-budget returns",
        "training_discounted_G0":"not logged; training curves are undiscounted; test includes discounted G0",
        "three_seed_dispersion":"sample standard deviation, not confidence intervals",
        "pdf_visual_review":"required after generation",
        "authors":"copied from P1: Leonardo Candio and Dimael Rivas; confirm before submission",
        "oral_presentation":"20-minute outline and demo supplied; presentation itself pending",
        "external_teacher_clarifications":"historical proxy, one-SKU scope and P1 tabular transfer not approved by evidence"}
    save(out/"delivery_checklist.json",readiness)
    rows=[]
    for name in NAMES:
        g=test["groups"][name]
        rows.append(f'{name} & {latex_number(validation["groups"][name]["base_return"]["mean"])} & '
                    f'{latex_number(g["base_return"]["mean"])} & {latex_number(g["base_return"]["std"])} & '
                    f'{100*g["stockout_rate"]["mean"]:.2f} & {100*g["fill_rate"]["mean"]:.2f} \\\\')
    learning_rows=[]
    for name in NAMES[1:]:
        ds=[d for d in diagnostics if d["experiment"]==name]
        hit=[d["steps_to_80pct"] for d in ds if not d["right_censored"]]
        learning_rows.append(f'{name} & {len(hit)}/3 & {latex_number(statistics.median(hit)) if hit else "--"} & '
                             f'{latex_number(statistics.mean(d["tail_30_train_return_std"] for d in ds))} & '
                             f'{latex_number(statistics.mean(d["tail_10_validation_return_std"] for d in ds))} \\\\')
    model_tex=[]
    for name in ("V2","V3"):
        ms=[m for m in model_rows if m["run"].startswith(name+"_")]
        model_tex.append(f'{name} & {statistics.mean(m["demand_mae"] for m in ms):.2f} & '
                         f'{statistics.mean(m["demand_rmse"] for m in ms):.2f} & '
                         f'{statistics.mean(m["forecast_mae"] for m in ms):.2f} \\\\')
    pair_tex=[]
    for contrast in dict.fromkeys(p["contrast"] for p in test["paired_ablations"]):
        values=[p["return_difference"] for p in test["paired_ablations"] if p["contrast"]==contrast]
        pair_tex.append(f'{contrast.replace(" minus "," - ")} & {latex_number(statistics.mean(values))} & '
                        f'{latex_number(statistics.stdev(values))} & {sum(v>0 for v in values)}/3 \\\\')
    changes=sum(abs(d["return_delta"])>.01 for d in cpu_deltas)
    tokens={"RESULT_ROWS":"\n".join(rows),"LEARNING_ROWS":"\n".join(learning_rows),
            "MODEL_ROWS":"\n".join(model_tex),"PAIR_ROWS":"\n".join(pair_tex),
            "BEST_VALIDATION":best,"BEST_TEST_RETURN":latex_number(final),"H0_TEST":latex_number(h0),
            "BEST_DELTA":latex_number(final-h0),"JOINT_SUCCESSES":str(joint),
            "THRESHOLD":latex_number(.8*protocol["threshold_reference"]),
            "CPU_CHANGES":str(changes),"COMMIT":protocol["source_commit"][:12]}
    template=(Path(__file__).resolve().parents[1]/"informe/p2_ieee_template.tex").read_text(encoding="utf-8")
    for key,value in tokens.items():
        template=template.replace("@@"+key+"@@",value)
    if "@@" in template:
        raise ValueError("Unfilled LaTeX report field")
    (out/"informe_p2.tex").write_text(template,encoding="utf-8")
    compiled=False
    if shutil.which("pdflatex"):
        with (out/"latex-build.log").open("w",encoding="utf-8") as log:
            for _ in range(2):
                subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","informe_p2.tex"],
                               cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        from pypdf import PdfReader
        pages=len(PdfReader(out/"informe_p2.pdf").pages)
        save(out/"pdf_check.json",{"pages":pages,"within_eight_page_limit":pages<=8,"visual_review":"pending"})
        if pages>8:
            raise ValueError("IEEE report exceeds eight pages; shorten the report without changing results.")
        compiled=True
    report=f"""# P2 Vía B: cierre experimental

Se evaluaron 24 políticas congeladas y H0 en el último año (d1577-d1941).
Tres semillas de desarrollo: 3, 4, 5; 109500 interacciones reales por política.
No se ejecutó entrenamiento adicional. Resultados descriptivos; no prueban significancia ni convergencia.

La familia con mayor retorno medio de validación fue {best}.
Su retorno medio de test es {final:.2f}, frente a H0={h0:.2f}; diferencia={final-h0:.2f}.
Las comparaciones completas están en results_table.csv y paired_ablations.csv.
{joint}/24 corridas satisfacen conjuntamente los dos umbrales de stockouts y el beneficio frente a los proxies.
Esto no certifica superar beneficio o quiebres históricos reales de Walmart.

## Archivos principales

- informe_p2.tex y {'informe_p2.pdf' if compiled else 'PDF pendiente: instalar pdflatex y volver a ejecutar el cierre'}: artículo IEEE, máximo 8 páginas.
- validation_returns.png, training_returns.png, losses.png: curvas y dispersión.
- test_results.png y results_table.csv: comparación final.
- learning_diagnostics.csv: eficiencia al umbral exploratorio y estabilidad.
- model_test_errors.csv: error de predicción de demanda en trayectorias de test, sin ajustes.
- daily_store.csv: trazas de los tres agentes.
- demo.html: replay interactivo de PPO semilla 3, elección fijada antes de test.
- analisis_p2.ipynb: cuaderno de lectura reproducible del paquete.
- protocol.json y SHA256SUMS.json: datos, fuentes, checkpoints y hashes.
- delivery_checklist.json: requisitos satisfechos, límites y pendientes.

## Alcance y limitaciones

La elasticidad de 1.5 es supuesta, no estimada. Las ventas son un proxy de demanda.
Un solo SKU y un único año no permiten generalizar a Walmart.
No hay observación de inventarios, pedidos históricos ni costes de compra.
El plan inicial reservaba semillas 0,1,2 para confirmación; se documenta su omisión por cierre de alcance.
El umbral de eficiencia es descriptivo, definido después del desarrollo y antes de test.
Las curvas de entrenamiento suman recompensas sin descuento; el G0 descontado sólo se calcula en evaluación.
La cola de entrenamiento mezcla ventanas temporales; la cola de validación usa la misma ventana.
{changes} corridas cambian el retorno de validación más de 0.01 al pasar a CPU; consultar cpu_validation_check.csv.
El contraejemplo de hacking es sintético; no se atribuye automáticamente a la política N0 en M5.
No hay evidencia de aprobación docente de las aproximaciones históricas.

## Revisión antes de entregar

Revisar visualmente el PDF y confirmar autores heredados de P1.
Leer las conclusiones junto con target_checks.csv; terminar el proceso no equivale a cumplir el objetivo cuantitativo.
La sustentación oral sigue pendiente. No se ha enviado nada a Canvas ni integrado la rama a main.
"""
    (out/"LEEME.md").write_text(report,encoding="utf-8")
    cells=[{"cell_type":"markdown","metadata":{},"source":["# P2: resultados congelados\n",
             "Este cuaderno lee los resultados del paquete. No entrena ni selecciona políticas.\n",
             "Ejecutar junto a test_summary.json. Véase LEEME.md para límites del estudio."]},
           {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
               "import json\nfrom pathlib import Path\nfrom IPython.display import display, Image\n",
               "results = json.loads(Path('test_summary.json').read_text())\nresults['groups']"]},
           {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
               "display(Image('test_results.png'))\ndisplay(Image('validation_returns.png'))\ndisplay(Image('losses.png'))"]},
           {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
               "import csv\nlist(csv.DictReader(open('target_checks.csv', encoding='utf-8')))"]}]
    save(out/"analisis_p2.ipynb",{"nbformat":4,"nbformat_minor":5,
                                "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
                                "cells":cells})
    (out/"sustentacion_20_min.md").write_text("""# Sustentación (20 minutos)
1. Min 0-2: problema retail, tres tiendas, inventario regional; distinguir datos y supuestos.
2. Min 2-5: observaciones, 15 acciones, demanda, lead time y recompensa base.
3. Min 5-8: Q-Learning, DDQN y PPO; misma información y presupuesto de interacciones.
4. Min 8-11: derivar telescopado PBRS; mostrar hacking_example.json y aclarar alcance sintético.
5. Min 11-13: Dyna híbrido: demanda aprendida, contabilidad exacta, sin inventar el objetivo.
6. Min 13-16: curvas y tablas de test, desviación entre semillas, ablaciones y metas incumplidas.
7. Min 16-18: abrir demo.html, reproducir y pausar un día con racionamiento.
8. Min 18-20: límites (elasticidad, proxy histórico, un SKU, tres semillas de desarrollo) y conclusiones.

Preguntas: ¿por qué puede ganar la heurística? ¿PBRS garantiza mejor entrenamiento?
¿Un modelo aprende o define la recompensa? ¿Por qué más pasos sintéticos pueden perjudicar?
¿Qué cambia al comparar interacciones reales y tiempo? ¿Se alcanzó el 30% frente a ambas referencias?
Responder con los resultados del paquete; no afirmar significancia con tres semillas.
""",encoding="utf-8")
