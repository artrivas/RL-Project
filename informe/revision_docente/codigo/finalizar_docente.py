"""Empaqueta exclusivamente resultados completos y un manuscrito compilado."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import zipfile

def comparar(metrics):
    checks=[]
    for split,runs in metrics.items():
        for ref in ("H0_0.9","H0_0.95"):
            h=runs[ref]
            for name,m in sorted(runs.items()):
                if not name.startswith("H0_"):
                    s=runs["S0_seed"+name.split("_seed")[1]]
                    reduction=lambda v:1-m["stockout_rate"]/v if v>0 else None
                    rh,rs=reduction(h["stockout_rate"]),reduction(s["stockout_rate"])
                    delta=m["base_return"]-h["base_return"]
                    checks.append(dict(periodo=split,referencia=ref,corrida=name,
                        ganancia=delta,reduccion_H0=rh,reduccion_S0=rs,
                        cumple=None if rh is None or rs is None else delta>0 and min(rh,rs)>=.30))
    return checks

def fr(value):
    return f"{value:,.2f}".replace(","," ").replace(".",",")

def entregar(out,review):
    from pypdf import PdfReader
    from pptx import Presentation
    from pptx.util import Inches,Pt
    metrics=json.loads((out/"metricas_calibradas.json").read_text())
    checks=comparar(metrics)
    if len(checks)!=96:raise ValueError("Se esperaban 96 comparaciones")
    with (out/"metas_calibradas.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        dest=Path(tmp)/"P2_final"
        shutil.copytree(review,dest,ignore=shutil.ignore_patterns("salidas_cierre","salidas_docker","__pycache__","*.pyc","SHA256_REVISION.json",".git"))
        ev=dest/"evidencia"
        for f in ("protocolo_calibrado.json","auditoria_corridas.json","g0_calibrado.csv","g0_entrenamiento.png","metricas_calibradas.json","metas_calibradas.csv"):
            shutil.copy2(out/f,ev/f)
        selected=[c for c in checks if c["periodo"]=="test" and c["corrida"].startswith("P0_")]
        def count(ref):
            xs=[c["cumple"] for c in selected if c["referencia"]==ref]
            return f"{xs.count(True)}/3"+(f" ({xs.count(None)} indet.)" if None in xs else "")
        avg=statistics.mean(metrics["test"][f"P0_seed{s}"]["base_return"] for s in (3,4,5))
        section=[r"\section*{VII. Campaña posterior con elasticidades estimadas}",
            "Se repitieron las ocho condiciones del estudio con las mismas tres semillas, 300 episodios por corrida y las elasticidades asociativas estimadas exclusivamente en entrenamiento. Se conservaron los signos de los coeficientes. Las asociaciones negativas de dos tiendas limitan su interpretación ante intervenciones reales de precio. El análisis utiliza un año de prueba ya examinado. Las secciones anteriores describen la primera campaña.",
            f"En esta campaña PPO obtuvo un retorno medio de {fr(avg)} en test. La meta conjunta se alcanzó en {count('H0_0.9')} semillas con H0 al 90\\% y {count('H0_0.95')} con H0 al 95\\%. Los contrastes individuales constan en el CSV de evidencia. Una tasa de quiebres de referencia cero deja indefinido el contraste relativo.",
            r"\begin{figure*}[!ht]\centering\includegraphics[width=.82\textwidth,height=.62\textheight,keepaspectratio]{evidencia/g0_entrenamiento.png}\caption{Retorno de entrenamiento descontado de la campaña con elasticidades estimadas. Media y desviación estándar muestral sobre tres semillas por condición.}\end{figure*}"]
        manuscript=(dest/"informe_p2_revision.tex").read_text(encoding="utf8")
        manuscript=manuscript.replace(r"\section*{Referencias}","\n\n".join(section)+"\n"+r"\section*{Referencias}")
        source=dest/"informe_final.tex";source.write_text(manuscript,encoding="utf8")
        for _ in range(2):
            subprocess.run(["pdflatex","-halt-on-error","-interaction=batchmode",source.name],cwd=dest,check=True)
        pdf=dest/"informe_final.pdf";doc=PdfReader(pdf)
        if len(doc.pages)>8 or not any("estimadas" in p.extract_text() for p in doc.pages):
            raise ValueError("Revisar el PDF compilado y el límite de ocho páginas")
        prs=Presentation(dest/"presentacion_p2_revision.pptx")
        slide=prs.slides.add_slide(prs.slides[0].slide_layout)
        def box(y,text,size):
            sh=slide.shapes.add_textbox(Inches(.75),Inches(y),Inches(11.8),Inches(1))
            sh.text_frame.word_wrap=True
            r=sh.text_frame.paragraphs[0].add_run();r.text=text;r.font.size=Pt(size)
        box(.45,"Campaña con elasticidades estimadas por tienda",32)
        box(1.7,"24 políticas · 3 semillas · 300 episodios por corrida",23)
        box(2.8,f"Retorno medio de PPO en el último año: {fr(avg)}.",23)
        box(3.8,f"Meta conjunta con H0 al 90 %: {count('H0_0.9')}. Con H0 al 95 %: {count('H0_0.95')}.",23)
        box(5.1,"La respuesta al precio es asociativa. El último año se había examinado en la campaña original.",20)
        prs.save(dest/"presentacion_final.pptx")
        (dest/"ABRIR_PRIMERO.md").write_text("# Entrega P2\n\nAbrir **informe_final.pdf**, **presentacion_final.pptx** y **demo_en_vivo.html**. Los archivos con el sufijo revision corresponden a la primera campaña. La última sección del informe y la última diapositiva contienen los resultados de la campaña calibrada. Los checkpoints de esta campaña se conservan en el directorio de salida de Docker.\n",encoding="utf8")
        evidence=dict(entorno="contenedor Linux CPU",políticas_entrenadas=24,episodios_con_G0=7200,
                      comparaciones=len(checks),paginas=len(doc.pages),pptx="presentacion_final.pptx")
        (ev/"verificacion_final.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf8")
        for ext in ("aux","log","out"): (dest/("informe_final."+ext)).unlink(missing_ok=True)
        hashes={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dest.rglob("*")) if p.is_file()}
        (dest/"SHA256_FINAL.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")
        zipname=Path(shutil.make_archive(str(Path(tmp)/"P2_final"),"zip",root_dir=tmp,base_dir="P2_final"))
        with zipfile.ZipFile(zipname) as z:
            if z.testzip() is not None:raise IOError("ZIP con corrupción")
            for name,value in hashes.items():
                if hashlib.sha256(z.read("P2_final/"+name)).hexdigest()!=value:raise IOError(name)
        zipname.replace(out/"P2_final.zip")

