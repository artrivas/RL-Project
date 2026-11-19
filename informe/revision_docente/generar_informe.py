"""Genera la versión de lectura y el manuscrito IEEE a partir de los mismos datos."""
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

D=Path(__file__).resolve().parent
def csvrows(rel):
    with (D/rel).open(encoding="utf8") as f:return list(csv.DictReader(f))
def num(x,n=1):
    return f"{float(x):,.{n}f}".replace(","," ").replace(".",",")
def make_curve():
    import os
    g=defaultdict(list)
    for r in csvrows("evidencia/g0_tabular.csv"):g[int(r["episodio"])].append(float(r["g0"]))
    im=PILImage.new("RGB",(1050,450),"white")
    draw=ImageDraw.Draw(im)
    try:font=ImageFont.truetype("DejaVuSans.ttf",24)
    except OSError:font=ImageFont.load_default(size=24)
    mean=[st.mean(g[i]) for i in range(1,301)]
    sd=[st.stdev(g[i]) for i in range(1,301)]
    low=[v-s for v,s in zip(mean,sd)]
    high=[v+s for v,s in zip(mean,sd)]
    lower=5000*math.floor(min(low)/5000)
    upper=5000*math.ceil(max(high)/5000)
    def xy(i,v):return (90+i*920/299,368-(v-lower)*292/(upper-lower))
    draw.text((85,10),"Retorno descontado (miles de unidades monetarias)",font=font,fill="#233f45")
    for y in range(lower,upper+1,10000):
        a,b=xy(0,y)
        draw.line((a,b,1010,b),fill="#dae3e0",width=2)
        draw.text((a-10,b),str(y//1000),anchor="rm",font=font,fill="#354f52")
    draw.polygon([xy(i,v) for i,v in enumerate(high)]+[xy(i,low[i]) for i in reversed(range(300))],fill="#c2dfda")
    draw.line([xy(i,v) for i,v in enumerate(mean)],fill="#175951",width=3)
    for i in (0,99,199,299):
        draw.text((xy(i,0)[0],374),str(i+1),font=font,anchor="mt",fill="#354f52")
    draw.text((545,411),"Episodio",font=font,anchor="mt",fill="#354f52")
    im.save(D/"evidencia/g0_tabular.png")
make_curve()
pages=[]
def column(): 
    global col
    col=[];pages.append(col)
def h(s):col.append(("h",s))
def p(s):col.append(("p",s))
def t(caption,rows):col.append(("t",(caption,rows)))
def fig(path,caption):col.append(("f",(path,caption)))

column()
h("Resumen")
p("Estudiamos cómo coordinar precios y reposición cuando tres tiendas de California compiten por el mismo inventario regional. Sobre ventas y precios de Walmart M5 entrenamos Q-learning, Double DQN y PPO, y examinamos el efecto del diseño de la recompensa y de la experiencia sintética. En el último año, PPO superó el beneficio de la referencia inducida al 90 % y redujo los quiebres en al menos 30 % frente a ambas referencias en dos de tres semillas. Con cobertura al 95 % ninguna política alcanzó el objetivo conjunto. La reposición inferida y la respuesta supuesta al precio condicionan la interpretación económica.")
h("I. Datos y proceso de decisión")
p("Usamos FOODS_3_090 en CA_1, CA_2 y CA_3. Los tres CSV de M5 contienen ventas diarias, precios semanales y calendario [1]. Los días 1–1211 se usaron para entrenamiento, 1212–1576 para validación y 1577–1941 para el test de la campaña original. La revisión de la referencia histórica se hizo después de conocer ese último año.")
p("El estado conjunto contiene el día, existencias locales y regionales, pedidos en tránsito e historial de ventas. Cada agente recibe una observación local, con calendario, precio base, stock regional, horizonte y pronóstico móvil de 28 días. Esta observación parcial limita la información para decidir. Cada tienda elige un multiplicador de precio entre 0,90 y 1,10 en pasos de 0,05 y una cobertura de 0, 3 o 7 días.")
p("El pedido cubre la diferencia entre el objetivo y las existencias, sujeto a capacidad local. El centro distribuye hasta su inventario disponible en proporción a las solicitudes y resuelve los remanentes por tienda. Los despachos llegan al día siguiente. Las capacidades son 1050 unidades por tienda y 1768 regionales, con suministro diario máximo de 253.")
column()
h("II. Objetivo y agentes")
p("Con ventas históricas y, multiplicador m y elasticidad e, la demanda diaria simulada es el entero más próximo a y × m elevado a −e. Las ventas q están limitadas por las existencias disponibles. La recompensa regional suma, en las tres tiendas, precio × q menos 0,0128 por unidad almacenada al cierre y 1,28 por unidad perdida. Se reporta en unidades monetarias del simulador. M5 carece de costes de compra e inventarios diarios para este cálculo.")
p("T0 reutiliza Q-learning de P1 con una tabla nueva por tienda y discretización de la observación. V0 aprende el valor de las acciones con Double DQN, replay y una red objetivo [2]. La red en línea elige la acción siguiente y la red objetivo la valora. P0 usa PPO con actores y críticos locales, clipping de 0,2 y GAE con parámetro 0,95 [3]. Los agentes cooperativos reciben el retorno regional. S0 aprende con la recompensa de su propia tienda.")
h("III. Shaping potencial y Dyna")
p("Para cada tienda se calcula la cobertura como stock más tránsito, dividido por el pronóstico con mínimo uno y truncado a 14 días. El potencial Φ es el negativo de la suma de la varianza de cobertura dividida por 196 y el déficit medio de cobertura por debajo de tres días, dividido por tres. El potencial terminal vale cero.")
p("Puesto que F = γΦ(s′) − Φ(s), sumar los términos descontados de F produce −Φ(s₀) + γᵀΦ(sT). En terminal, G′ = G − Φ(s₀). Para cualquier acción, Q* con shaping difiere de Q* base en la misma constante del estado inicial. El máximo conserva la misma acción [4]. En un ejemplo de dos pasos, servir supera a acaparar por 0,99 frente a 0,891. El bono ingenuo eleva acaparar a 1,881 e invierte el orden. PBRS conserva la preferencia por servir.")
p("V1 incorpora PBRS, V2 incorpora Dyna y V3 combina ambos. N0 utiliza el bono ingenuo. Dyna aprende demanda a partir de observación, acción y tienda. Tras 256 transiciones reales, produce una transición sintética por paso elegible. El simulador calcula inventario y recompensa con sus ecuaciones contables.")
column()
h("IV. Comparación experimental")
p("Cada política se entrenó durante 300 episodios de 365 días con semillas 3, 4 y 5. Se congeló el checkpoint final. La incertidumbre entre semillas representa variación del aprendizaje sobre un mismo período comercial. Las bandas son desviaciones estándar muestrales. La evaluación económica excluye el shaping.")
rr=csvrows("campana/results_table.csv")
val={r["experiment"]:r for r in rr if r["split"]=="validation"}
test={r["experiment"]:r for r in rr if r["split"]=="test"}
t("Tabla I. Retorno medio y quiebres en el último año. H0 corresponde a la cobertura del 95 %.",[["Agente","Validación","Test","σ test","Quiebres"]]+[[k,num(val[k]["return_mean"],0),num(r["return_mean"],0),num(r["return_std"],0) if r["return_std"] else "—",num(100*float(r["stockout_mean"]),2)+" %"] for k,r in test.items()])
fig("campana/validation_returns.png","Fig. 1. Retorno en validación según pasos reales. Cada panel compara una familia o una ablación con tres semillas.")
column()
h("Eficiencia muestral y estabilidad")
p("El umbral descriptivo es 72 143,5, el 80 % del mayor retorno empírico de validación de la campaña. Se evalúa cada 3650 pasos. Las tres semillas de cada condición alcanzaron el umbral. La estabilidad se mide como dispersión entre episodios dentro de cada semilla y luego se promedia entre semillas.")
rows=csvrows("campana/learning_diagnostics.csv")
t("Tabla II. Mediana de pasos al umbral y desviación estándar de los últimos 30 episodios y 10 controles de validación.",[["Agente","Pasos","σ entren.","σ valid."]]+[[k,num(st.median([float(x["steps_to_80pct"]) for x in rows if x["experiment"]==k]),0),num(st.mean([float(x["tail_30_train_return_std"]) for x in rows if x["experiment"]==k]),0),num(st.mean([float(x["tail_10_validation_return_std"]) for x in rows if x["experiment"]==k]),0)] for k in test if k!="H0"])
fig("evidencia/g0_tabular.png","Fig. 2. Retorno descontado G₀ de Q-learning con γ = 0,99, media ±1σ. Se reprodujeron las tres semillas y coincidieron los 900 retornos episódicos sin descuento archivados.")
fig("campana/losses_summary.png","Fig. 3. Pérdidas de entrenamiento por familia. Cada serie conserva su propia escala y tres semillas.")
p("En los métodos profundos, los registros conservan el retorno acumulado sin descuento por episodio. La Fig. 4 muestra esa métrica. El G₀ descontado de estos métodos está disponible en test. La receta de reproducción registra ambos retornos en los entrenamientos futuros.")
column()
h("V. Referencia histórica inducida con M5")
p("Se fija el precio histórico y se elige por tienda la menor acción de reposición disponible que cubriría las ventas acumuladas de dos días en el porcentaje objetivo de ventanas de entrenamiento. Ese plazo comprende la decisión actual y el arribo del despacho. La cobertura deseada es un supuesto. La regla usa únicamente datos previos al período de evaluación.")
t("Tabla III. Efecto de la cobertura objetivo y comparación con PPO. Los objetivos se expresan en días para CA_1, CA_2 y CA_3.",[["Política","Objetivos","Ret. valid.","Ret. test","Quiebres¹"],["H0 90 %","7, 7, 3","54 777","83 354","11,88 %"],["H0 95 %","7, 7, 7","88 130","98 915","1,26 %"],["PPO","Aprendidos","86 092","96 590","1,33 %"]])
p("¹ Tasa sobre días-tienda con demanda positiva en test. El retorno de PPO se promedia entre semillas. En CA_3 el error absoluto medio de ventas en validación baja de 36,93 a 1,41 unidades diarias al pasar de tres a siete días de cobertura. Esto da respaldo empírico al escenario exigente, aunque las ventas observadas por sí solas admiten múltiples trayectorias de inventario.")
h("Criterio conjunto por semilla")
p("Se requieren beneficio superior a H0 y reducción relativa de quiebres de al menos 30 % frente a H0 y al egoísta S0 de la misma semilla. Para P0, elegida por retorno medio de validación, los resultados del escenario principal son los siguientes.")
t("Tabla IV. PPO en test, frente a H0 al 90 % y al control egoísta emparejado.",[["Semilla","Δ retorno H0","Red. H0","Red. S0","Cumple"],["3","+13 195","87,61 %","94,42 %","Sí"],["4","+12 723","89,38 %","42,86 %","Sí"],["5","+13 792","89,38 %","20,00 %","No"]])
p("Al 90 % cumplen 7 de 24 corridas de la matriz. Al 95 % cumplen 0 de 24. El escenario del 95 % reproduce la referencia original. Las dos coberturas formaron parte del reanálisis posterior de políticas congeladas. Este contraste utiliza un año de prueba ya examinado.")
column()
h("VI. Elasticidad, ablaciones y discusión")
p("Las políticas archivadas usaron e = 1,5 en las tres tiendas. Se estimó después, en entrenamiento, una regresión de Poisson por tienda con precio logarítmico, tendencia, estacionalidad, calendario y SNAP. Las elasticidades asociativas fueron 0,51, −1,73 y −1,49. Hubo entre 11 y 14 cambios de precio por tienda. La estimación común con tendencias por tienda pasó de −2,22 con 95 semanas a 2,20 con 173. La escasa variación de precio, los ceros y la disponibilidad desconocida limitan la identificación causal.")
p("Al sustituir 1,5 por los tres coeficientes con sus signos observados en validación, T0 congelado de semilla 3 pasó de 86 419,91 a 83 022,89 de retorno y de 4,57 % a 5,63 % de quiebres. H0 conserva su trayectoria porque el multiplicador siempre es uno. Esta sensibilidad se ejecutó con la política tabular recuperada. Las cifras de PPO pertenecen al escenario original de elasticidad.")
reentreno = csvrows("evidencia/tabular_reentrenado_calibrado.csv")
prueba_tabular = [r for r in reentreno if r["periodo"]=="test"]
p("También se reentrenó Q-learning con los coeficientes asociativos de su propia tienda y las mismas tres semillas y 300 episodios. En el último año el retorno medio fue "+num(st.mean(float(r["retorno"]) for r in prueba_tabular),2)+" y la tasa media de quiebres fue "+num(100*st.mean(float(r["quiebres"]) for r in prueba_tabular),2)+" %. Este reanálisis conserva el período ya examinado. Las políticas egoístas calibradas son necesarias para medir la meta conjunta en ese entorno.")
p("A igual número de interacciones reales, V1 obtuvo 348,7 unidades menos de retorno medio de test que V0, y V2 obtuvo 6591,5 menos. V3 recuperó 3737,4 frente a V2. Dyna redujo el MAE de demanda a 28,32 en V2 frente a 29,07 del pronóstico reciente, pero empeoró el RMSE. El menor error absoluto medio no aseguró mejor control. Una sola serie de producto y tres semillas limitan la generalización.")
fig("campana/training_returns.png","Fig. 4. Retorno episódico base sin descuento. Los episodios de entrenamiento abarcan ventanas distintas.")
h("Referencias")
p("[1] University of Nicosia, M5 dataset, Zenodo, DOI 10.5281/zenodo.10203108. [2] H. van Hasselt et al., Deep Reinforcement Learning with Double Q-learning, AAAI, 2016. [3] J. Schulman et al., Proximal Policy Optimization Algorithms, 2017. [4] A. Ng et al., Policy Invariance Under Reward Transformations, ICML, 1999.")


TITLE="Control colaborativo de precios e inventario con aprendizaje por refuerzo en Walmart M5"
AUTHORS="Leonardo Candio y Dimael Rivas"
def render():
    import os
    # Times is available on Windows. Liberation Serif covers Greek and operators on Linux.
    fonts=[(Path("C:/Windows/Fonts/times.ttf"),Path("C:/Windows/Fonts/timesbd.ttf")),
           (Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"))]
    family,bold="Times-Roman","Times-Bold"
    for regular,heavy in fonts:
        if regular.is_file() and heavy.is_file():
            pdfmetrics.registerFont(TTFont("Articulo",str(regular)))
            pdfmetrics.registerFont(TTFont("ArticuloB",str(heavy)))
            family,bold="Articulo","ArticuloB"
            break
    c=Canvas(str(D/"informe_p2_revision.pdf"),pagesize=letter)
    c.setTitle(TITLE);c.setAuthor(AUTHORS)
    margin=38;gap=18;cw=(612-2*margin-gap)/2
    def elements(col,scale):
        out=[]
        def para(text,size,font,leading,below):
            el=Paragraph(escape(text),ParagraphStyle("texto",fontName=font,fontSize=size*scale,leading=leading*scale,textColor=HexColor("#152626")))
            out.append((el,below*scale))
        for tag,body in col:
            if tag=="h":para(body,11,bold,13.2,4)
            if tag=="p":para(body,10,family,12.4,7)
            if tag=="t":
                caption,rr=body
                para(caption,8.6,bold,10.5,4)
                tab=Table(rr,colWidths=[cw/len(rr[0])]*len(rr[0]),hAlign="LEFT")
                tab.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),family),("FONTNAME",(0,0),(-1,0),bold),("FONTSIZE",(0,0),(-1,-1),8.4*scale),("LEADING",(0,0),(-1,-1),9*scale),("ALIGN",(1,0),(-1,-1),"RIGHT"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),2*scale),("BOTTOMPADDING",(0,0),(-1,-1),2*scale),("LINEBELOW",(0,0),(-1,0),.6,HexColor("#284743")),("LINEBELOW",(0,-1),(-1,-1),.4,HexColor("#70847d"))]))
                out.append((tab,6))
            if tag=="f":
                from reportlab.platypus import Image
                path,caption=body
                width,height=PILImage.open(D/path).size
                w=min(cw,268*scale*width/height)
                out.append((Image(str(D/path),width=w,height=w*height/width),3))
                para(caption,8.6,family,10.5,7)
        return out
    for page in range(3):
        if page==0:
            c.setFont(bold,19);c.drawCentredString(306,751,"Control colaborativo de precios e inventario con");c.drawCentredString(306,728,"aprendizaje por refuerzo en Walmart M5")
            c.setFont(family,12);c.drawCentredString(306,706,AUTHORS)
            c.setFont(family,10);c.drawCentredString(306,690,"Universidad de Ingeniería y Tecnología · DS5345 · Grupo 1")
        top=670 if page==0 else 744
        for k in range(2):
            x=margin+k*(cw+gap)
            for scale in [1,.97,.94,.91,.88,.85,.82]:
                els=elements(pages[2*page+k],scale)
                height=sum(el.wrap(cw,900)[1]+below for el,below in els)
                if height<=top-44:break
            else:raise RuntimeError(f"Columna desbordada {page},{k},{height}")
            y=top
            for el,below in els:
                width,hh=el.wrap(cw,900)
                el.drawOn(c,x+(cw-width)/2,y-hh)
                y-=hh+below
        c.setFont(family,9);c.drawCentredString(306,25,str(page+1));c.showPage()
    c.save()
def tex_escape(value):
    specials={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','≥':r'$\geq$','−':r'$-$','γ':r'$\gamma$','Φ':r'$\Phi$','σ':r'$\sigma$','Δ':r'$\Delta$','₀':r'$_0$','ₜ':r'$_t$','ᵀ':r'$^{T}$','′':r"$'$",'×':r'$\times$','±':r'$\pm$','¹':r'$^{1}$','–':'--','—':'---'}
    return ''.join(specials.get(c,c) for c in value)
def latex():
    lines=[r'\documentclass[conference]{IEEEtran}',r'\usepackage[T1]{fontenc}',r'\usepackage[utf8]{inputenc}',r'\usepackage[spanish,es-tabla]{babel}',r'\usepackage{graphicx,booktabs}',r'\title{'+TITLE+'}',r'\author{\IEEEauthorblockN{'+AUTHORS+r'}\IEEEauthorblockA{Universidad de Ingeniería y Tecnología\\ DS5345, Grupo 1}}',r'\begin{document}\maketitle']
    for col in pages:
        for i,(tag,body) in enumerate(col):
            if tag=="h" and body!="Resumen":
                lines.append(r'\section*{'+tex_escape(body)+'}')
            if tag=="p":
                val=tex_escape(body)
                lines.append((r'\begin{abstract}'+val+r'\end{abstract}' if i and col[i-1]==("h","Resumen") else val)+"\n")
            if tag=="f":
                path,caption=body
                lines.extend([r'\begin{figure}[!ht]\centering',r'\includegraphics[width=\columnwidth,height=.36\textheight,keepaspectratio]{'+path+'}',r'\caption{'+tex_escape(caption.split('. ',2)[-1] if caption.startswith('Fig.') else caption)+'}',r'\end{figure}'])
            if tag=="t":
                caption,rows=body
                lines.extend([r'\begin{table}[!ht]\centering\footnotesize',r'\caption{'+tex_escape(caption.split('. ',1)[-1])+'}',r'\begin{tabular}{l'+'r'*(len(rows[0])-1)+r'}\toprule'])
                for j,row in enumerate(rows):
                    lines.append(' & '.join(map(tex_escape,row))+r'\\'+(r'\midrule' if j==0 else ''))
                lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (D/"informe_p2_revision.tex").write_text('\n'.join(lines+[r'\end{document}']),encoding='utf8')
if __name__=="__main__":
    render()
    latex()

