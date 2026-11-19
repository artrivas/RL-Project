"""Figura de G0 de la nueva reproducción completa, por condición y semilla."""
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

def main(suite,dest):
    fig,axes=plt.subplots(4,2,figsize=(10,10),sharex=True,constrained_layout=True)
    for ax,method in zip(axes.flat,("T0","V0","P0","S0","V1","V2","V3","N0")):
        series=[]
        for seed in (3,4,5):
            records=json.loads((suite/f"{method}_seed{seed}"/"training.json").read_text())
            series.append([row["discounted_base_return"] for row in records])
        values=np.array(series)
        if values.shape!=(3,300):raise ValueError(f"{method}: historial incompleto")
        x=np.arange(1,301)
        mean=values.mean(axis=0)
        sd=values.std(axis=0,ddof=1)
        ax.plot(x,mean,linewidth=.9)
        ax.fill_between(x,mean-sd,mean+sd,alpha=.25)
        ax.set_title(method)
    fig.supxlabel("Episodio de entrenamiento")
    fig.supylabel("Retorno regional descontado")
    fig.savefig(dest/"g0_entrenamiento.png",dpi=180)
if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
