"""Diagnóstico exploratorio del soporte de precios usando solo las 173 semanas de entrenamiento."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from calibrar_m5 import HERE, ITEM, STORES, HASHES, load_m5


def compare(y, p, trend=True, price=True):
    # Perfil de Poisson con interceptos por semana: condicionar en el total regional.
    weeks = np.arange(len(y)) / 52
    store = np.broadcast_to(np.eye(3)[:, :2], (len(y), 3, 2))
    terms = [store]
    if trend:
        terms.append(store * weeks[:, None, None])
    if price:
        terms.append(np.log(p)[:, :, None])
    x = np.concatenate(terms, axis=2)
    x -= x.mean(axis=1, keepdims=True)
    n = y.sum(axis=1)
    active = n>0
    if np.linalg.matrix_rank(x[active].reshape(-1,x.shape[2])) < x.shape[2]:
        return None
    def calculate(b):
        eta = x @ b
        eta -= eta.max(axis=1,keepdims=True)
        logprob = eta - np.log(np.exp(eta).sum(axis=1,keepdims=True))
        prob = np.exp(logprob)
        resid = x - (x*prob[:,:,None]).sum(axis=1,keepdims=True)
        grad = np.einsum('wi,wik->k',n[:,None]*prob-y,x)
        hess = np.einsum('wi,wik,wil->kl',n[:,None]*prob,resid,resid)
        return -(y*logprob).sum(), grad, hess
    b = np.zeros(x.shape[2])
    for _ in range(80):
        loss, grad, hess = calculate(b)
        if np.abs(grad).max() < 1e-8*n.sum():
            return b
        delta = np.linalg.solve(hess,grad)
        for k in range(25):
            candidate = b-delta/2**k
            if calculate(candidate)[0] <= loss:
                b = candidate
                break
        else:
            return None
    return None


def diagnose(data):
    n = 1211
    sales = np.array(data.sales)[:, :n].T
    daily_prices = np.array(data.prices)[:, :n].T
    assert len({d['week'] for d in data.calendar[:n]}) == n//7
    assert all(len({d['week'] for d in data.calendar[t:t+7]})==1 for t in range(0,n,7))
    y = sales.reshape(-1,7,3).sum(axis=1)
    pp = daily_prices.reshape(-1,7,3)
    assert (pp == pp[:,:1,:]).all()
    p=pp[:,0,:]
    relative=np.round(np.log(p)-np.log(p[:,:1]),9)
    zero_runs={}
    for i,s in enumerate(STORES):
        runs=[list(g) for zero,g in itertools.groupby(enumerate(sales[:,i],1),lambda z:z[1]==0) if zero]
        a=max(runs,key=len)
        zero_runs[s]={'dias_sin_ventas':int((sales[:,i]==0).sum()),'racha_mas_larga':[a[0][0],a[-1][0]]}
    fits=[]
    for end in (95,121,147,173):
        for trend in (False,True):
            b=compare(y[:end],p[:end],trend)
            fits.append({'semanas':end,'tendencia_por_tienda':trend,
                         'elasticidad_comun_asociativa':None if b is None else float(-b[-1])})
    return {'sku':ITEM,'dias':[1,n],'rachas':zero_runs,'semanas_con_ventas':int((y.sum(1)>0).sum()),
            'semanas_con_distinto_precio_y_ventas':int(((np.ptp(p,axis=1)>0)&(y.sum(1)>0)).sum()),
            'cambios_en_precios_relativos':int(np.any(np.diff(relative,axis=0)!=0,axis=1).sum()),
            'ajustes':fits}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--datos',type=Path,default=HERE.parent/'notebooks/data/m5/raw')
    cli.add_argument('--salida',type=Path,default=HERE/'diagnostico_precios_m5.json')
    args=cli.parse_args()
    data=load_m5(args.datos,ITEM)
    if data.manifest['sha256']!=HASHES:
        raise ValueError('Las huellas difieren de la campaña M5')
    report=diagnose(data)
    with args.salida.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report))

