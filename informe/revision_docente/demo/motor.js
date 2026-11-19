// Inferencia greedy de las tres tablas Q. El estado cambia con cada despacho y venta.
function iniciar(datos, politica, cfg, inicio=1577, fin=1941) {
  const tiendas = [0,1,2];
  let dia = inicio, regional = cfg.initial_regional_stock;
  let existencias = tiendas.map(()=>cfg.initial_store_stock), transito = [0,0,0];
  const historial = datos.sales.map(serie=>serie.slice(inicio-1-cfg.history_days,inicio-1));
  const suma = valores=>valores.reduce((a,b)=>a+b,0);
  const indice = (x, cortes)=>cortes.filter(c=>x>=c).length;
  return function avanzar() {
    if (dia>fin) return null;
    regional = Math.min(cfg.regional_capacity,regional+cfg.regional_supply);
    existencias = existencias.map((x,i)=>x+transito[i]);
    const anterior = [...existencias], disponible = regional;
    const pronostico = historial.map(xs=>suma(xs)/xs.length);
    const acciones = tiendas.map(i=>{
      const f=Math.max(1,pronostico[i]);
      const a=indice(existencias[i]/f,[.01,1,3,7]);
      const b=indice(pronostico[i],[1,5,10]);
      const c=indice(regional/(3*f),[1,3,7]);
      const e=indice(fin-dia+1,[7,30,90]);
      const estado=((((a*4+b)*4+c)*7+datos.calendar[dia-1].weekday)*4+e);
      return politica[i][estado];
    });
    const multiplicador=acciones.map(a=>cfg.price_multipliers[Math.floor(a/cfg.target_days.length)]);
    const cobertura=acciones.map(a=>cfg.target_days[a%cfg.target_days.length]);
    const pedidos=tiendas.map(i=>Math.min(cfg.store_capacity-existencias[i],
      Math.max(0,Math.ceil(cobertura[i]*pronostico[i])-existencias[i])));
    const total=suma(pedidos);
    transito=[...pedidos];
    if(total>regional) {
      transito=pedidos.map(p=>Math.floor(regional*p/total));
      const orden=[...tiendas].sort((i,j)=>(regional*pedidos[j])%total-(regional*pedidos[i])%total || i-j);
      for(const i of orden.slice(0,regional-suma(transito)))transito[i]++;
    }
    regional-=suma(transito);
    const demanda=tiendas.map(i=>Math.floor(datos.sales[i][dia-1]*multiplicador[i]**(-cfg.elasticities[i])+.5));
    const ventas=tiendas.map(i=>Math.min(existencias[i],demanda[i]));
    const perdidas=demanda.map((d,i)=>d-ventas[i]);
    existencias=existencias.map((x,i)=>x-ventas[i]);
    const retorno=suma(tiendas.map(i=>ventas[i]*datos.prices[i][dia-1]*multiplicador[i]
      -cfg.holding_cost*existencias[i]-cfg.shortage_cost*perdidas[i]));
    for(const i of tiendas){historial[i].shift();historial[i].push(ventas[i]);}
    return {dia:dia++,fecha:datos.calendar[dia-2].date,disponible,regional,anterior,pronostico,
      acciones,multiplicador,cobertura,pedidos,despachos:[...transito],ventas,perdidas,retorno};
  };
}
if(typeof module!=="undefined")module.exports={iniciar};
