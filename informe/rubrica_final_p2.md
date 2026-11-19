# Autoevaluación final de P2 · Vía B · Walmart M5

**Estado para la revisión docente, 18 de noviembre de 2026.** El cierre de Docker terminó con código 0. Los tres pilares del tronco común tienen evidencia. La meta comercial queda parcialmente alcanzada porque el resultado conjunto depende de la semilla y de cómo se reconstruye la política histórica. El enunciado presenta una lista de verificación sin ponderaciones. Los estados de esta tabla expresan cumplimiento documentado y no equivalen a una nota.

## Criterios del enunciado

| Criterio | Estado | Evidencia y alcance |
|---|---|---|
| Pilar 1. Familias de algoritmos | Completo | Q-learning tabular, DDQN (valor) y PPO con GAE (actor-crítico) se comparan en el entorno de las tres tiendas. |
| Pilar 2. PBRS y aprendizaje basado en modelo | Completo | El informe deriva la invariancia de la política óptima con la función potencial, contrasta el bono ingenuo y evalúa Dyna con ablaciones. La eficiencia se mide por variante. |
| Pilar 3. Escalera, curvas y métricas | Completo | La escalera incluye el agente tabular y la variante con Dyna. La campaña calibrada registra 7 200 retornos descontados de entrenamiento (8 condiciones × 3 semillas × 300 episodios) con media y banda de ±1 desviación estándar. La primera campaña aporta curvas de pérdidas y diagnósticos de retorno final, pasos hasta el 80 % y estabilidad. |
| Vía B. Datos, entorno y coordinación | Completo | Ventas, precios y calendario M5 para CA_1, CA_2 y CA_3. Cada agente decide precio y pedido con inventario regional finito. La campaña final usa coeficientes de respuesta al precio estimados por tienda con datos de entrenamiento. |
| Vía B. Superar el beneficio histórico | Alcanzado por PPO frente a H0 | En el último año, las tres semillas de PPO superan el beneficio simulado de ambas referencias históricas inducidas. H0 usa precios históricos y una regla de reposición estimada a partir de ventas M5. El beneficio depende del modelo de costes e inventario. |
| Vía B. Reducir los quiebres ≥30 % frente a H0 y S0, además de superar el beneficio | Parcial | Una de las tres semillas de PPO cumple todo con H0 al 90 %. Ninguna cumple todo con H0 al 95 %. S0 es el agente egoísta de la misma semilla. El desglose figura abajo. |
| Reproducibilidad Docker | Ejecución local completa | El comando de cierre concluyó el entrenamiento y la generación de resultados, figura, PDF y ZIP. El acceso del docente a la versión final en GitHub sigue sin verificar. |
| Informe técnico | Completo | PDF en español, compilado desde LaTeX, de 4 páginas en formato IEEE (máximo 8). Incluye modelado, algoritmos, PBRS y Dyna, resultados por semilla, análisis de Vía B y discusión. |
| Sustentación y entrega | Material preparado | Presentación final de 17 diapositivas, guion de 20 minutos y demo local de inferencia tabular. Quedan la constancia de entrega en Canvas y la sustentación grupal con preguntas individuales. |

El equipo eligió la Vía B. El criterio de RoboCup corresponde a la Vía A.

## Resultado que se presenta al profesor

Último año de la **campaña calibrada**. La diferencia de beneficio es el beneficio de PPO menos el de H0. Una reducción negativa indica más quiebres que la referencia. Los niveles 90 % y 95 % identifican dos reglas de reposición de H0.

| Semilla PPO | Diferencia de beneficio vs. H0 90 % | Diferencia de beneficio vs. H0 95 % | Reducción de quiebres vs. H0 90 % | Reducción de quiebres vs. H0 95 % | Reducción de quiebres vs. S0 | Meta conjunta |
|---|---:|---:|---:|---:|---:|---|
| 3 | +18 148,59 | +2 586,85 | 87,61 % | −16,67 % | 33,33 % | Cumple con H0 90 % |
| 4 | +26 875,27 | +11 313,52 | 86,73 % | −25,00 % | 16,67 % | Ninguna referencia |
| 5 | +17 281,86 | +1 720,11 | 88,50 % | −8,33 % | 18,75 % | Ninguna referencia |

En las 24 corridas entrenadas, solo PPO con semilla 3 alcanza la meta conjunta ante H0 al 90 %. Ante H0 al 95 % se obtienen 0/24. El resultado anterior de PPO de 2/3 ante H0 al 90 % corresponde a la campaña con elasticidad fija de 1,5. Ambas campañas usaron el mismo año de prueba, que ya se había examinado al realizar el reanálisis. La segunda evaluación aporta sensibilidad al supuesto de demanda.

## Archivos de respaldo

- [Enunciado oficial, criterios de Vía B y lista de verificación](../enunciado_proyecto_p2_alumno.pdf)
- [Informe final](revision_docente/salidas_cierre/informe_final.pdf) y [presentación final](revision_docente/salidas_cierre/presentacion_final.pptx)
- [Metas por corrida y referencia](revision_docente/salidas_cierre/metas_calibradas.csv) y [retorno por episodio](revision_docente/salidas_cierre/g0_calibrado.csv)
- [Diagnósticos de eficiencia y estabilidad](revision_docente/campana/learning_diagnostics.csv) y [ablaciones](revision_docente/campana/paired_ablations.csv)
- [Paquete de entrega](revision_docente/salidas_cierre/P2_final.zip)
