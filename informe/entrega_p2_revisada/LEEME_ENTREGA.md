# Entrega y revisión final — P2 Vía B

Autores del informe: Leonardo Candio y Dimael Rivas (Grupo 1).

## Archivos de uso inmediato

- informe_p2.pdf: informe IEEE original de 4 páginas, conservado sin modificaciones.
- presentacion_p2.pptx: 15 diapositivas editables con notas del orador.
- guion_20_minutos.md: guion distribuido en 20 minutos, incluida la demo.
- demo_p2.html: demo sin conexión, con controles para eventos relevantes.
- preguntas_defensa.md: respuestas a las principales objeciones metodológicas.
- resultados_originales/: paquete recibido, con resultados, figuras, notebook, LaTeX, protocolo y huellas originales.
- diagnostico_politicas.csv: diagnóstico adicional de acciones y racionamiento.
- revision_final.json: alcance de esta revisión.

## Qué se verificó

El ZIP original pasó la comprobación CRC y las 57 huellas del manifiesto coinciden. Se recalcularon las métricas de las 25 trayectorias de test y se comprobó la contabilidad de recompensas, la conservación de inventario y la llegada al día siguiente de los despachos: 9,125 transiciones. Las medias, desviaciones muestrales, comparaciones y comprobaciones de metas coinciden. La reevaluación CPU registrada presenta diferencia cero en retorno de validación en los 25 casos.

Se inspeccionaron visualmente las cuatro páginas del PDF. Las 15 diapositivas se revisaron visualmente y su paquete se validó estructuralmente; no se comprobó su apertura en la aplicación PowerPoint. La demo se probó en Edge sin errores JavaScript: tres tiendas, controles y saltos a eventos correctos. Las referencias de archivos del notebook están presentes; no se ejecutó completo en esta revisión porque el entorno local carece de IPython.

Esta revisión no entrenó políticas ni volvió a evaluar el test. Conserva la selección realizada mediante validación. El gráfico editable contiene medias redondeadas a cuatro decimales para compatibilidad con Excel; los CSV originales conservan toda su precisión.

## Interpretación que debe mantenerse

H0 obtiene retorno 98,915.49. Q-learning es el mejor método aprendido por retorno medio de test: 98,649.14, aproximadamente 0.27% inferior a H0. PPO había sido elegido por validación y se conserva para la demo principal; en test obtiene 96,590.33, aproximadamente 2.35% inferior a H0. PPO presenta el mayor fill rate y la menor desviación del retorno entre los métodos aprendidos.

Ninguna de las 24 corridas aprendidas cumple conjuntamente la reducción de quiebres de al menos 30% frente a H0 y al egoísta emparejado, junto con retorno superior a H0. Un resultado negativo es un resultado experimental válido; no acredita haber alcanzado la meta del enunciado.

Son tres semillas de optimización sobre un mismo año de test. No son tres años independientes ni permiten afirmar significancia por sí solas. Las diferencias de medias de ablaciones deben acompañarse de su heterogeneidad entre semillas.

La elasticidad 1.5 fue supuesta, no estimada. M5 no proporciona los inventarios ni costes históricos requeridos para observar directamente quiebres o beneficio neto. H0 es una reconstrucción dentro del simulador. Las conclusiones se limitan a ese modelo y al SKU utilizado.

El modelo de demanda reduce MAE, pero empeora RMSE frente al pronóstico reciente ajustado. Los errores se miden en las trayectorias de cada política: no constituyen una comparación causal sobre una distribución común. El contraejemplo de reward hacking es sintético; no prueba que la política N0 haya aprendido ese comportamiento en M5.

Las curvas de entrenamiento registran retorno sin descuento. No deben rotularse como G0 descontado. La evaluación sí incluye retorno descontado.

## Cómo mostrar la demo (2 minutos)

1. Abrir demo_p2.html. Mantener PPO semilla 3, la trayectoria principal.
2. Pulsar Primer quiebre: día 27. Explicar inventario disponible, ventas y unidades perdidas.
3. Señalar que PPO mantiene los mismos índices de acción: precio ×0.95 en CA_1 y ×1.10 en CA_2/CA_3, con cobertura objetivo de 7 días. Los pedidos cambian con el estado. No afirmar adaptación diaria del multiplicador de precio en esta trayectoria.
4. Cambiar a DDQN semilla 3 y pulsar Primer racionamiento: día 63. Los pedidos suman 371, los despachos 317 y el inventario regional final es cero. Los despachos llegan al día siguiente.
5. Aclarar que este segundo caso se añadió después del test para ilustrar el mecanismo; no reemplaza la selección por validación ni demuestra superioridad.

## Reproducción y preservación

Código fijado: https://github.com/artrivas/RL-Project/tree/4644eb088135060cffba499490b75986f701d29d. El protocolo y las huellas del paquete identifican las fuentes y checkpoints auditados.

Este ZIP no es un respaldo de los checkpoints de entrenamiento ni de los CSV originales de M5. Se debe conservar el volumen persistente del Pod hasta respaldar esos archivos. No es necesario mantener entrenamiento en ejecución.

No se validó la reproducción completa mediante Docker en esta revisión. La configuración Compose revisada monta src y notebooks, pero no todos los scripts/directorios que utiliza la receta final. Por ello no se debe afirmar que esa receta fue probada dentro del contenedor. El notebook suministrado analiza resultados ya guardados; no sustituye la reproducción del entrenamiento.

El informe PDF y su LaTeX permanecen originales. Los hallazgos adicionales de esta revisión figuran aquí, en las diapositivas y en el diagnóstico. La frase de cierre del PDF que pide revisar y preparar la sustentación pertenece al informe original.

## Último paso

Confirmar los nombres de los autores, ensayar el guion de 20 minutos y abrir los archivos en el equipo que se usará para sustentar. Entregar el PDF, la presentación y los complementos que exija Canvas. Antes de afirmar cumplimiento total del enunciado, deben reconocerse expresamente la meta no alcanzada, la elasticidad no estimada y la reproducción Docker no validada.
