# Proyecto P2 · Revisión docente del 19 de noviembre de 2026

**Leonardo Candio y Dimael Rivas · UTEC · DS5345 · Vía B**

Abrir primero **informe_p2_revision.pdf**. La presentación editable tiene 16 diapositivas y el guion distribuye la sustentación en 20 minutos. Para ejecutar la política, abrir **demo_en_vivo.html** directamente en un navegador moderno. El archivo funciona sin conexión. Elegir el período y avanzar uno o siete días. La política tabular decide en cada paso y el centro reparte el inventario.

## Resultado que se somete a revisión

Con los precios históricos de M5, la reposición se induce usando ventas de entrenamiento. La regla al 90 % asigna objetivos de siete, siete y tres días a CA_1, CA_2 y CA_3. La alternativa al 95 % asigna siete días a las tres. En el último año PPO cumple simultáneamente el objetivo de beneficio y los dos objetivos de reducción de quiebres en dos de tres semillas con la primera regla y en ninguna con la segunda. La segunda regla reproduce más de cerca las ventas observadas de CA_3 en validación. El reanálisis de las referencias se hizo después de abrir el año de prueba original.

Para la revisión del docente interesa especialmente saber si la justificación de las dos coberturas, incluida la sensibilidad y el error de ventas por tienda, responde a la indicación de inducir la política histórica con M5.

La revisión incluye además tres entrenamientos nuevos de Q-learning con los coeficientes por tienda. Los archivos `tabular_reentrenado_calibrado.csv` y `g0_tabular_reentrenado_calibrado.csv` separan esta campaña de la primera.

## Evidencia y procedencia

- `evidencia/meta_por_corrida.csv` contiene los 48 contrastes para ocho métodos, tres semillas y dos referencias. Cada corrida se compara también con S0 de su propia semilla.
- `evidencia/calibracion_m5.json` y `evidencia/diagnostico_precios_m5.json` identifican la fuente, las tiendas, los coeficientes y la variación temporal. La demanda de la campaña usa elasticidad 1,5. La sensibilidad en validación usa los tres coeficientes asociativos con sus signos originales en un cálculo explícito de demanda.
- `evidencia/g0_tabular.csv` recoge 900 episodios reproducidos. Las sumas sin descuento coinciden con las archivadas y los tres resultados de test coinciden en retorno y tasa de quiebres. La curva descontada de las familias profundas sigue pendiente de una reproducción completa.
- `demo/t0_semilla3.npz` contiene las tablas Q reproducidas. La huella binaria del archivo difiere de la del checkpoint original; las comparaciones verifican el comportamiento en los recorridos evaluados. En el navegador se consulta la acción greedy exportada para cada estado posible de cada tienda. Se contrastó el motor con los 365 días archivados, incluidos pedidos, despachos, ventas, faltantes y retorno diario. El archivo de PPO en `campana/test` contiene la trayectoria para el análisis.
- `campana/` conserva intactos los resultados y el informe IEEE compilado de la campaña original. Sus listas de pendientes reflejan el momento del primer cierre. El informe de revisión integra la evidencia posterior. Su PDF se maquetó a dos columnas con la misma fuente de contenido que el archivo LaTeX IEEE incluido.
- `codigo/base_4644eb0/` contiene las fuentes utilizadas en el protocolo original. El código archivado se identifica en [GitHub](https://github.com/artrivas/RL-Project/tree/4644eb088135060cffba499490b75986f701d29d). Los archivos nuevos de la revisión se incluyen en este paquete.
- `analisis_revision.ipynb` incluye las salidas de las tres celdas de análisis ejecutadas en Python y puede abrirse desde Jupyter. Los datos del producto para la demostración están dentro de la entrega.

## Cierre completo en Ubuntu

Desde esta carpeta, ejecutar `bash cerrar_hoy.sh`. Se construye una imagen CPU y se compila primero este informe. El contenedor aprovecha los CSV M5 del repositorio si existen. Después ejecuta las pruebas y entrena ocho condiciones con tres semillas, 300 episodios y las asociaciones precio-ventas por tienda estimadas en los días 1 a 1211. Esta campaña adicional conserva los signos observados y se presenta como sensibilidad ante la especificación de demanda. El año de prueba ya era conocido. Cada episodio guarda el retorno base y el descontado. Se auditan los checkpoints, se evalúan las políticas y se contrasta la meta frente a H0 al 90 %, H0 al 95 % y S0 de la misma semilla.

El primer PDF compilado queda en `salidas_cierre/lectura/informe_p2_revision.pdf`. Después de terminar las 24 corridas, la fuente y el PDF finales incorporan la figura con 7200 retornos y los resultados medidos. La entrega actualizada queda en `salidas_cierre/P2_final.zip`. La terminal muestra cada corrida que termina y cada una guarda su log. Si se interrumpe, el mismo comando recupera los checkpoints profundos parciales. Tres trabajadores por defecto. Para ajustar el paralelismo se puede usar, por ejemplo, `M5_WORKERS=2 bash cerrar_hoy.sh`.

## Reproducción

Desde la carpeta extraída que contiene este documento, el siguiente comando vuelve a calcular las comparaciones y la sensibilidad usando los archivos incluidos.

```bash
uv run --python 3.11 --with-requirements codigo/requirements.txt analizar_revision.py
```

Para reproducir las tres semillas tabulares desde los CSV originales, indicar la carpeta con los tres CSV M5 y una carpeta nueva de salida. El consumidor puede verificar las huellas en la calibración.

```bash
uv run --python 3.11 --with-requirements codigo/requirements.txt codigo/reproducir_tabular_revision.py --datos /ruta/a/m5/raw --archivo campana --salida reproduccion_tabular
```

Para construir la imagen CPU y ejecutar la campaña completa con 300 episodios por semilla, el comando siguiente descarga y valida los datos, ejecuta las pruebas, entrena, evalúa y escribe las figuras. Cada ejecución crea una carpeta distinta en `codigo/salidas_docker/`. La descarga inicial necesita internet. La imagen genera su propio commit local para identificar la copia instrumentada. Su registro de G₀ abarca las ocho condiciones. La compilación de la imagen y la campaña Docker completa siguen sin verificarse en un motor Docker.

```bash
docker compose -f codigo/compose.revision.yaml up --build --abort-on-container-exit --exit-code-from reproduccion
```

Para compilar el artículo con IEEEtran desde una instalación LaTeX con soporte español, ejecutar dos veces desde esta carpeta.

```bash
pdflatex -interaction=nonstopmode -halt-on-error informe_p2_revision.tex
pdflatex -interaction=nonstopmode -halt-on-error informe_p2_revision.tex
```

La versión de lectura y el documento editable se generan con `generar_informe.py` a partir de las mismas tablas. La fuente LaTeX revisada queda disponible para compilación independiente. El manifiesto SHA-256 cubre los archivos distribuidos.
