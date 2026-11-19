# Cierre M5: un batch, sin entrenamiento adicional

## Estado y alcance

El usuario ha compartido un resumen completo de la matriz de desarrollo:
H0 y ocho condiciones por semillas 3, 4 y 5; cada política aprendida tiene
109500 pasos. Los 21 manifiestos profundos registran CUDA y PyTorch 2.8.0+cu129.
El cierre verifica estos artefactos directamente antes de evaluar.

El plan inicial proponía semillas nuevas 0, 1 y 2 para confirmación. Se registra
la enmienda de alcance: terminar con las tres semillas de desarrollo disponibles
y evaluar sus políticas finales congeladas en el año de test. No se describe esta
campaña como confirmación independiente. No se elige una semilla por test.

## Un comando dentro del Pod

Desde el checkout fijado de la rama feat/p2-via-b:

```bash
bash scripts/launch_m5_final.sh /workspace/rl-project-validation/m5-final-01
```

Argumentos opcionales, en orden: salida, batch de origen, datos crudos.
Defaults de entrada: /workspace/rl-project-validation/m5-night-01 y
/workspace/data/m5/raw.

El launcher prepara sólo dependencias ausentes (CPU PyTorch >=2.6, NumPy,
Matplotlib, pypdf y LaTeX/IEEEtran), ejecuta los tests y lanza el cierre. LaTeX
se instala por apt si falta; esto requiere root, como en el Pod actual.
No requiere GPU, no vuelve a entrenar y no modifica artefactos de origen.
Usa un hilo de cómputo y evalúa secuencialmente para limitar RAM.
Un lock impide dos cierres concurrentes en la misma salida.

Para desconectarse durante la instalación/evaluación, ejecutar el mismo comando
con nohup y redirigir el log fuera de la carpeta de entrega. Repetir exactamente
el mismo cierre reusa evaluaciones terminadas sólo si coincide todo el protocolo
congelado (código, datos, pesos y versión de evaluación).

## Qué produce

- Auditoría de 25 corridas y hashes de datos, código y checkpoints.
- Protocolo escrito antes del primer rollout de test.
- Evaluación greedy CPU en validación, para registrar diferencias numéricas
  respecto a CUDA, y una evaluación en test por política (d1577-d1941).
- Métricas base, G0 descontado, contabilidad y trazas por tienda.
- Medias y desviación muestral, contrastes emparejados y metas frente a H0/S0.
- Curvas de entrenamiento, validación y pérdidas con bandas de dispersión.
- Error de demanda de los modelos congelados frente a un pronóstico sencillo.
- Eficiencia al 80% del máximo empírico de validación, definida post hoc:
  análisis descriptivo, nunca estimación de un óptimo o selección con test.
- Estabilidad de las colas de 30 episodios de entrenamiento y 10 validaciones.
- Contraejemplo sintético de hacking frente a PBRS, separado de resultados M5.
- Informe IEEE LaTeX/PDF (comprobación de máximo 8 páginas), notebook,
  demo HTML de PPO semilla 3 y guion de sustentación de 20 minutos.
- ZIP de la entrega y SHA256SUMS.json; sin datos crudos, pesos ni credenciales.

Los autores se heredan del informe P1: Leonardo Candio y Dimael Rivas.
La revisión visual del PDF, la confirmación de autores y la sustentación
presencial siguen pendientes. No se envía a Canvas ni se fusiona main.

## Interpretación y brechas del enunciado

La elasticidad de 1.5 es supuesta, no estimada. Un solo SKU y el proxy histórico
no acreditan beneficios ni quiebres reales de Walmart. Una ejecución correcta
no equivale a cumplir el 30%; target_checks.csv lo calcula sin ocultar fallos.
Los retornos finales no prueban convergencia asintótica. El G0 descontado no
se registró durante entrenamiento; las curvas se etiquetan sin descuento.
El ejemplo de hacking prueba un mecanismo en un MDP construido, no que N0
haya explotado ese mecanismo empíricamente en M5.

Los tests locales de cierre no necesitan PyTorch, salvo una comparación
de inferencia que se ejecuta obligatoriamente en el Pod. La generación de
figuras/PDF se comprueba en el Pod si faltan esas dependencias locales.

## Reproducción desde cero (costosa; no necesaria para cerrar)

Con los datos M5 auditados y dependencias de docker/requirements.p2.txt:

```bash
bash scripts/reproduce_m5.sh /workspace/data/m5/raw /workspace/reproduction-new cpu
```

Ese comando sí reentrena las 24 políticas y después genera el cierre.
La opción cuda requiere PyTorch/CUDA compatible. La reproducción numérica
CPU/GPU puede diferir; se registra hardware y no se promete identidad entre ambos.
Los archivos docker/m5_final_assumptions.json y m5_final_settings.json congelan
las configuraciones efectivas. El contenedor P2 existente instala las dependencias
de entrenamiento; los scripts se ejecutan desde un checkout Git montado.

El Pod no se detiene ni se elimina automáticamente.
