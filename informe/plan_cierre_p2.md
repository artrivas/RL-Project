# Plan final de entrega de P2 · Vía B, Walmart M5

**Estado después del cierre de Docker.** Consulta la [rúbrica actualizada](rubrica_final_p2.md) para los resultados de la campaña calibrada y la situación de cada criterio. Este plan conserva el punto de partida anterior al cierre.

Actualizado el 18 de noviembre de 2026. Sustentación el viernes 20 en horario de clase, con entrega previa en Canvas. El equipo debe comprobar en Canvas la hora exacta. Los hitos del miércoles y jueves son metas internas propuestas.

## Punto de partida

La campaña conserva 25 evaluaciones finales, ocho condiciones entrenables con tres semillas y una referencia determinista. El informe IEEE en español tiene cuatro páginas. Ya existen la comparación tabular–DDQN–PPO, la demostración de invariancia de PBRS, el experimento de bono ingenuo, Dyna y sus ablaciones. El material de defensa ya incluye 15 diapositivas, un guion y un visor de trayectorias.

La reconstrucción de la reposición con los primeros 1211 días de M5 ya terminó. En el último año, PPO cumple las tres condiciones conjuntas en dos de tres semillas con H0 al 90 % y en cero de tres con H0 al 95 %. Este último escenario reprodujo mejor las ventas de CA_3 en validación. Los modelos publicados usaron elasticidad 1,5. La estimación posterior por tienda y el cambio de signo en las ventanas temporales están documentados. El año de prueba ya se examinó. Estos hechos deben aparecer de forma coherente en todos los entregables.

## Miércoles 18 · Resolver las dependencias que pueden exigir cómputo

| Orden | Trabajo | Evidencia para cerrar |
|---|---|---|
| 1. Recuperar los archivos de entrenamiento | Localizar en el Pod o en copias existentes los checkpoints finales, configuraciones, registros por episodio y posibles recompensas diarias. Priorizar PPO semilla 3 para la demo y conservar los manifiestos de todas las corridas. | Inventario con rutas y hashes contrastados con el protocolo. Un checkpoint PPO carga en CPU y reproduce la evaluación guardada. |
| 2. Resolver el retorno episódico | Buscar recompensas base ordenadas por paso e identificadores de episodio. Si existen, calcular G₀ = Σ γᵗRₜ con γ = 0,99. Comprobar la fórmula contra el G₀ de test guardado y la suma sin descuento contra el registro de entrenamiento. Si solo existen totales, instrumentar el entrenamiento para guardar ambos retornos desde el primer episodio. | Curva de G₀ de entrenamiento por condición, media ±1 desviación estándar entre las tres semillas. Si se requieren corridas nuevas, conservar su procedencia y asociarles sus propias pérdidas. |
| 3. Habilitar la reproducción en Docker | Partir del código de la campaña 4644eb0. Corregir los montajes de scripts, configuración, datos y salida. Fijar las dependencias y documentar un comando desde un clon limpio que entrene, evalúe y genere las figuras. | Construcción limpia y prueba corta de extremo a extremo que produzca archivos reales. Medir el tiempo por episodio para decidir si cabe la reproducción completa antes de la entrega. Registrar por separado su ejecución y resultado. |
| 4. Fijar cómo se usará la elasticidad | Conservar como referencia la campaña de elasticidad 1,5. Implementar la carga de un coeficiente por tienda y comprobar que al precio histórico se reproducen las unidades de demanda base. Como primera prueba de sensibilidad, aplicar las tres estimaciones de entrenamiento con su signo original a las políticas congeladas, H0 y S0 en validación. Documentar precios disponibles, estabilidad temporal y efecto sobre demanda, quiebres y retorno. | Configuración versionada, tabla de coeficientes y comparación bajo condiciones comunes. Identificar las cifras como sensibilidad de políticas congeladas. El uso de estas estimaciones en una campaña nueva requiere justificar su comportamiento con datos de entrenamiento y validación. |

Los pasos 2 y 3 pueden avanzar en paralelo. En cuanto se recupere el checkpoint de PPO, se puede preparar la demo. La redacción de las secciones ya verificadas puede comenzar el mismo miércoles.

### Decisión sobre una única campaña adicional

- Si las trazas permiten recuperar G₀, reconstruir las curvas de la campaña publicada. Si faltan las recompensas por paso, usar el registro corregido en una reproducción de las ocho condiciones con semillas 3, 4 y 5. Verificar que las métricas de las corridas reproducidas sean comparables e identificar cualquier diferencia.
- Si la calibración justifica adoptar un entorno de demanda revisado, fijar primero estimador, parámetros, semillas, presupuesto y checkpoint final. Usar esa misma campaña para recoger G₀ y evaluar todos los métodos, incluida la referencia egoísta. Esto evita duplicar entrenamientos. Las nuevas tablas y curvas deben identificar el entorno y la versión.
- Mantener arquitectura y 300 episodios por corrida como punto de partida. La justificación para repetir es recuperar una métrica exigida o evaluar un entorno corregido. El resultado del último año queda fuera de la selección de umbral, modelo, checkpoint y semillas.
- Usar la duración medida y el cómputo disponible para reservar tiempo de compilación, revisión y ensayo. Si una reproducción completa no cabe o el modelo de precios sigue siendo inestable, presentar la evidencia disponible, indicar la brecha exacta y dejar la meta como condicionada. Una sensibilidad con agentes congelados solo acredita su comportamiento en ese escenario.

## Jueves 19 · Cerrar los materiales en español

| Entregable | Trabajo y prueba de aceptación |
|---|---|
| Tabla de la meta conjunta | Generar desde las salidas una fila por política, semilla y H0. Comparar en el mismo último año el retorno con H0 y la reducción de quiebres con H0 y S0 emparejado. Mostrar el umbral del 30 %, los denominadores y ambas coberturas. Como control de la campaña existente deben salir 7/24 y 0/24, y para PPO 2/3 y 0/3. Incluir resultados agregados y dispersión. |
| Informe IEEE | Actualizar el LaTeX, el resumen, la sección de retail, figuras y conclusiones. Explicar cómo se induce H0, qué costes se suponen, qué elasticidad usó cada corrida y qué representa cada retorno. Incluir curvas, pérdidas, eficiencia al 80 %, estabilidad y ablaciones. Revisar visualmente el PDF de hasta ocho páginas y contrastar cada cifra con las tablas. |
| Sustentación y demo | Actualizar las diapositivas y el guion. Cargar PPO semilla 3 y ejecutar inferencia desde el checkpoint. Mostrar en pantalla observación, precio, pedido, asignación regional y ventas de las tres tiendas durante varios días. Probar en el equipo de exposición o grabar la ejecución en alta resolución. Ensayar los 20 minutos y repartir las preguntas teóricas entre integrantes. |
| Repositorio y ZIP | Integrar las versiones revisadas en el repositorio a entregar, identificar el commit y comprobar el acceso al mismo. Ejecutar el notebook y los comandos documentados desde una carpeta limpia. Generar un ZIP nuevo, extraerlo y abrir PDF, presentación y demo. Verificar que los resultados indicados en el README coincidan con el material efectivamente incluido. |

## Viernes 20 · Entrega y sustentación

El equipo sube a Canvas los archivos solicitados antes de la presentación, verifica la constancia y abre el PDF entregado. Lleva al aula la demo probada o el video y una copia local del material. La versión de entrega debe identificar cualquier punto de la rúbrica que permanezca parcial, junto con la prueba que sí se completó.

**Primer paso de ejecución.** Recuperar el checkpoint de PPO semilla 3 y revisar una corrida completa buscando recompensas diarias de entrenamiento. Esta inspección determina de inmediato si podemos preparar la demo y reconstruir G₀ con los archivos existentes.

Evidencia de base en [el resumen de calibración y la verificación del último año](../cierre_p2/README.md).
