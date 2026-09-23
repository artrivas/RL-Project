# Starter Kit  - Proyecto de Aprendizaje por Refuerzo (DS5345)

**Curso:** DS5345 · Aprendizaje por Refuerzo (2026-II)  
**Institución:** UTEC - Universidad de Ingeniería y Tecnología  
**Docente:** Percy W. Lovon Ramos  
**Dominio:** Fútbol Autónomo Multi-Agente en Simulación 2D (RoboCup 2D Soccer Simulation League)

---

## 1. Contenido de este Starter Kit

Este paquete contiene todo lo necesario para comenzar a trabajar de inmediato en un entorno reproducible y aislado de tu sistema operativo:

```text
Starter_Kit_Proyecto_RoboCup2D/
├── Enunciado Proyecto.pdf      # Documento oficial completo con rúbricas y requerimientos
├── docker-compose.yml                  # Orquestador del simulador (rcssserver) y el agente (rl-agent)
├── Dockerfile                          # Configuración de la imagen del agente (Python 3.10 + PyTorch)
├── requirements.txt                    # Dependencias de Python instaladas
├── README.md                           # Esta guía paso a paso
│
├── workspace/                          # Carpeta montada en el contenedor (aquí programas)
│   ├── robocup_client.py              # Cliente UDP oficial para comunicar Python con rcssserver
│   ├── test_random_agent.py           # Script de humo (smoke test) para validar la conexión
│   ├── tutorial_primer_paso.ipynb     # Cuaderno tutorial: lectura de observaciones y acciones
│   └── agente_cero_mc_control_persecucion.ipynb  # Baseline completo de Monte Carlo Control
│
└── plantilla_informe_latex/           # Plantilla oficial en LaTeX para redactar los informes
    ├── informe_proyecto_template.tex   # Código fuente LaTeX estructurado según los criterios de rúbrica
    ├── informe_proyecto_template.pdf   # Versión compilada de referencia
    └── logo.png                        # Isotipo institucional UTEC
```

---

## 2. Guía de Inicio Rápido 

### Requisito Previo
Tener instalado y abierto **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (funciona en macOS Apple Silicon / Intel, Linux y Windows con WSL2).

### Paso 1: Abrir la terminal en esta carpeta
```bash
cd Starter_Kit_Proyecto_RoboCup2D
```

### Paso 2: Iniciar los contenedores
Ejecuta el siguiente comando para levantar el simulador y el agente:
```bash
docker compose up -d
```
> **Nota:** La primera ejecución descargará las imágenes y construirá el entorno. En ejecuciones posteriores iniciará en menos de 5 segundos.

### Paso 3: Abrir Jupyter Lab en tu navegador
Una vez iniciado, abre en tu navegador:
**[http://localhost:8888](http://localhost:8888)**

No requiere contraseña. En la barra lateral verás la carpeta `workspace/` con todos los archivos de trabajo.

---

## 3. Verificación de la Conexión 

Para verificar que el contenedor del agente se comunica correctamente con el servidor de RoboCup:

### Opción A: Desde Jupyter Lab (Recomendada)
1. Abre el cuaderno `workspace/tutorial_primer_paso.ipynb`.
2. Ejecuta las celdas secuencialmente (`Shift + Enter`).
3. Observarás el mensaje `Conectado exitosamente a rcssserver` y una gráfica de trayectoria.

### Opción B: Desde la terminal
```bash
docker compose exec rl-agent python /workspace/test_random_agent.py
```

---

## 4. Hoja de Ruta del Proyecto

El proyecto se desarrolla en dos fases acumulativas:

| Fase | Entrega | Ponderación | Contenido Principal |
| :--- | :--- | :---: | :--- |
| **P1** | **Semana 7** | **10%** | **Baseline Tabular en Tarea Restringida:** Modelado MDP formal ($\mathcal{S}, \mathcal{A}, \mathcal{R}, \gamma$), discretización justificada analíticamente, implementación de Monte Carlo Control / Q-Learning / SARSA, comparación de políticas de exploración ($\varepsilon$ fijo vs. decreciente) e informe técnico de 5 páginas. |
| **P2** | **Semana 15** | **20%** | **Deep RL y Desafío de Integración:** Redes neuronales continuas (DQN, PPO, DDPG), reward shaping para evitar *reward hacking*, y elección entre: <br>• **Vía A:** Partido competitivo MARL contra campeones RoboCup (HELIOS, CYRUS). <br>• **Vía B:** Transferencia de políticas (*Transfer Learning*) a robótica móvil, búsqueda y rescate, o AGVs logísticos. |

---

## 5. Comandos Útiles de Docker

* **Ver el estado de los contenedores:**
  ```bash
  docker compose ps
  ```
* **Ver los logs de la simulación o el agente:**
  ```bash
  docker compose logs -f rcssserver
  docker compose logs -f rl-agent
  ```
* **Reiniciar el simulador:**
  ```bash
  docker compose restart rcssserver
  ```
* **Detener los servicios:**
  ```bash
  docker compose down
  ```

---

## 6. Formato de Entrega del Grupo

Cada equipo (hasta 3 estudiantes) deberá entregar:
1. **Repositorio privado de GitHub / GitLab** (acceso otorgado al docente) con la siguiente estructura:
   ```text
   grupo-XX-proyecto-rl/
   ├── docker/              # Archivos Docker para reproducibilidad
   ├── src/                 # Código modular Python (agente, discretizador, cliente)
   ├── notebooks/           # Cuadernos Jupyter con curvas de entrenamiento
   ├── informe_p1.pdf       # Informe compilado desde la plantilla LaTeX
   └── README.md            # Instrucciones claras para reproducir los experimentos
   ```
2. **Informe Técnico en PDF:** Compilado desde la plantilla provista en `plantilla_informe_latex/` y entregado a través de la plataforma Canvas.

---
¡Muchos éxitos en el desarrollo del proyecto! Para dudas o soporte técnico, consulta con el docente o a través de los canales oficiales del curso.
