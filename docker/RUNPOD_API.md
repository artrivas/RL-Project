# Validar P2 mediante la API key desde Ubuntu/WSL

Usar cuando la API key ve el Pod pero la conexión OAuth del plugin no lo ve.
El script usa únicamente Python estándar y la API v2 oficial. No necesita SSH.
No inicia/detiene Pods, no instala dependencias y no ejecuta la matriz M5.

Desde la raíz del proyecto, con un .env local que contenga id y api_key:

```bash
python3 scripts/runpod_validate.py --env-file .env
```

Acepta también RUNPOD_POD_ID y RUNPOD_API_KEY. Nunca imprime la clave ni la
transmite al contenedor: solo va en el encabezado HTTPS de autenticación de Runpod.

Valida exactamente el commit 4f399fe5534fe24bdb49b404aa9fa74f570c9c51.
Hace un checkout nuevo dentro de /workspace/rl-project-validation/p2-*,
sin tocar repositorios existentes. Ejecuta:
1. Diagnóstico GPU/PyTorch.
2. src.retail.preflight --require-cuda.
3. src.retail.smoke_deep --device cuda (ocho pasos por variante).

Los comandos usan tiempos máximos por etapa; el job queda desacoplado de la
conexión HTTP. Cada llamada a exec espera hasta 10 segundos, y el cliente consulta
el estado cada cinco segundos. No interpretar un timeout de la API como cancelación.

El identificador del trabajo se guarda, sin credenciales, en
notebooks/artifacts/p2_via_b/setup/runpod-validation.json. Tras una desconexión:

```bash
python3 scripts/runpod_validate.py --env-file .env --resume
```

El resultado y el log se guardan en el archivo vecino runpod-validation.result.json.
Un estado launch_pending sin directorio significa que el resultado del lanzamiento
es incierto: inspeccionar el Pod antes de lanzar otra validación.
Para una validación nueva deliberada, usar otro --state-file; no se borran trabajos.

Errores: 401 clave no aceptada; 403 permisos insuficientes; 404 Pod no accesible
con esa clave; 409 Pod no encendido; 503 host posiblemente sin soporte de exec.
Un fallo de preflight/smoke queda registrado y debe resolverse antes de M5.
