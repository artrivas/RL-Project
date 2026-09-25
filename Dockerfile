FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Crear y establecer directorio de trabajo
WORKDIR /workspace

# Copiar e instalar dependencias de Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Exponer el puerto de Jupyter Lab
EXPOSE 8888

# Comando por defecto: levantar Jupyter Lab accesible sin token para entorno local
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
