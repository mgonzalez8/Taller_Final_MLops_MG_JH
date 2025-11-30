# Usar imagen base de Python
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY requirements.txt .
COPY . ./

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Argumentos de construcción para variables de entorno
ARG AWS_ACCESS_KEY
ARG AWS_SECRET_KEY
ARG AWS_BUCKET
ARG MODEL_FILE
ARG ENV

# Establecer variables de entorno en el contenedor
ENV AWS_ACCESS_KEY=${AWS_ACCESS_KEY}
ENV AWS_SECRET_KEY=${AWS_SECRET_KEY}
ENV AWS_BUCKET=${AWS_BUCKET}
ENV MODEL_FILE=${MODEL_FILE}
ENV ENV=${ENV}


# Descargar el modelo desde S3 durante la construcción
RUN python scripts/setup.py && \
    python scripts/pretrain.py

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
