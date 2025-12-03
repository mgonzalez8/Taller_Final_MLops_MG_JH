import os
import logging
import boto3
import onnxruntime as ort
import numpy as np
import streamlit as st
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import scripts.download_model as download_model
from scripts.class_mapper import decode_prediction

# Cargar variables de entorno desde .env
load_dotenv()

# --- Configuración Inicial ---
st.set_page_config(page_title="API de Inferencia ONNX", layout="centered")

# Configurar logs de la aplicación
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de Entorno (Estas se inyectan desde el Dockerfile o docker-compose)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.getenv("AWS_BUCKET", "mi-bucket-modelos")
ENV = os.getenv("ENV", "dev")
MODEL_FILE_PATH = os.getenv("MODEL_FILE", "model.onnx")

# Obtener el nombre del archivo del modelo desde la ruta
MODEL_FILENAME = os.path.basename(MODEL_FILE_PATH)

# Ruta completa del modelo en la carpeta raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FULL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILENAME)

# Cliente S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Variable global para cargar el modelo en memoria (usando session_state de Streamlit)
if "ort_session" not in st.session_state:
    st.session_state.ort_session = None

def log_prediction_to_s3(features, prediction):
    file_name = f"predicciones_{ENV}.txt"
    local_file = os.path.join(PROJECT_ROOT, file_name)
    
    log_entry = f"Input: {features} | Prediction: {prediction}\n"

    try:
        # 1. Intentar descargar el archivo de logs actual (si existe) para mantener el historial
        try:
            s3_client.download_file(BUCKET, file_name, local_file)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info("Archivo de logs no existe aún, creando uno nuevo.")
            else:
                logger.warning(f"Error bajando logs: {e}")

        # 2. Agregar la nueva línea (Append)
        with open(local_file, "a") as f:
            f.write(log_entry)

        # 3. Subir el archivo actualizado a S3
        s3_client.upload_file(local_file, BUCKET, file_name)
        logger.info(f"Log actualizado en S3: {file_name}")

    except Exception as e:
        logger.error(f"Fallo al guardar el log en S3: {e}")

# --- Eventos de Ciclo de Vida ---

def load_model():
    if st.session_state.ort_session is not None:
        return st.session_state.ort_session
    
    try:
        # 1. Descargar el modelo si no existe
        if not os.path.exists(MODEL_FULL_PATH):
            logger.info(f"Intentando descargar {MODEL_FILENAME} del bucket {BUCKET}...")
            download_model.download_model()
        
        # 2. Cargar el modelo en ONNX Runtime
        if not os.path.exists(MODEL_FULL_PATH):
            logger.error(f"El archivo del modelo no existe en la ruta: {MODEL_FULL_PATH}")
            st.error("El archivo del modelo no se encontró. Verifica la descarga.")
            return None
        
        ort_session = ort.InferenceSession(MODEL_FULL_PATH)
        st.session_state.ort_session = ort_session
        logger.info("Sesión de ONNX Runtime iniciada correctamente.")
        return ort_session
    except Exception as e:
        logger.error(f"Error cargando ONNX: {e}")
        st.error(f"Error al cargar el modelo: {e}")
        return None

# --- Interfaz Streamlit ---

st.title("🤖 API de Inferencia ONNX-Prueba")
st.markdown(f"**Entorno:** `{ENV}` | **Modelo:** `{MODEL_FILENAME}`")

# Cargar el modelo al iniciar
ort_session = load_model()

if ort_session is None:
    st.error("No se pudo cargar el modelo. Verifica la configuración de AWS y el archivo de modelo.")
    st.stop()

# --- Formulario de Predicción ---
st.header("Realizar Predicción")
st.markdown("Ingresa los features para obtener una predicción del modelo ONNX.")

# Obtener información del modelo
input_info = ort_session.get_inputs()[0]
num_features = input_info.shape[1] if len(input_info.shape) > 1 else 1

# Crear inputs dinámicos según el número de features
st.subheader("Features de entrada")
features = []
cols = st.columns(min(num_features, 4))  # Máximo 4 columnas
for i in range(num_features):
    col = cols[i % len(cols)]
    value = col.number_input(
        f"Feature {i+1}",
        value=0.0,
        step=0.1,
        format="%.4f"
    )
    features.append(value)

# Botón para realizar predicción
if st.button("🚀 Hacer Predicción", use_container_width=True):
    try:
        # Preparar datos
        input_name = ort_session.get_inputs()[0].name
        input_data = np.array([features], dtype=np.float32)
        
        # Inferencia
        result = ort_session.run(None, {input_name: input_data})
        
        if isinstance(result[0], np.ndarray):
            prediction_numeric = float(result[0].flatten()[0])
        else:
            prediction_numeric = float(result[0][0]) if isinstance(result[0], (list, tuple)) else float(result[0])
        
        prediction_result = decode_prediction(prediction_numeric)
        st.success("✅ Predicción realizada exitosamente")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicción", prediction_result)
        with col2:
            st.metric("Entorno", ENV)
        
        # Registrar en S3
        with st.spinner("Guardando registro en S3..."):
            log_prediction_to_s3(features, prediction_result)
        st.info("📝 Predicción registrada en S3")
        
    except Exception as e:
        logger.error(f"Error en inferencia: {e}")
        st.error(f"❌ Error durante la predicción: {e}")

# --- Sidebar con información ---
st.sidebar.header("ℹ️ Información")
st.sidebar.markdown("""
### Configuración
- **Modelo:** `{}`
- **Entorno:** `{}`
""".format(MODEL_FILENAME, ENV))

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Acerca de
Esta aplicación utiliza **ONNX Runtime** para realizar inferencias
con modelos pre-entrenados. Los resultados se registran en **Amazon S3**.
""")
