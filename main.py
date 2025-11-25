import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import joblib
import os


app = FastAPI(
    title="API Modelos ML",
    description="modelos de machine learning",
    version="1.0"
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# obtener la ruta del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Crear archivo de log en la raíz del proyecto para ver las peticiones que llegan.
log_file = os.path.join(os.getcwd(), "backend.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#======================== Regresión Logistica ============================================

rl_path  = os.path.join(BASE_DIR, "Modelo-RL", "Modelo_RL.pkl")
scaler_rl_path  = os.path.join(BASE_DIR, "Modelo-RL", "Scaler_RL.pkl")
rl = joblib.load(rl_path)
scaler_rl = joblib.load(scaler_rl_path)


# --- Conversión de categorías numéricas ---
def convertir_valores(data):

    conv = {}
    conv["Cargos_Mensuales"] = data.Cargos_Mensuales
    conv["Genero"] = 1 if data.Genero.lower() == "male" else 0
    conv["Adulto_Mayor"] = int(data.Adulto_Mayor)
    conv["Pareja"] = 1 if data.Pareja.lower() == "yes" else 0
    conv["Dependientes"] = 1 if data.Dependientes.lower() == "yes" else 0
    conv["Antiguedad"] = int(data.Antiguedad)
    conv["Servicio_Telefonico"] = 1 if data.Servicio_Telefonico.lower() == "yes" else 0
    conv["Lineas_Multiples"] = 1 if data.Lineas_Multiples == "Yes" else 0
    conv["Servicio_Internet"] = {"No": 0, "DSL": 1, "Fiber optic": 2 }.get(data.Servicio_Internet, 0)
    conv["Seguridad_Online"] = 1 if data.Seguridad_Online == "Yes" else 0
    conv["Backup_Online"] = 1 if data.Backup_Online == "Yes" else 0
    conv["Proteccion_Dispositivo"] = 1 if data.Proteccion_Dispositivo == "Yes" else 0
    conv["Soporte_Tecnico"] = 1 if data.Soporte_Tecnico == "Yes" else 0
    conv["Contrato"] = {"Month-to-month": 0, "One year": 1, "Two year": 2 }.get(data.Contrato, 0)

    # Guardar LOG en el archivo backend.log
    logger.info("Datos recibidos convertidos: %s", conv)
    return conv


class EntradaClienteRL(BaseModel):
    Cargos_Mensuales: float
    Genero: str
    Adulto_Mayor: int
    Pareja: str
    Dependientes: str
    Antiguedad: int
    Servicio_Telefonico: str
    Lineas_Multiples: str
    Servicio_Internet: str
    Seguridad_Online: str
    Backup_Online: str
    Proteccion_Dispositivo: str
    Soporte_Tecnico: str
    Contrato: str

# Ruta que se debe consumir desde el front
@app.post("/predict-regresion")
def predecir_regresion_logistica(data: EntradaClienteRL):
    valores = convertir_valores(data)

    # Orden correcto para el modelo
    orden = [
        "Cargos_Mensuales","Genero","Adulto_Mayor","Pareja","Dependientes",
        "Antiguedad","Servicio_Telefonico","Lineas_Multiples","Servicio_Internet",
        "Seguridad_Online","Backup_Online","Proteccion_Dispositivo","Soporte_Tecnico",
        "Contrato"
    ]

    X = np.array([[valores[col] for col in orden]])

    # Escalar
    X_scaled = scaler_rl.transform(X)

    # Predecir churn (0/1)
    pred = int(rl.predict(X_scaled)[0])

    return {
        "churn": pred,
        "mensaje": "Predicción generada correctamente con el módelo Regresión Logística"
    }



#======================== KNN ============================================

knn_path  = os.path.join(BASE_DIR, "Modelo-KNN", "Modelo_knn.pkl")
scaler_knn_path  = os.path.join(BASE_DIR, "Modelo-KNN", "Scaler_knn.pkl")
knn = joblib.load(knn_path)
scaler_knn = joblib.load(scaler_knn_path)


# --- Conversión de categorías numéricas ---
def convertir_valoresKNN(data):

    conv = {}
    conv["Cargos_Mensuales"] = data.Cargos_Mensuales
    conv["Genero"] = 1 if data.Genero.lower() == "male" else 0
    conv["Adulto_Mayor"] = int(data.Adulto_Mayor)
    conv["Pareja"] = 1 if data.Pareja.lower() == "yes" else 0
    conv["Dependientes"] = 1 if data.Dependientes.lower() == "yes" else 0
    conv["Antiguedad"] = int(data.Antiguedad)
    conv["Servicio_Telefonico"] = 1 if data.Servicio_Telefonico.lower() == "yes" else 0
    conv["Lineas_Multiples"] = 1 if data.Lineas_Multiples == "Yes" else 0
    conv["Servicio_Internet"] = {"No": 0, "DSL": 1, "Fiber optic": 2}.get(data.Servicio_Internet, 0)
    conv["Seguridad_Online"] = 1 if data.Seguridad_Online == "Yes" else 0
    conv["Backup_Online"] = 1 if data.Backup_Online == "Yes" else 0
    conv["Proteccion_Dispositivo"] = 1 if data.Proteccion_Dispositivo == "Yes" else 0
    conv["Soporte_Tecnico"] = 1 if data.Soporte_Tecnico == "Yes" else 0
    conv["Contrato"] = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(data.Contrato, 0)

    # Guardar LOG en backend.log
    logger.info("Datos recibidos convertidos: %s", conv)
    return conv


class EntradaClienteKNN(BaseModel):
    Cargos_Mensuales: float
    Genero: str
    Adulto_Mayor: int
    Pareja: str
    Dependientes: str
    Antiguedad: int
    Servicio_Telefonico: str
    Lineas_Multiples: str
    Servicio_Internet: str
    Seguridad_Online: str
    Backup_Online: str
    Proteccion_Dispositivo: str
    Soporte_Tecnico: str
    Contrato: str

# Ruta que se debe consumir desde el front
@app.post("/predict-knn")
def predecir_regresion_knn(data: EntradaClienteKNN):

    valores = convertir_valoresKNN(data)
    # Orden correcto para el modelo
    orden = [
        "Cargos_Mensuales","Genero","Adulto_Mayor","Pareja","Dependientes",
        "Antiguedad","Servicio_Telefonico","Lineas_Multiples","Servicio_Internet",
        "Seguridad_Online","Backup_Online","Proteccion_Dispositivo","Soporte_Tecnico",
        "Contrato"
    ]

    X = np.array([[valores[col] for col in orden]])

    # Escalar
    X_scaled = scaler_rl.transform(X)

    # Predecir churn (0/1)
    pred = int(rl.predict(X_scaled)[0])

    return {
        "churn": pred,
        "mensaje": "Predicción generada correctamente con el módelo Regresión  KNN. "
    }



#========================= K-MEANS ===========================================

kmeans_path = os.path.join(BASE_DIR, "Modelo-KMEANS", "modelo_kmeans.pkl")
scaler_path = os.path.join(BASE_DIR, "Modelo-KMEANS", "scaler.pkl")
variables_path = os.path.join(BASE_DIR, "Modelo-KMEANS", "variables.pkl")

kmeans = joblib.load(kmeans_path)
scaler = joblib.load(scaler_path)
variables = joblib.load(variables_path)

# Descripciones precalculadas
descripciones = {
    0: "Cluster 0 – Clientes con uso moderado: nivel medio de compras, uso frecuente de cuotas, avances relativamente altos y límite de crédito medio.",
    1: "Cluster 1 – Clientes de alto poder adquisitivo: mayor volumen de compras, mayor uso de un solo pago, altos límites de crédito y pagos más elevados.",
    2: "Cluster 2 – Clientes de bajo gasto: menores niveles de compras, límites de crédito bajos y uso moderado de avances."
}

class EntradaCliente(BaseModel):
    Saldo_Actual: float
    Frecuencia_Actualizacion_Saldo: float
    Compras_Totales: float
    Compras_Unico_Pago: float
    Compras_Cuotas: float
    Avances_Efectivo: float
    Frecuencia_Compras: float
    Frecuencia_Compras_Unico_Pago: float
    Frecuencia_Compras_Cuotas: float
    Frecuencia_Avances: float
    Transacciones_Avances: float
    Transacciones_Compras: float
    Limite_Credito: float
    Pagos: float
    Pagos_Minimos: float
    Porcentaje_Pago_Completo: float
    Antiguedad_Cliente: float

# Ruta que se debe consumir desde el front
@app.post("/predict-kmeans")
def predecir_cluster(data: EntradaCliente):

    X = np.array([[getattr(data, var) for var in variables]])

    # Escalar
    X_scaled = scaler.transform(X)

    # Predecir cluster
    cluster = int(kmeans.predict(X_scaled)[0])

    return {
        "cluster": cluster,
        "descripcion": descripciones[cluster]
    }

