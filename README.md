# ML---RL-KNN-KMEANS

Repositorio para realizar análisis de dos datasets con los modelos de Regresión Logística / KNN / KMEANS mediante gráficas y por último generando una web que diga por medio de un formulario de variables realice las predicciones.

# Clonar el proyecto

1: git clone https://github.com/AnderG-Uni/ML---RL-KNN-KMEANS.git
2: cd ML---RL-KNN-KMEANS
3: ejecutar el proyecto

# Estructura Recomendada del Proyecto

ML---RL-KNN-KMEANS/
│
├── Datasets/
│ ├── CC*GENERAL.csv
│ └── WA_Fn-UseC*-Telco-Customer-Churn.csv
│
├── img/
│ └── (imágenes usadas)
│
├── Modelo-KMEANS/
│ ├── modelo_kmeans.pkl
│ ├── pca.pkl
│ ├── scaler.pkl
│ └── variables.pkl
│
├── Modelo-KNN/
│ ├── Modelo_knn.pkl
│ └── Scaler_knn.pkl
│
├── Modelo-RL/
│ ├── Modelo_RL.pkl
│ └── Scaler_RL.pkl
│
├── backend.log
├── index.html
├── main.py
├── README.md
└── requirements.txt

# Versión de python recomendado

Python 3.12.2 o superior

# Instalar dependencias para evitar errores

pip install fastapi uvicorn joblib scikit-learn numpy

# Cómo ejecutar el Backend (FastAPI)

uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Endpoints disponibles

1: POST http://127.0.0.1:8000/predict-regresion
2: POST http://127.0.0.1:8000/predict-knn
3: POST http://127.0.0.1:8000/predict

# Para realizar pruebas con postman : ejemplos

- http://127.0.01:8000/predict-regresion
  {
  "Cargos_Mensuales": 90.50,
  "Genero": "Male",
  "Adulto_Mayor": 5,
  "Pareja": "Yes",
  "Dependientes": "No",
  "Antiguedad": 12,
  "Servicio_Telefonico": "Yes",
  "Lineas_Multiples": "Yes",
  "Servicio_Internet": "Fiber optic",
  "Seguridad_Online": "No",
  "Backup_Online": "Yes",
  "Proteccion_Dispositivo": "No",
  "Soporte_Tecnico": "Yes",
  "Contrato": "Month-to-month"
  }

- http://127.0.01:8000/predict-knn
  {
  "Cargos_Mensuales": 90.50,
  "Genero": "Male",
  "Adulto_Mayor": 5,
  "Pareja": "Yes",
  "Dependientes": "No",
  "Antiguedad": 12,
  "Servicio_Telefonico": "Yes",
  "Lineas_Multiples": "Yes",
  "Servicio_Internet": "Fiber optic",
  "Seguridad_Online": "No",
  "Backup_Online": "Yes",
  "Proteccion_Dispositivo": "No",
  "Soporte_Tecnico": "Yes",
  "Contrato": "Month-to-month"
  }

- http://127.0.01:8000/predict-kmeans
  {
  "Saldo_Actual": 1300,
  "Frecuencia_Actualizacion_Saldo": 0.96,
  "Compras_Totales": 270,
  "Compras_Unico_Pago": 130,
  "Compras_Cuotas": 140,
  "Avances_Efectivo": 530,
  "Frecuencia_Compras": 0.32,
  "Frecuencia_Compras_Unico_Pago": 0.07,
  "Frecuencia_Compras_Cuotas": 0.25,
  "Frecuencia_Avances": 0.10,
  "Transacciones_Avances": 1.78,
  "Transacciones_Compras": 5.72,
  "Limite_Credito": 3200,
  "Pagos": 610,
  "Pagos_Minimos": 450,
  "Porcentaje_Pago_Completo": 0.016,
  "Antiguedad_Cliente": 12
  }

# Cómo ejecutar el frontend

Se recomienda ejecutar un servidor web para evitar problemas de cors aunque el backend permite cualquier origen.

- python -m http.server 8080

# url de acceso

http://127.0.0.1:8080

# Modelos de Machine Learning

Los modelos fueron entrenados previamente y guardados con joblib
