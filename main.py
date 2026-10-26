"""
Microservicio de Analítica Predictiva y Teoría de Colas CITEF
Facultad de Ciencias Económicas y Empresariales - UAGRM

Puerto: 8001
Prefijo de API: /api/v1/ml
Documentación interactiva: /docs
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Asegurar que el directorio raíz de citef_ml esté en el sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from endpoints.prediccion import router as prediccion_router
from ml.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicialización de recursos al arrancar el microservicio.
    Carga o entrena en memoria el modelo RandomForestRegressor.
    """
    print("[CITEF ML Service] Inicializando modelo predictivo...")
    predictor = get_predictor()
    print(f"[CITEF ML Service] Modelo listo en memoria. Estimadores: {len(predictor.modelo.estimators_)}")
    yield
    print("[CITEF ML Service] Finalizando servicio predictivo.")


app = FastAPI(
    title="CITEF Machine Learning & Queuing Service",
    description=(
        "Microservicio de Inteligencia Artificial y Analítica Predictiva para el Sistema CITEF "
        "(Facultad de Ciencias Económicas y Empresariales - UAGRM). Implementa inferencia de "
        "demanda estudiantil horaria con RandomForestRegressor y modelado estocástico de Teoría de Colas M/M/c."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configuración de CORS para permitir consumo desde NestJS (puerto 3000) y Angular (puerto 4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de rutas bajo el prefijo institucional
app.include_router(prediccion_router, prefix="/api/v1/ml")


@app.get("/", tags=["Root"])
def root_info():
    """
    Información general del estado del microservicio.
    """
    return {
        "service": "CITEF Machine Learning & Queuing Service",
        "institution": "Facultad de Ciencias Económicas y Empresariales - UAGRM",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/v1/ml/health",
            "predecir_demanda": "/api/v1/ml/predecir-demanda",
            "simular_colas": "/api/v1/ml/simular-colas",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
