"""
Endpoints FastAPI para Inferencia Predictiva y Teoría de Colas
Sistema CITEF - Facultad de Ciencias Económicas y Empresariales (UAGRM)
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from math_models.colas_mmc import calcular_mmc
from ml.predictor import get_predictor

router = APIRouter(tags=["Analítica Predictiva y Colas"])


# --- Modelos Pydantic para Inferencia de Demanda ---

class PredecirDemandaRequest(BaseModel):
    id_sala: int = Field(..., description="Identificador único del laboratorio", examples=[1])
    fecha: str = Field(..., description="Fecha de consulta (YYYY-MM-DD)", examples=["2026-10-26"])
    capacidad_sala: int = Field(..., gt=0, description="Aforo total físico de puestos de la sala", examples=[35])
    es_semana_examenes: bool = Field(False, description="Indica si corresponde a periodo de parciales/finales")
    hora_inicio: int = Field(7, ge=7, le=22, description="Hora de inicio de la jornada (7 a 22)")
    hora_fin: int = Field(21, ge=7, le=22, description="Hora de finalización de la jornada (7 a 22)")


class ProyeccionHora(BaseModel):
    hora: int
    hora_formato: str
    demanda_esperada: float
    aforo_maximo: int
    porcentaje_saturacion: float
    nivel_riesgo: str


class PredecirDemandaResponse(BaseModel):
    fecha: str
    id_sala: int
    hora_pico: int
    hora_pico_formato: str
    demanda_pico: float
    saturacion_maxima_porcentaje: float
    proyecciones: List[ProyeccionHora]
    recomendacion: str


# --- Modelos Pydantic para Simulación de Colas M/M/c ---

class SimularColasRequest(BaseModel):
    lambda_llegadas: float = Field(..., gt=0, description="Tasa de arribo de estudiantes por hora (lambda)")
    mu_servicio: float = Field(..., gt=0, description="Tasa de atención de cada puesto por hora (mu)")
    c_servidores: int = Field(..., ge=1, description="Cantidad de servidores/puestos de cómputo en sala (c)")
    tolerancia_espera_min: float = Field(5.0, gt=0, description="Tiempo de espera en cola tolerado en minutos")


class SimularColasResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lambda_llegadas: float = Field(..., alias="lambda")
    mu: float
    c: int
    factor_utilizacion_rho: float
    prob_espera_erlang_c: float
    prob_sistema_vacio_p0: Optional[float] = None
    tiempo_espera_cola_minutos: Optional[float] = None
    tiempo_espera_cola_horas: Optional[float] = None
    longitud_cola_estudiantes: Optional[float] = None
    tiempo_total_sistema_minutos: Optional[float] = None
    tiempo_total_sistema_horas: Optional[float] = None
    numero_medio_sistema_l: Optional[float] = None
    prob_esperar_mas_de_tolerancia: Optional[float] = None
    tolerancia_espera_min: float
    estado_sistema: str
    recomendacion_puestos: str
    c_optimo_recomendado: int


# --- Modelos Pydantic para Monitoreo Health ---

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


# --- Definición de Rutas ---

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Verifica la disponibilidad del microservicio y la carga en memoria del modelo de ML.
    """
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_loaded=(predictor.modelo is not None)
    )


@router.post("/predecir-demanda", response_model=PredecirDemandaResponse)
def post_predecir_demanda(payload: PredecirDemandaRequest):
    """
    CU26: Pronóstico horario de afluencia estudiantil y nivel de saturación
    mediante ensamble RandomForestRegressor entrenado con patrones FCEE.
    """
    try:
        predictor = get_predictor()
        resultado = predictor.predecir_demanda(
            fecha=payload.fecha,
            id_sala=payload.id_sala,
            capacidad_sala=payload.capacidad_sala,
            es_semana_examenes=payload.es_semana_examenes,
            hora_inicio=payload.hora_inicio,
            hora_fin=payload.hora_fin,
        )
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al calcular la predicción de demanda: {str(e)}"
        )


@router.post("/simular-colas", response_model=SimularColasResponse)
def post_simular_colas(payload: SimularColasRequest):
    """
    CU26: Evaluación estocástica de saturación y teoría de colas M/M/c (Erlang-C),
    calculando factor rho, tiempo Wq y dimensionamiento de puestos óptimos.
    """
    try:
        resultado = calcular_mmc(
            lambda_llegadas=payload.lambda_llegadas,
            mu_servicio=payload.mu_servicio,
            c_servidores=payload.c_servidores,
            tolerancia_espera_min=payload.tolerancia_espera_min,
        )
        # Adaptar clave 'lambda' a 'lambda_llegadas' para compatibilidad Pydantic alias
        resultado["lambda_llegadas"] = resultado.pop("lambda")
        return resultado
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en simulación estocástica M/M/c: {str(e)}"
        )
