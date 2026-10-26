"""
Pipeline de Entrenamiento de Modelo RandomForestRegressor
Demanda y Saturación de Laboratorios CITEF (FCEE - UAGRM)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_FILENAME = "modelo_rf_demanda.pkl"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

FEATURE_COLUMNS = [
    "dia_semana",
    "hora",
    "capacidad_sala",
    "es_semana_examenes",
    "es_turno_noche",
    "historico_promedio",
]


def generar_dataset_sintetico(n_muestras: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético con alta fidelidad a la dinámica académica
    de la Facultad de Ciencias Económicas y Empresariales de la UAGRM.
    """
    rng = np.random.RandomState(seed)

    dias = rng.choice([0, 1, 2, 3, 4, 5], size=n_muestras)  # 0=Lun .. 5=Sab
    horas = rng.randint(7, 22, size=n_muestras)             # 7 a 21
    capacidades = rng.choice([25, 30, 35, 40, 45], size=n_muestras)
    es_examenes = rng.binomial(1, 0.22, size=n_muestras)   # 22% de semanas son parciales/finales
    es_noche = np.where(horas >= 18, 1, 0)

    # Curva base horaria de demanda en facultades universitarias (proporción de capacidad)
    # 7-8: arranque matutino
    # 9-11: pico matutino
    # 12-13: valle almuerzo
    # 14-17: tarde activa
    # 18-20: pico turno noche (fuerte concurrencia de estudiantes trabajadores)
    # 21: cierre
    factores_hora = {
        7: 0.35, 8: 0.60, 9: 0.88, 10: 0.92, 11: 0.85,
        12: 0.45, 13: 0.40, 14: 0.65, 15: 0.78, 16: 0.82,
        17: 0.85, 18: 0.95, 19: 0.98, 20: 0.88, 21: 0.50
    }

    # Factor día: Lun a Jueves muy activos, Viernes tarde menor, Sábado sólo mañana
    factores_dia = {0: 1.0, 1: 1.02, 2: 1.0, 3: 0.98, 4: 0.90, 5: 0.75}

    demandas = []
    historicos = []

    for d, h, cap, exam, noche in zip(dias, horas, capacidades, es_examenes, es_noche):
        f_h = factores_hora.get(h, 0.5)
        f_d = factores_dia.get(d, 1.0)
        if d == 5 and h > 13:
            # Sábado en la tarde hay muy baja concurrencia
            f_h *= 0.25

        base_ratio = f_h * f_d
        # Semanas de exámenes aumentan afluencia en un 25% - 40%
        if exam == 1:
            base_ratio *= rng.uniform(1.20, 1.40)

        # Promedio histórico típico con leve ruido gaussiano
        hist = np.clip(cap * base_ratio + rng.normal(0, 1.5), 0, cap * 1.3)
        historicos.append(round(float(hist), 1))

        # Demanda observada (puede superar capacidad cuando hay colas y sobrecupo)
        dem = hist + rng.normal(0, 2.0)
        dem = max(0.0, round(float(dem), 1))
        demandas.append(dem)

    df = pd.DataFrame({
        "dia_semana": dias,
        "hora": horas,
        "capacidad_sala": capacidades,
        "es_semana_examenes": es_examenes,
        "es_turno_noche": es_noche,
        "historico_promedio": historicos,
        "demanda_estudiantes": demandas,
    })
    return df


def entrenar_modelo(
    guardar_ruta: str = DEFAULT_MODEL_PATH,
    n_muestras: int = 6000,
    seed: int = 42,
) -> RandomForestRegressor:
    """
    Entrena un RandomForestRegressor sobre el dataset generado y lo persiste en disco.
    """
    df = generar_dataset_sintetico(n_muestras=n_muestras, seed=seed)
    X = df[FEATURE_COLUMNS]
    y = df["demanda_estudiantes"]

    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X, y)

    # Evaluación interna rápida
    y_pred = modelo.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    os.makedirs(os.path.dirname(os.path.abspath(guardar_ruta)), exist_ok=True)
    joblib.dump(modelo, guardar_ruta)
    print(f"[ML Pipeline] Modelo entrenado con éxito. R2: {r2:.4f}, MAE: {mae:.2f}. Guardado en: {guardar_ruta}")
    return modelo


def cargar_o_entrenar_modelo(modelo_ruta: str = DEFAULT_MODEL_PATH) -> RandomForestRegressor:
    """
    Carga el modelo serializado si existe; de lo contrario, lo entrena automáticamente.
    """
    if os.path.exists(modelo_ruta):
        try:
            modelo = joblib.load(modelo_ruta)
            return modelo
        except Exception as e:
            print(f"[ML Pipeline] Advertencia al cargar {modelo_ruta}: {e}. Reentrenando...")
    return entrenar_modelo(guardar_ruta=modelo_ruta)


if __name__ == "__main__":
    entrenar_modelo()
