"""
Servicio de Inferencia Predictiva de Demanda y Saturación
Sistema CITEF - Facultad de Ciencias Económicas y Empresariales (UAGRM)
"""

from datetime import datetime, date
from typing import List, Dict, Any, Union
import pandas as pd
from .pipeline_rf import cargar_o_entrenar_modelo, FEATURE_COLUMNS


class PredictorDemanda:
    def __init__(self, modelo_path: str = None):
        self.modelo = cargar_o_entrenar_modelo(modelo_path) if modelo_path else cargar_o_entrenar_modelo()

    def _estimar_historico_base(self, dia_semana: int, hora: int, capacidad: int, es_examenes: bool) -> float:
        """
        Calcula un promedio histórico contextualizado según el horario universitario FCEE.
        """
        factores_hora = {
            7: 0.35, 8: 0.60, 9: 0.88, 10: 0.92, 11: 0.85,
            12: 0.45, 13: 0.40, 14: 0.65, 15: 0.78, 16: 0.82,
            17: 0.85, 18: 0.95, 19: 0.98, 20: 0.88, 21: 0.50, 22: 0.20
        }
        factores_dia = {0: 1.0, 1: 1.02, 2: 1.0, 3: 0.98, 4: 0.90, 5: 0.75, 6: 0.10}

        f_h = factores_hora.get(hora, 0.45)
        f_d = factores_dia.get(dia_semana, 1.0)
        if dia_semana == 5 and hora > 13:
            f_h *= 0.25

        ratio = f_h * f_d
        if es_examenes:
            ratio *= 1.30

        return round(capacidad * ratio, 1)

    def predecir_demanda(
        self,
        fecha: Union[str, date],
        id_sala: int,
        capacidad_sala: int,
        es_semana_examenes: bool = False,
        hora_inicio: int = 7,
        hora_fin: int = 21,
    ) -> Dict[str, Any]:
        """
        Genera la curva horaria proyectada de demanda de estudiantes y saturación del laboratorio.
        """
        if isinstance(fecha, str):
            try:
                # Soporta 'YYYY-MM-DD' o ISO datetime
                parsed_date = datetime.fromisoformat(fecha.replace("Z", "+00:00")).date()
            except Exception:
                parsed_date = datetime.strptime(fecha[:10], "%Y-%m-%d").date()
        else:
            parsed_date = fecha

        fecha_str = parsed_date.strftime("%Y-%m-%d")
        dia_semana = parsed_date.weekday()  # 0=Lunes .. 6=Domingo

        if hora_inicio < 7:
            hora_inicio = 7
        if hora_fin > 22:
            hora_fin = 22
        if hora_inicio > hora_fin:
            hora_inicio, hora_fin = hora_fin, hora_inicio

        registros = []
        filas_features = []

        for hora in range(hora_inicio, hora_fin + 1):
            es_noche = 1 if hora >= 18 else 0
            hist = self._estimar_historico_base(
                dia_semana, hora, capacidad_sala, es_semana_examenes
            )
            fila = {
                "dia_semana": dia_semana if dia_semana < 6 else 5,
                "hora": hora,
                "capacidad_sala": capacidad_sala,
                "es_semana_examenes": 1 if es_semana_examenes else 0,
                "es_turno_noche": es_noche,
                "historico_promedio": hist,
            }
            filas_features.append(fila)

        df_input = pd.DataFrame(filas_features)[FEATURE_COLUMNS]
        predicciones = self.modelo.predict(df_input)

        proyecciones: List[Dict[str, Any]] = []
        demanda_max = -1.0
        saturacion_max = -1.0
        hora_pico = hora_inicio

        for fila, pred in zip(filas_features, predicciones):
            hora = fila["hora"]
            demanda_est = max(0.0, round(float(pred), 1))
            saturacion_pct = round((demanda_est / capacidad_sala) * 100.0, 2)

            if saturacion_pct < 60.0:
                nivel_riesgo = "BAJO"
            elif saturacion_pct < 80.0:
                nivel_riesgo = "MODERADO"
            elif saturacion_pct < 95.0:
                nivel_riesgo = "ALTO"
            else:
                nivel_riesgo = "CRITICO"

            if demanda_est > demanda_max:
                demanda_max = demanda_est
                saturacion_max = saturacion_pct
                hora_pico = hora

            proyecciones.append({
                "hora": hora,
                "hora_formato": f"{hora:02d}:00",
                "demanda_esperada": demanda_est,
                "aforo_maximo": capacidad_sala,
                "porcentaje_saturacion": saturacion_pct,
                "nivel_riesgo": nivel_riesgo,
            })

        # Generar recomendación operativa contextualizada
        if saturacion_max >= 95.0:
            recomendacion = (
                f"ALERTA CRÍTICA: Se proyecta sobrecupo severo en la hora pico ({hora_pico:02d}:00) "
                f"con {demanda_max:.0f} estudiantes para una capacidad de {capacidad_sala} puestos "
                f"({saturacion_max:.1f}% saturación). Se recomienda habilitar una sala de respaldo "
                f"o derivar grupos hacia la franja valle (12:00 a 14:00)."
            )
        elif saturacion_max >= 80.0:
            recomendacion = (
                f"ALERTA DE CONGESTIÓN: En la hora pico ({hora_pico:02d}:00) la sala alcanzará un "
                f"{saturacion_max:.1f}% de ocupación ({demanda_max:.0f} estudiantes). Se sugiere "
                f"priorizar cátedras con exámenes programados y limitar reservas de práctica libre."
            )
        elif saturacion_max >= 60.0:
            recomendacion = (
                f"OPERACIÓN MODERADA: Afluencia fluida con ocupación máxima del {saturacion_max:.1f}% "
                f"a las {hora_pico:02d}:00. La capacidad instalada de {capacidad_sala} puestos es adecuada."
            )
        else:
            recomendacion = (
                f"DISPONIBILIDAD AMPLIA: Sala holgada durante toda la jornada (pico {saturacion_max:.1f}%). "
                f"Excelente franja para programar mantenimientos de software o asignaciones de cátedras grandes."
            )

        return {
            "fecha": fecha_str,
            "id_sala": id_sala,
            "hora_pico": hora_pico,
            "hora_pico_formato": f"{hora_pico:02d}:00",
            "demanda_pico": demanda_max,
            "saturacion_maxima_porcentaje": saturacion_max,
            "proyecciones": proyecciones,
            "recomendacion": recomendacion,
        }


# Instancia singleton para reutilización de memoria
_predictor_instancia = None


def get_predictor() -> PredictorDemanda:
    global _predictor_instancia
    if _predictor_instancia is None:
        _predictor_instancia = PredictorDemanda()
    return _predictor_instancia
