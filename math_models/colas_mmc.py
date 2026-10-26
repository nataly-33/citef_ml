"""
Módulo de Teoría de Colas M/M/c (Erlang-C)
Sistema CITEF - Facultad de Ciencias Económicas y Empresariales (UAGRM)

Implementación matemática rigurosa para análisis estocástico de saturación
y dimensionamiento de puestos en salas de cómputo universitarias.
"""

import math
from typing import Dict, Any, Optional


def _calcular_metricas_estables(
    lambda_llegadas: float,
    mu_servicio: float,
    c_servidores: int,
    tolerancia_espera_min: float = 5.0,
) -> Dict[str, Any]:
    """
    Calcula las métricas de Erlang-C asumiendo que rho < 1.0.
    """
    rho = lambda_llegadas / (c_servidores * mu_servicio)
    c_rho = c_servidores * rho  # lambda / mu

    # Sumatoria para P0: sum_{n=0}^{c-1} (c*rho)^n / n!
    sumatoria = 0.0
    termino = 1.0  # para n = 0: (c*rho)^0 / 0! = 1
    sumatoria += termino
    for n in range(1, c_servidores):
        termino *= c_rho / n
        sumatoria += termino

    # Último término de P0: (c*rho)^c / (c! * (1 - rho))
    # Calculamos (c*rho)^c / c!
    termino_c = termino * (c_rho / c_servidores)
    termino_cola = termino_c / (1.0 - rho)

    p0 = 1.0 / (sumatoria + termino_cola)

    # Probabilidad de espera Erlang-C (Pc)
    pc = termino_cola * p0
    # Asegurar cota numérica [0, 1]
    pc = max(0.0, min(1.0, pc))

    # Longitud media de la cola Lq
    lq = (pc * rho) / (1.0 - rho)

    # Tiempo medio de espera en cola Wq (horas y minutos)
    wq = lq / lambda_llegadas
    wq_minutos = wq * 60.0

    # Tiempo total medio en el sistema Ws (horas y minutos)
    ws = wq + (1.0 / mu_servicio)
    ws_minutos = ws * 60.0

    # Número medio de estudiantes en el sistema L
    l_estudiantes = lambda_llegadas * ws

    # Probabilidad de esperar más de t minutos P(Wq > t)
    t_horas = tolerancia_espera_min / 60.0
    p_espera_mayor_t = pc * math.exp(-c_servidores * mu_servicio * (1.0 - rho) * t_horas)

    return {
        "rho": rho,
        "p0": p0,
        "pc": pc,
        "lq": lq,
        "wq": wq,
        "wq_minutos": wq_minutos,
        "ws": ws,
        "ws_minutos": ws_minutos,
        "l": l_estudiantes,
        "p_espera_mayor_t": p_espera_mayor_t,
    }


def encontrar_c_optimo(
    lambda_llegadas: float,
    mu_servicio: float,
    tolerancia_espera_min: float = 5.0,
    max_c: int = 500,
) -> int:
    """
    Calcula cuál es el c_optimo mínimo para que rho < 0.80 y Wq_minutos < tolerancia_espera_min.
    """
    # Mínimo absoluto para que rho < 0.80: c > lambda / (0.80 * mu)
    c_inicial = max(1, math.floor(lambda_llegadas / (0.80 * mu_servicio)) + 1)

    for c in range(c_inicial, max_c + 1):
        rho = lambda_llegadas / (c * mu_servicio)
        if rho >= 0.80:
            continue
        metricas = _calcular_metricas_estables(lambda_llegadas, mu_servicio, c, tolerancia_espera_min)
        if metricas["wq_minutos"] < tolerancia_espera_min:
            return c

    return c_inicial


def calcular_mmc(
    lambda_llegadas: float,
    mu_servicio: float,
    c_servidores: int,
    tolerancia_espera_min: float = 5.0,
) -> Dict[str, Any]:
    """
    Cálculo estocástico riguroso de Teoría de Colas M/M/c (Erlang-C).

    Parámetros:
        lambda_llegadas: Tasa media de arribos (estudiantes/hora).
        mu_servicio: Tasa media de atención de un puesto (estudiantes/hora).
        c_servidores: Cantidad de puestos de cómputo disponibles.
        tolerancia_espera_min: Tiempo máximo de espera aceptable en cola (minutos).

    Retorna un diccionario estructurado con los indicadores de desempeño
    y la recomendación de capacidad óptima.
    """
    if lambda_llegadas <= 0 or mu_servicio <= 0 or c_servidores < 1:
        raise ValueError("lambda y mu deben ser > 0, y c debe ser >= 1.")

    rho = lambda_llegadas / (c_servidores * mu_servicio)
    c_optimo = encontrar_c_optimo(lambda_llegadas, mu_servicio, tolerancia_espera_min)

    # Caso Saturado / Inestable
    if rho >= 1.0:
        puestos_faltantes = max(0, c_optimo - c_servidores)
        recomendacion = (
            f"SISTEMA SATURADO: El factor de utilización (ρ = {rho:.2f}) supera o iguala la "
            f"capacidad total del laboratorio ({c_servidores} puestos). La cola de espera crecerá "
            f"indefinidamente. Se requiere habilitar de inmediato un mínimo de {c_optimo} puestos "
            f"(+{puestos_faltantes} adicionales) para estabilizar la tasa de servicio a un "
            f"nivel seguro (< 80%) y lograr un tiempo de espera inferior a {tolerancia_espera_min:.1f} min."
        )
        return {
            "lambda": round(float(lambda_llegadas), 4),
            "mu": round(float(mu_servicio), 4),
            "c": int(c_servidores),
            "factor_utilizacion_rho": round(float(rho), 4),
            "prob_espera_erlang_c": 1.0,
            "prob_sistema_vacio_p0": 0.0,
            "tiempo_espera_cola_minutos": None,
            "tiempo_espera_cola_horas": None,
            "longitud_cola_estudiantes": None,
            "tiempo_total_sistema_minutos": None,
            "tiempo_total_sistema_horas": None,
            "numero_medio_sistema_l": None,
            "prob_esperar_mas_de_tolerancia": 1.0,
            "tolerancia_espera_min": float(tolerancia_espera_min),
            "estado_sistema": "SATURADO",
            "c_optimo_recomendado": int(c_optimo),
            "recomendacion_puestos": recomendacion,
        }

    # Caso Estable (rho < 1.0)
    metricas = _calcular_metricas_estables(
        lambda_llegadas, mu_servicio, c_servidores, tolerancia_espera_min
    )

    if rho < 0.60:
        estado_sistema = "OPTIMO"
    elif rho < 0.80:
        estado_sistema = "MODERADO"
    elif rho < 0.95:
        estado_sistema = "ALTO"
    else:
        estado_sistema = "CRITICO"

    if c_servidores >= c_optimo:
        recomendacion = (
            f"CAPACIDAD ADECUADA: Con {c_servidores} puestos operativos, la utilización es del "
            f"{rho * 100:.1f}% (estado {estado_sistema}) y el tiempo medio en cola es de "
            f"{metricas['wq_minutos']:.2f} min (tolerancia {tolerancia_espera_min:.1f} min). "
            f"Capacidad mínima recomendada: {c_optimo} puestos."
        )
    else:
        puestos_sugeridos = c_optimo - c_servidores
        recomendacion = (
            f"CAPACIDAD INSUFICIENTE: Aunque el sistema es matemáticamente estable (ρ = {rho:.2f}), "
            f"el tiempo de espera en cola ({metricas['wq_minutos']:.2f} min) o la utilización superan "
            f"los umbrales de servicio óptimo FCEE. Se recomienda elevar la capacidad a {c_optimo} puestos "
            f"(+{puestos_sugeridos} puestos) para mitigar cuellos de botella."
        )

    return {
        "lambda": round(float(lambda_llegadas), 4),
        "mu": round(float(mu_servicio), 4),
        "c": int(c_servidores),
        "factor_utilizacion_rho": round(float(rho), 4),
        "prob_espera_erlang_c": round(float(metricas["pc"]), 4),
        "prob_sistema_vacio_p0": round(float(metricas["p0"]), 4),
        "tiempo_espera_cola_minutos": round(float(metricas["wq_minutos"]), 2),
        "tiempo_espera_cola_horas": round(float(metricas["wq"]), 4),
        "longitud_cola_estudiantes": round(float(metricas["lq"]), 2),
        "tiempo_total_sistema_minutos": round(float(metricas["ws_minutos"]), 2),
        "tiempo_total_sistema_horas": round(float(metricas["ws"]), 4),
        "numero_medio_sistema_l": round(float(metricas["l"]), 2),
        "prob_esperar_mas_de_tolerancia": round(float(metricas["p_espera_mayor_t"]), 4),
        "tolerancia_espera_min": float(tolerancia_espera_min),
        "estado_sistema": estado_sistema,
        "c_optimo_recomendado": int(c_optimo),
        "recomendacion_puestos": recomendacion,
    }
