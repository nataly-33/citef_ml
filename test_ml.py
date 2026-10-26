"""
Suite de Pruebas Unitarias y de Integración
Microservicio CITEF Machine Learning & Queuing Service
"""

import sys
import os
import unittest
import math

# Incluir raíz del microservicio en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from math_models.colas_mmc import calcular_mmc, encontrar_c_optimo, _calcular_metricas_estables
from ml.pipeline_rf import generar_dataset_sintetico, entrenar_modelo, cargar_o_entrenar_modelo
from ml.predictor import PredictorDemanda, get_predictor
from fastapi.testclient import TestClient
from main import app


class TestTeoriaColasMMC(unittest.TestCase):
    """
    Pruebas matemáticas rigurosas para Erlang-C y Teoría de Colas M/M/c.
    """

    def test_sistema_estable_metricas_basicas(self):
        # lambda=20 estud/hora, mu=1 estud/hora por servidor, c=25 puestos
        # rho = 20 / (25 * 1) = 0.80
        resultado = calcular_mmc(lambda_llegadas=20.0, mu_servicio=1.0, c_servidores=25, tolerancia_espera_min=5.0)

        self.assertEqual(resultado["c"], 25)
        self.assertAlmostEqual(resultado["factor_utilizacion_rho"], 0.80, places=3)
        self.assertIn(resultado["estado_sistema"], ["MODERADO", "ALTO"])
        self.assertGreater(resultado["prob_espera_erlang_c"], 0.0)
        self.assertLess(resultado["prob_espera_erlang_c"], 1.0)
        self.assertIsNotNone(resultado["tiempo_espera_cola_minutos"])
        self.assertGreaterEqual(resultado["tiempo_espera_cola_minutos"], 0.0)

        # Ley de Little: Lq = lambda * Wq (en horas)
        lq = resultado["longitud_cola_estudiantes"]
        wq_horas = resultado["tiempo_espera_cola_horas"]
        self.assertAlmostEqual(lq, 20.0 * wq_horas, delta=0.05)

        # Ws = Wq + 1/mu -> Ws = Wq + 1 hora -> Ws_min = Wq_min + 60 min
        self.assertAlmostEqual(
            resultado["tiempo_total_sistema_minutos"],
            resultado["tiempo_espera_cola_minutos"] + 60.0,
            delta=0.05
        )

    def test_sistema_saturado(self):
        # lambda=30, mu=1, c=25 -> rho = 30 / 25 = 1.20 >= 1.0 (Inestable)
        resultado = calcular_mmc(lambda_llegadas=30.0, mu_servicio=1.0, c_servidores=25)

        self.assertEqual(resultado["estado_sistema"], "SATURADO")
        self.assertAlmostEqual(resultado["factor_utilizacion_rho"], 1.20, places=2)
        self.assertEqual(resultado["prob_espera_erlang_c"], 1.0)
        self.assertIsNone(resultado["tiempo_espera_cola_minutos"])
        self.assertIsNone(resultado["longitud_cola_estudiantes"])
        # Para estabilizar rho < 0.80: c > 30 / 0.80 = 37.5 -> c >= 38
        self.assertGreaterEqual(resultado["c_optimo_recomendado"], 38)
        self.assertIn("SISTEMA SATURADO", resultado["recomendacion_puestos"])

    def test_encontrar_c_optimo(self):
        # lambda=40, mu=2 (cada puesto atiende 2 estud/hora -> media 30 min)
        # c_optimo debe asegurar rho < 0.80 y Wq < 5 min
        c_opt = encontrar_c_optimo(lambda_llegadas=40.0, mu_servicio=2.0, tolerancia_espera_min=5.0)
        self.assertGreaterEqual(c_opt, 26)  # 40 / (c * 2) < 0.80 -> c > 25 -> c >= 26

        metricas = calcular_mmc(lambda_llegadas=40.0, mu_servicio=2.0, c_servidores=c_opt, tolerancia_espera_min=5.0)
        self.assertLess(metricas["factor_utilizacion_rho"], 0.80)
        self.assertLess(metricas["tiempo_espera_cola_minutos"], 5.0)

    def test_validaciones_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            calcular_mmc(lambda_llegadas=-5.0, mu_servicio=1.0, c_servidores=10)
        with self.assertRaises(ValueError):
            calcular_mmc(lambda_llegadas=10.0, mu_servicio=0.0, c_servidores=10)
        with self.assertRaises(ValueError):
            calcular_mmc(lambda_llegadas=10.0, mu_servicio=1.0, c_servidores=0)


class TestPipelineRandomForest(unittest.TestCase):
    """
    Pruebas para generación de datos sintéticos y entrenamiento del RandomForest.
    """

    def test_dataset_sintetico(self):
        df = generar_dataset_sintetico(n_muestras=500, seed=123)
        self.assertEqual(len(df), 500)
        self.assertTrue(all(col in df.columns for col in [
            "dia_semana", "hora", "capacidad_sala", "es_semana_examenes",
            "es_turno_noche", "historico_promedio", "demanda_estudiantes"
        ]))
        self.assertTrue(df["dia_semana"].between(0, 5).all())
        self.assertTrue(df["hora"].between(7, 21).all())
        self.assertTrue((df["demanda_estudiantes"] >= 0).all())

    def test_entrenamiento_y_carga_modelo(self):
        modelo = cargar_o_entrenar_modelo()
        self.assertIsNotNone(modelo)
        self.assertEqual(modelo.n_estimators, 100)
        self.assertEqual(modelo.max_depth, 10)


class TestPredictorDemanda(unittest.TestCase):
    """
    Pruebas para inferencia de curva horaria y detección de horas pico.
    """

    def setUp(self):
        self.predictor = get_predictor()

    def test_prediccion_jornada_completa(self):
        # Lunes 2026-10-26, sala con 35 puestos, semana normal
        resultado = self.predictor.predecir_demanda(
            fecha="2026-10-26",
            id_sala=1,
            capacidad_sala=35,
            es_semana_examenes=False,
            hora_inicio=7,
            hora_fin=21,
        )

        self.assertEqual(resultado["fecha"], "2026-10-26")
        self.assertEqual(resultado["id_sala"], 1)
        self.assertEqual(len(resultado["proyecciones"]), 15)  # 7 a 21 inclusive

        # Validar consistencia de las horas
        horas = [p["hora"] for p in resultado["proyecciones"]]
        self.assertEqual(horas, list(range(7, 22)))

        # Validar identificación de hora pico
        self.assertIn(resultado["hora_pico"], horas)
        self.assertGreater(resultado["demanda_pico"], 0)
        self.assertGreater(resultado["saturacion_maxima_porcentaje"], 0)

        # Validar que los niveles de riesgo sean coherentes
        niveles_validos = {"BAJO", "MODERADO", "ALTO", "CRITICO"}
        for p in resultado["proyecciones"]:
            self.assertIn(p["nivel_riesgo"], niveles_validos)
            self.assertEqual(p["aforo_maximo"], 35)

    def test_impacto_semana_examenes(self):
        # Comparar misma fecha con y sin exámenes
        res_normal = self.predictor.predecir_demanda(
            fecha="2026-10-27", id_sala=2, capacidad_sala=40, es_semana_examenes=False
        )
        res_examenes = self.predictor.predecir_demanda(
            fecha="2026-10-27", id_sala=2, capacidad_sala=40, es_semana_examenes=True
        )

        # En exámenes la demanda pico debe ser mayor
        self.assertGreater(res_examenes["demanda_pico"], res_normal["demanda_pico"])
        self.assertGreater(
            res_examenes["saturacion_maxima_porcentaje"],
            res_normal["saturacion_maxima_porcentaje"]
        )


class TestFastAPIEndpoints(unittest.TestCase):
    """
    Pruebas HTTP sobre la API REST FastAPI con TestClient.
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/v1/ml/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.0.0")
        self.assertTrue(data["model_loaded"])

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "operational")
        self.assertIn("documentation", data)

    def test_endpoint_predecir_demanda(self):
        payload = {
            "id_sala": 10,
            "fecha": "2026-10-28",
            "capacidad_sala": 30,
            "es_semana_examenes": True,
            "hora_inicio": 8,
            "hora_fin": 20,
        }
        response = self.client.post("/api/v1/ml/predecir-demanda", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id_sala"], 10)
        self.assertEqual(data["fecha"], "2026-10-28")
        self.assertEqual(len(data["proyecciones"]), 13)  # 8 a 20 inclusive
        self.assertIn("recomendacion", data)
        self.assertGreater(data["saturacion_maxima_porcentaje"], 0)

    def test_endpoint_simular_colas_estable(self):
        payload = {
            "lambda_llegadas": 15.0,
            "mu_servicio": 1.0,
            "c_servidores": 20,
            "tolerancia_espera_min": 5.0,
        }
        response = self.client.post("/api/v1/ml/simular-colas", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["c"], 20)
        self.assertAlmostEqual(data["factor_utilizacion_rho"], 0.75, places=2)
        self.assertIn(data["estado_sistema"], ["OPTIMO", "MODERADO"])
        self.assertIsNotNone(data["tiempo_espera_cola_minutos"])

    def test_endpoint_simular_colas_saturado(self):
        payload = {
            "lambda_llegadas": 50.0,
            "mu_servicio": 1.0,
            "c_servidores": 30,
            "tolerancia_espera_min": 5.0,
        }
        response = self.client.post("/api/v1/ml/simular-colas", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["estado_sistema"], "SATURADO")
        self.assertAlmostEqual(data["prob_espera_erlang_c"], 1.0)
        self.assertIsNone(data["tiempo_espera_cola_minutos"])
        self.assertGreater(data["c_optimo_recomendado"], 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
