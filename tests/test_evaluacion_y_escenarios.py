import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from bike_sharing import datos, escenarios, estadisticos, evaluacion, variables


def test_metricas_casos_conocidos():
    y, p = np.array([10.0, 0.0, 30.0]), np.array([12.0, 1.0, 27.0])
    m = evaluacion.metricas(y, p)
    assert m["MAE"] == pytest.approx(2.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((4 + 1 + 9) / 3))
    assert m["MAPE"] == pytest.approx((0.2 + 0.1) / 2 * 100)  # ignora el real igual a cero
    assert m["WAPE"] == pytest.approx(6 / 40 * 100)


@pytest.fixture(scope="module")
def diaria():
    return datos.serie_diaria()


@pytest.fixture(scope="module")
def horaria():
    return datos.serie_horaria()


def test_dia_a_dia_no_usa_el_dia_objetivo(diaria):
    entrenamiento, _ = datos.particion(diaria["cnt"])
    res = estadisticos.ajustar(entrenamiento, (1, 1, 1), (0, 1, 1, 7))
    b = estadisticos.pronostico_dia_a_dia(res, diaria["cnt"])
    assert len(b) == 30
    # El pronóstico del día D debe coincidir con el de un modelo que solo vio hasta D − 1.
    for k in [0, 10, 29]:
        dia = b.index[k]
        hasta_ayer = res.apply(diaria["cnt"][diaria.index < dia], refit=False)
        assert b["media"].iloc[k] == pytest.approx(hasta_ayer.forecast(1).iloc[0])


def test_ingenuos_diarios_coinciden_con_la_suma_horaria(horaria, diaria):
    X_tr, y_tr, X_te, y_te = escenarios.conjunto_ml(variables.variables_b(horaria), horaria["cnt"])
    suma = escenarios.a_diario(pd.DataFrame({escenarios.PERSISTENCIA: X_te["lag_24h"],
                                             escenarios.ESTACIONAL: X_te["lag_168h"]}))
    ingenuos = escenarios.ingenuos_diarios(diaria)["B"]
    pd.testing.assert_frame_equal(suma, ingenuos, check_freq=False, check_dtype=False, check_names=False)


def test_recursivo_usa_sus_propias_predicciones(horaria):
    # Un modelo que copia lag_1h: en C debe repetir el último valor observado durante 720 horas.
    X_tr, y_tr, _, _ = escenarios.conjunto_ml(variables.variables_a(horaria), horaria["cnt"])
    columnas = list(X_tr.columns)
    pesos = np.array([1.0 if c == "lag_1h" else 0.0 for c in columnas])
    modelo = LinearRegression().fit(X_tr.iloc[:10], X_tr.iloc[:10] @ pesos)
    modelo.coef_, modelo.intercept_ = pesos, 0.0
    c = escenarios.pronostico_recursivo(modelo, horaria)
    ultimo = horaria.loc[horaria.index < datos.INICIO_PRUEBA, "cnt"].iloc[-1]
    assert len(c) == 720
    np.testing.assert_allclose(c.to_numpy(), ultimo)
