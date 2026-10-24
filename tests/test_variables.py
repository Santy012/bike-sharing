"""Pruebas de no fuga: alterar la demanda posterior al instante permitido no cambia nada."""

import numpy as np
import pandas as pd
import pytest

from bike_sharing import datos, variables

HORAS = ["2011-06-15 00:00", "2011-06-15 13:00", "2012-03-01 23:00", "2012-12-10 08:00"]


@pytest.fixture(scope="module")
def horaria():
    return datos.serie_horaria()


def _perturbar_desde(h: pd.DataFrame, instante) -> pd.DataFrame:
    """Copia con la demanda desde ``instante`` (incluido) reemplazada por ruido."""
    otra = h.copy()
    futuro = otra.index >= pd.Timestamp(instante)
    otra.loc[futuro, "cnt"] = np.random.default_rng(0).integers(0, 1000, futuro.sum())
    return otra


@pytest.mark.parametrize("t", HORAS)
def test_escenario_a_no_usa_la_hora_objetivo_ni_posteriores(horaria, t):
    original = variables.variables_a(horaria).loc[t]
    perturbada = variables.variables_a(_perturbar_desde(horaria, t)).loc[t]
    pd.testing.assert_series_equal(original, perturbada)


@pytest.mark.parametrize("t", HORAS)
def test_escenario_b_no_usa_nada_del_dia_objetivo(horaria, t):
    dia = pd.Timestamp(t).normalize()
    horas_del_dia = slice(dia, dia + pd.Timedelta(hours=23))
    original = variables.variables_b(horaria).loc[horas_del_dia]
    perturbada = variables.variables_b(_perturbar_desde(horaria, dia)).loc[horas_del_dia]
    pd.testing.assert_frame_equal(original, perturbada)


def test_escenario_b_si_usa_el_dia_anterior(horaria):
    # Control de la prueba anterior: perturbar el día previo sí debe cambiar las variables.
    dia = pd.Timestamp("2012-03-01")
    original = variables.variables_b(horaria).loc[dia]
    perturbada = variables.variables_b(_perturbar_desde(horaria, dia - pd.Timedelta(hours=1))).loc[dia]
    assert not original.equals(perturbada)


def test_rezagos_exactos(horaria):
    x = variables.variables_a(horaria)
    t = pd.Timestamp("2012-07-04 17:00")
    assert x.loc[t, "lag_1h"] == horaria.loc[t - pd.Timedelta(hours=1), "cnt"]
    assert x.loc[t, "lag_24h"] == horaria.loc[t - pd.Timedelta(days=1), "cnt"]
    assert x.loc[t, "lag_168h"] == horaria.loc[t - pd.Timedelta(days=7), "cnt"]
    ventana = horaria.loc[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1), "cnt"]
    assert x.loc[t, "media_24h"] == pytest.approx(ventana.mean())


def test_media_al_cierre_es_la_del_dia_anterior(horaria):
    x = variables.variables_b(horaria)
    dia = pd.Timestamp("2012-07-04")
    anterior = horaria.loc[dia - pd.Timedelta(days=1):dia - pd.Timedelta(hours=1), "cnt"]
    np.testing.assert_allclose(x.loc[dia:dia + pd.Timedelta(hours=23), "media_24h_cierre"], anterior.mean())


def test_fila_desde_historia_reproduce_variables_a(horaria):
    x = variables.variables_a(horaria)
    y = horaria["cnt"].to_numpy(dtype=float)
    for t in [200, 5000, 17543]:
        fila = variables.fila_a_desde_historia(y, t)
        for col in variables.columnas_dinamicas_a():
            assert fila[col] == pytest.approx(x.iloc[t][col])


@pytest.mark.parametrize("conjunto, n", [("completo", 6), ("clima", 5), ("calendario", 4)])
def test_exogenas_diarias(conjunto, n):
    x = variables.exogenas_diarias(datos.serie_diaria(), conjunto)
    assert x.shape == (731, n) and not x.isna().any().any()
