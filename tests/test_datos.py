import numpy as np
import pandas as pd
import pytest

from bike_sharing import datos


@pytest.fixture(scope="module")
def crudos():
    return datos.cargar_crudos()


@pytest.fixture(scope="module")
def horaria():
    return datos.serie_horaria()


@pytest.fixture(scope="module")
def diaria():
    return datos.serie_diaria()


def test_csv_verificados():
    assert datos._verificados(datos.descargar_datos())


def test_integridad_de_los_crudos(crudos):
    r = datos.revisar_integridad(*crudos)
    assert r["dias_faltantes"] == 0 and r["dias_duplicados"] == 0
    assert r["horas_esperadas"] == 17544 and r["horas_presentes"] == 17379
    assert len(r["horas_faltantes"]) == 165 and r["horas_duplicadas"] == 0
    assert r["nulos_day"] == 0 and r["nulos_hour"] == 0
    assert r["dias_cnt_distinto"] == 0
    assert r["casual_mas_registered"]
    assert all(r["calendario_day"].values()) and all(r["calendario_hour"].values())


def test_rejilla_horaria_completa_y_regular(horaria):
    assert len(horaria) == 17544
    assert horaria.index.freq == "h" or (horaria.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert not horaria.isna().any().any()


def test_imputacion_de_demanda(horaria, crudos):
    day, _ = crudos
    assert horaria["imputada"].sum() == 165
    assert (horaria.loc[horaria["imputada"], "cnt"] == 0).all()
    np.testing.assert_array_equal(horaria["cnt"].resample("D").sum().to_numpy(), day["cnt"].to_numpy())


def test_calendario_horario_coincide_con_el_diario(horaria, crudos):
    day, _ = crudos
    fechas = horaria.index.normalize()
    for col in ["season", "holiday", "workingday", "weekday", "yr", "mnth"]:
        np.testing.assert_array_equal(horaria[col].to_numpy(), day[col].reindex(fechas).to_numpy())


def test_humedad_corregida(horaria, diaria):
    assert (horaria["hum_pct"] > 0).all()
    assert (diaria["hum_pct"] > 0).all()
    vecinos = diaria.loc[["2011-03-09", "2011-03-11"], "hum_pct"].mean()
    assert diaria.loc["2011-03-10", "hum_pct"] == pytest.approx(vecinos)


def test_desnormalizacion_en_rangos_fisicos(horaria):
    assert horaria["temp_c"].between(-8, 39).all()
    assert horaria["atemp_c"].between(-16, 50).all()
    assert horaria["hum_pct"].between(0, 100).all()
    assert horaria["viento_kmh"].between(0, 67).all()


def test_particion(diaria, horaria):
    entrenamiento, prueba = datos.particion(diaria)
    assert len(prueba) == 30 and prueba.index[0] == datos.INICIO_PRUEBA
    assert entrenamiento.index[-1] == datos.INICIO_PRUEBA - pd.Timedelta(days=1)
    assert len(datos.particion(horaria)[1]) == 720
