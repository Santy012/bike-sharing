"""Variables de entrada por escenario, construidas solo con información disponible.

- Escenario A (hora siguiente): para la hora ``t`` se conoce la demanda hasta ``t − 1``.
- Escenario B (día siguiente): para las 24 horas del día ``D`` se conoce la demanda hasta
  las 23:00 de ``D − 1``.
- Escenario C (mes completo): se usan las variables de A, alimentadas con las propias
  predicciones a partir del origen del pronóstico (ver ``fila_a_desde_historia``).

El clima y el calendario de la hora objetivo se suponen conocidos en todos los escenarios.
Los rezagos con ``shift`` son correctos porque la serie horaria es una rejilla completa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OBJETIVO = "cnt"
REZAGOS_A = (1, 24, 168)
REZAGOS_B = (24, 168)
VENTANAS = (24, 168)

CLIMA_HORARIO = ["temp_c", "hum_pct", "viento_kmh", "weathersit"]
CALENDARIO_HORARIO = ["hora", "dia_semana", "mes", "yr", "season", "holiday", "workingday",
                      "es_finde", "dia_anio", "hora_sin", "hora_cos", "mes_sin", "mes_cos"]


def exogenas_horarias(h: pd.DataFrame) -> pd.DataFrame:
    """Calendario y clima de cada hora (conocidos de antemano por supuesto)."""
    idx = h.index
    x = pd.DataFrame(index=idx)
    x["hora"] = idx.hour
    x["dia_semana"] = idx.day_of_week  # 0 = lunes
    x["mes"] = idx.month
    x["yr"] = h["yr"]
    x["season"] = h["season"]
    x["holiday"] = h["holiday"]
    x["workingday"] = h["workingday"]
    x["es_finde"] = (idx.day_of_week >= 5).astype(int)
    x["dia_anio"] = idx.day_of_year
    x["hora_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    x["hora_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    x["mes_sin"] = np.sin(2 * np.pi * idx.month / 12)
    x["mes_cos"] = np.cos(2 * np.pi * idx.month / 12)
    for col in CLIMA_HORARIO:
        x[col] = h[col]
    return x


def variables_a(h: pd.DataFrame) -> pd.DataFrame:
    """Escenario A: rezagos de 1, 24 y 168 h y estadísticas móviles hasta ``t − 1``."""
    x = exogenas_horarias(h)
    y = h[OBJETIVO].astype(float)
    for k in REZAGOS_A:
        x[f"lag_{k}h"] = y.shift(k)
    pasado = y.shift(1)  # la ventana termina en t − 1: nunca incluye la hora a predecir
    for w in VENTANAS:
        x[f"media_{w}h"] = pasado.rolling(w).mean()
        x[f"desv_{w}h"] = pasado.rolling(w).std()
    return x


def variables_b(h: pd.DataFrame) -> pd.DataFrame:
    """Escenario B: solo información disponible al cierre del día anterior.

    La hora objetivo está entre 1 y 24 horas después del cierre, así que los rezagos de 24
    y 168 h siempre caen antes de él. Las estadísticas móviles se calculan a las 23:00 del
    día anterior y se repiten en las 24 horas del día.
    """
    x = exogenas_horarias(h)
    y = h[OBJETIVO].astype(float)
    for k in REZAGOS_B:
        x[f"lag_{k}h"] = y.shift(k)
    cierre = h.index.normalize() - pd.Timedelta(hours=1)
    for w in VENTANAS:
        x[f"media_{w}h_cierre"] = y.rolling(w).mean().reindex(cierre).to_numpy()
        x[f"desv_{w}h_cierre"] = y.rolling(w).std().reindex(cierre).to_numpy()
    return x


def columnas_dinamicas_a() -> list[str]:
    return [f"lag_{k}h" for k in REZAGOS_A] + [f"{s}_{w}h" for w in VENTANAS for s in ("media", "desv")]


def fila_a_desde_historia(historia: np.ndarray, t: int) -> dict[str, float]:
    """Variables dinámicas de A para la posición ``t`` usando ``historia[:t]``.

    En el escenario C, ``historia`` contiene la demanda real hasta el origen y las
    predicciones propias después; así el pronóstico recursivo usa las mismas variables
    con las que se entrenó el modelo.
    """
    fila = {f"lag_{k}h": historia[t - k] for k in REZAGOS_A}
    for w in VENTANAS:
        ventana = historia[t - w:t]
        fila[f"media_{w}h"] = ventana.mean()
        fila[f"desv_{w}h"] = ventana.std(ddof=1)
    return fila


def exogenas_diarias(day: pd.DataFrame, conjunto: str = "completo") -> pd.DataFrame:
    """Exógenas para SARIMAX sobre la serie diaria.

    - ``clima``: temperatura, humedad, viento e indicadoras de ``weathersit`` 2 (nublado) y
      3 (lluvia o nieve ligera). ``atemp_c`` se excluye por su colinealidad con ``temp_c``.
    - ``calendario``: festivo, laboral e indicadoras de estación (base: invierno).
      El día de la semana ya lo recoge la parte estacional de periodo 7.
    - ``completo``: clima + festivo + laboral.
    """
    clima = pd.DataFrame({
        "temp_c": day["temp_c"],
        "hum_pct": day["hum_pct"],
        "viento_kmh": day["viento_kmh"],
        "weather_2": (day["weathersit"] == 2).astype(float),
        "weather_3": (day["weathersit"] >= 3).astype(float),
    })
    calendario = pd.DataFrame({
        "holiday": day["holiday"].astype(float),
        "workingday": day["workingday"].astype(float),
    })
    estaciones = pd.DataFrame({f"season_{s}": (day["season"] == s).astype(float) for s in (2, 3, 4)})
    if conjunto == "clima":
        return clima
    if conjunto == "calendario":
        return pd.concat([calendario, estaciones], axis=1)
    if conjunto == "completo":
        return pd.concat([clima, calendario], axis=1)
    raise ValueError(f"Conjunto de exógenas desconocido: {conjunto}")
