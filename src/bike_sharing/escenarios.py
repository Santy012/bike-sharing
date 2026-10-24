"""Los tres escenarios de pronóstico sobre los 30 días de prueba (2012-12-02 a 2012-12-31).

| Escenario | Información disponible | Nivel de evaluación |
| --- | --- | --- |
| A. Hora siguiente | demanda real hasta ``t − 1`` | hora |
| B. Día siguiente | demanda real hasta el cierre del día anterior | día (y hora para ML) |
| C. Mes completo | demanda real hasta el 2012-12-01 | día (y hora para ML) |

Cada escenario incluye dos referencias ingenuas. Las predicciones negativas se recortan a
cero en todos los modelos: la demanda no puede ser negativa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import estadisticos, variables
from .datos import INICIO_PRUEBA, particion

PERSISTENCIA = "Ingenuo: persistencia"
ESTACIONAL = "Ingenuo: estacional"


def conjunto_ml(X: pd.DataFrame, y: pd.Series):
    """Quita las filas iniciales sin historia suficiente y divide en entrenamiento y prueba."""
    validas = X.notna().all(axis=1)
    X, y = X[validas], y[validas]
    (X_tr, X_te), (y_tr, y_te) = particion(X), particion(y)
    return X_tr, y_tr, X_te, y_te


def a_diario(horaria: pd.DataFrame | pd.Series):
    return horaria.resample("D").sum()


# ---------------------------------------------------------------- escenario A
def escenario_a(h: pd.DataFrame, modelos: dict) -> tuple[pd.DataFrame, dict]:
    """Predicciones horarias de prueba con todos los rezagos. Devuelve también los modelos ajustados."""
    X_tr, y_tr, X_te, y_te = conjunto_ml(variables.variables_a(h), h["cnt"])
    pred = pd.DataFrame(index=X_te.index)
    ajustados = {}
    for nombre, modelo in modelos.items():
        ajustados[nombre] = clone(modelo).fit(X_tr, y_tr)
        pred[nombre] = np.clip(ajustados[nombre].predict(X_te), 0, None)
    pred[PERSISTENCIA] = X_te["lag_1h"]
    pred[ESTACIONAL] = X_te["lag_24h"]
    pred.insert(0, "real", y_te)
    return pred, ajustados


# ---------------------------------------------------------------- escenario B
def escenario_b_ml(h: pd.DataFrame, modelos: dict) -> pd.DataFrame:
    """Predicciones horarias del día siguiente con información al cierre del día anterior."""
    X_tr, y_tr, X_te, y_te = conjunto_ml(variables.variables_b(h), h["cnt"])
    pred = pd.DataFrame(index=X_te.index)
    for nombre, modelo in modelos.items():
        pred[nombre] = np.clip(clone(modelo).fit(X_tr, y_tr).predict(X_te), 0, None)
    pred[PERSISTENCIA] = X_te["lag_24h"]  # misma hora de ayer: suma diaria = total de ayer
    pred[ESTACIONAL] = X_te["lag_168h"]  # misma hora de la semana pasada
    pred.insert(0, "real", y_te)
    return pred


def escenario_b_estadisticos(day: pd.DataFrame, ajustes: dict) -> dict[str, pd.DataFrame]:
    """Pronóstico a un día de cada día de prueba para cada modelo estadístico ajustado."""
    salida = {}
    for nombre, (res, conjunto) in ajustes.items():
        exog = variables.exogenas_diarias(day, conjunto) if conjunto else None
        salida[nombre] = estadisticos.pronostico_dia_a_dia(res, day["cnt"], exog)
    return salida


# ---------------------------------------------------------------- escenario C
def pronostico_recursivo(modelo, h: pd.DataFrame) -> pd.Series:
    """Escenario C para ML: 720 pasos hora a hora con las predicciones propias como rezagos."""
    if hasattr(modelo, "n_jobs"):
        modelo = modelo.set_params(n_jobs=1)  # una fila por llamada: el paralelismo solo estorba
    exo = variables.exogenas_horarias(h)
    columnas = list(exo.columns) + variables.columnas_dinamicas_a()
    historia = h["cnt"].to_numpy(dtype=float).copy()
    inicio = h.index.get_loc(INICIO_PRUEBA)
    historia[inicio:] = np.nan
    exo_np = exo.to_numpy(dtype=float)
    n_exo = exo_np.shape[1]
    fila = np.empty((1, len(columnas)))
    for t in range(inicio, len(h)):
        fila[0, :n_exo] = exo_np[t]
        fila[0, n_exo:] = list(variables.fila_a_desde_historia(historia, t).values())
        historia[t] = max(float(modelo.predict(pd.DataFrame(fila, columns=columnas))[0]), 0.0)
    return pd.Series(historia[inicio:], index=h.index[inicio:])


def escenario_c_ml(h: pd.DataFrame, ajustados_a: dict) -> pd.DataFrame:
    _, prueba = particion(h)
    pred = pd.DataFrame(index=prueba.index)
    for nombre, modelo in ajustados_a.items():
        pred[nombre] = pronostico_recursivo(modelo, h)
    # Referencias: repetir el último día observado y la última semana observada.
    entrenamiento = h.loc[h.index < INICIO_PRUEBA, "cnt"].to_numpy(dtype=float)
    pred[PERSISTENCIA] = np.resize(entrenamiento[-24:], len(prueba))
    pred[ESTACIONAL] = np.resize(entrenamiento[-168:], len(prueba))
    pred.insert(0, "real", prueba["cnt"])
    return pred


def escenario_c_estadisticos(day: pd.DataFrame, ajustes: dict) -> dict[str, pd.DataFrame]:
    _, prueba = particion(day)
    salida = {}
    for nombre, (res, conjunto) in ajustes.items():
        exog = variables.exogenas_diarias(prueba, conjunto) if conjunto else None
        salida[nombre] = estadisticos.pronostico_mes(res, exog, pasos=len(prueba))
    return salida


def ingenuos_diarios(day: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Referencias diarias de B (ayer y hace una semana) y C (repetición desde el origen)."""
    y = day["cnt"].astype(float)
    _, prueba = particion(y)
    entrenamiento = y[y.index < INICIO_PRUEBA].to_numpy()
    return {
        "B": pd.DataFrame({PERSISTENCIA: y.shift(1), ESTACIONAL: y.shift(7)}).loc[prueba.index],
        "C": pd.DataFrame({PERSISTENCIA: np.repeat(entrenamiento[-1], len(prueba)),
                           ESTACIONAL: np.resize(entrenamiento[-7:], len(prueba))}, index=prueba.index),
    }
