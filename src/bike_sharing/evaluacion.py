"""Métricas de error.

El MAPE se dispara cuando la demanda real es baja (madrugadas, Navidad); por eso se
reporta también el WAPE, el error absoluto total dividido entre la demanda total, que es
estable y comparable entre series horarias y diarias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p) -> float:
    """MAPE en %, sobre las observaciones con demanda real distinta de cero."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def wape(y, p) -> float:
    """WAPE en %: ``Σ|y − p| / Σ|y|``."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def metricas(y, p) -> dict[str, float]:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "WAPE": wape(y, p)}


def tabla(reales: pd.Series, predicciones: pd.DataFrame) -> pd.DataFrame:
    """Una fila de métricas por columna de ``predicciones``, ordenada por WAPE."""
    filas = {nombre: metricas(reales, predicciones[nombre].reindex(reales.index)) for nombre in predicciones}
    return pd.DataFrame(filas).T.sort_values("WAPE")


def por_grupo(reales: pd.Series, prediccion: pd.Series, grupos: pd.Series) -> pd.DataFrame:
    """Métricas de una predicción separadas por grupo (tipo de día, estación…)."""
    filas = {}
    for g in grupos.unique():
        m = (grupos == g).to_numpy()
        filas[g] = {**metricas(reales[m], prediccion[m]), "n": int(m.sum())}
    return pd.DataFrame(filas).T
