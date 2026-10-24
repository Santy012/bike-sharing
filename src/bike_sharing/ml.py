"""Modelos de machine learning sobre la serie horaria, validación temporal y ajuste.

La validación usa ``TimeSeriesSplit`` de 5 particiones: cada partición entrena con el
pasado y valida con el bloque siguiente. Un ``train_test_split`` aleatorio mezclaría horas
futuras en el entrenamiento y, con rezagos de 1 h, el modelo vería casi el mismo dato que
debe predecir; el error de validación saldría optimista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import RANDOM_STATE
from .evaluacion import metricas

N_PARTICIONES = 5
N_ITERACIONES_BUSQUEDA = 20


def modelos_base() -> dict:
    """Ridge y SVR con variables estandarizadas; los árboles no lo necesitan.

    El SVR también estandariza el objetivo: con ``C = 1`` y la demanda en su escala
    original (cientos de alquileres por hora) el margen quedaría demasiado rígido.
    """
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            transformer=StandardScaler(),
        ),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0),
    }


ESPACIOS = {
    "XGBoost": {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.8, 1.0],
    },
    "SVR": {
        "regressor__svr__C": [0.3, 1, 3, 10],
        "regressor__svr__epsilon": [0.01, 0.05, 0.1],
        "regressor__svr__gamma": ["scale", 0.01, 0.03],
    },
    "Ridge": {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
}


def validacion_cruzada(modelos: dict, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas medias y desviaciones por modelo y predicciones fuera de partición."""
    tscv = TimeSeriesSplit(n_splits=N_PARTICIONES)
    filas, oof = {}, pd.DataFrame(index=y.index, columns=list(modelos), dtype=float)
    for nombre, modelo in modelos.items():
        por_particion = []
        for i_tr, i_va in tscv.split(X):
            m = clone(modelo).fit(X.iloc[i_tr], y.iloc[i_tr])
            p = np.clip(m.predict(X.iloc[i_va]), 0, None)
            oof.iloc[i_va, oof.columns.get_loc(nombre)] = p
            por_particion.append(metricas(y.iloc[i_va], p))
        df = pd.DataFrame(por_particion)
        filas[nombre] = {**{f"{k}_media": v for k, v in df.mean().items()},
                         **{f"{k}_desv": v for k, v in df.std().items()}}
    return pd.DataFrame(filas).T.sort_values("MAE_media"), oof.dropna()


def buscar_hiperparametros(nombre: str, X: pd.DataFrame, y: pd.Series):
    """Búsqueda aleatoria con la misma validación temporal; criterio: MAE."""
    busqueda = RandomizedSearchCV(
        modelos_base()[nombre], ESPACIOS[nombre], n_iter=N_ITERACIONES_BUSQUEDA,
        cv=TimeSeriesSplit(n_splits=N_PARTICIONES), scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=1, refit=False,
    )
    busqueda.fit(X, y)
    modelo = clone(modelos_base()[nombre]).set_params(**busqueda.best_params_)
    return modelo, busqueda.best_params_, -busqueda.best_score_
