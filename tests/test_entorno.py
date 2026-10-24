"""Prueba de humo: el paquete importa y el stack estadístico y de ML funciona."""

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import bike_sharing


def test_paquete_importa():
    assert bike_sharing.RANDOM_STATE == 42


def test_sarimax_y_xgboost_ajustan():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=120))
    res = SARIMAX(y, order=(1, 1, 0)).fit(disp=False)
    assert np.isfinite(res.forecast(5)).all()

    X = rng.normal(size=(200, 3))
    modelo = XGBRegressor(n_estimators=5, max_depth=2).fit(X, X[:, 0])
    assert modelo.predict(X).shape == (200,)
