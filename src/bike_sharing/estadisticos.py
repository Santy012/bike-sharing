"""ARIMA, SARIMA y SARIMAX sobre la serie diaria (``statsmodels`` y ``pmdarima``).

Los pronósticos de los escenarios B y C usan los mismos parámetros, estimados una sola
vez con el entrenamiento:

- B (día siguiente): el filtro de Kalman incorpora cada día observado y pronostica el
  siguiente, sin reestimar parámetros (``results.apply``).
- C (mes completo): pronóstico a 30 pasos desde el 2012-12-01.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .datos import INICIO_PRUEBA

ORDEN_ARIMA = (1, 1, 1)
# Elegido por AIC y BIC entre p, q ∈ {0, 1, 2} y P, Q ∈ {0, 1} con d = D = 1 y s = 7
# (ver notebook 02); la ACF de la serie diferenciada respalda Q = 1 y P = 0.
ORDEN_SARIMA = (2, 1, 1)
ORDEN_ESTACIONAL = (0, 1, 1, 7)
NIVEL_INTERVALO = 0.05


@dataclass
class Especificacion:
    nombre: str
    orden: tuple
    orden_estacional: tuple = (0, 0, 0, 0)
    exogenas: str | None = None  # "completo", "clima", "calendario" o None


def especificaciones(orden_auto: tuple) -> list[Especificacion]:
    """Los modelos estadísticos que entran en los escenarios B y C."""
    return [
        Especificacion(f"ARIMA{ORDEN_ARIMA}", ORDEN_ARIMA),
        Especificacion(f"ARIMA{tuple(orden_auto)} auto", tuple(orden_auto)),
        Especificacion("SARIMA", ORDEN_SARIMA, ORDEN_ESTACIONAL),
        Especificacion("SARIMAX completo", ORDEN_SARIMA, ORDEN_ESTACIONAL, "completo"),
        Especificacion("SARIMAX solo clima", ORDEN_SARIMA, ORDEN_ESTACIONAL, "clima"),
        Especificacion("SARIMAX solo calendario", ORDEN_SARIMA, ORDEN_ESTACIONAL, "calendario"),
    ]


def ajustar(y: pd.Series, orden, orden_estacional=(0, 0, 0, 0), exog: pd.DataFrame | None = None):
    """Ajusta un SARIMAX por máxima verosimilitud con las restricciones por defecto."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=orden, seasonal_order=orden_estacional).fit(disp=False, maxiter=200)


def auto_arima(y: pd.Series):
    """Búsqueda escalonada de ``pmdarima`` sin estacionalidad, con d = 1 (por la prueba ADF)."""
    return pm.auto_arima(y, d=1, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False,
                         stepwise=True, suppress_warnings=True, error_action="ignore")


def diagnostico(res, rezagos=(7, 14)) -> dict:
    """AIC, BIC y Ljung-Box de los residuos (sin los primeros días de inicialización)."""
    burn = max(res.loglikelihood_burn, 8)
    lb = acorr_ljungbox(res.resid.iloc[burn:], lags=list(rezagos), return_df=True)
    return {
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        **{f"LjungBox_p_{k}": float(lb.loc[k, "lb_pvalue"]) for k in rezagos},
    }


def coeficientes(res) -> pd.DataFrame:
    return pd.DataFrame({"coef": res.params, "error_std": res.bse, "p_valor": res.pvalues})


def _intervalos(pred) -> pd.DataFrame:
    ci = pred.conf_int(alpha=NIVEL_INTERVALO)
    return pd.DataFrame({"media": pred.predicted_mean, "inferior": ci.iloc[:, 0], "superior": ci.iloc[:, 1]})


def pronostico_dia_a_dia(res, y_total: pd.Series, exog_total: pd.DataFrame | None = None) -> pd.DataFrame:
    """Escenario B: pronóstico a un paso de cada día de prueba con los parámetros fijos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actualizado = res.apply(y_total, exog=exog_total, refit=False)
    return _intervalos(actualizado.get_prediction(start=INICIO_PRUEBA)).loc[INICIO_PRUEBA:]


def pronostico_mes(res, exog_prueba: pd.DataFrame | None = None, pasos: int = 30) -> pd.DataFrame:
    """Escenario C: pronóstico a 30 días desde el final del entrenamiento."""
    return _intervalos(res.get_forecast(steps=pasos, exog=exog_prueba))
