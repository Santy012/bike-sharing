"""Ajusta todos los modelos, corre los tres escenarios y guarda los resultados.

Uso: ``python scripts/run.py`` (unos 25 minutos en un portátil; el SVR es lo más lento).
``python scripts/run.py --solo-resumen`` rehace solo ``por_grupo.json`` desde las
predicciones guardadas. Escribe en ``results/``:

- ``escenario_a.json``, ``escenario_b.json``, ``escenario_c.json``: una tabla de métricas por
  escenario (y, en B y C, por nivel: diario para todos, horario para ML).
- ``estadisticos.json``: órdenes, AIC, BIC, Ljung-Box y coeficientes.
- ``validacion_cruzada.json``: ``TimeSeriesSplit`` de 5 particiones y búsqueda de hiperparámetros.
- ``por_grupo.json``: error por tipo de día, estación y periodo (antes y durante las fiestas).
- ``predicciones_*.csv`` y ``oof_*.csv``: predicciones para las figuras de los notebooks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from bike_sharing import datos, escenarios, estadisticos, evaluacion, ml, variables

RESULTADOS = Path(__file__).resolve().parents[1] / "results"


def guardar_json(nombre: str, contenido: dict) -> None:
    (RESULTADOS / nombre).write_text(json.dumps(contenido, indent=2, ensure_ascii=False, default=float),
                                     encoding="utf-8")


def como_dict(df: pd.DataFrame) -> dict:
    return {str(k): v for k, v in df.to_dict(orient="index").items()}


def tipo_de_dia(indice: pd.DatetimeIndex, day: pd.DataFrame) -> pd.Series:
    laboral = day["workingday"].reindex(indice.normalize()).to_numpy() == 1
    return pd.Series(["laboral" if x else "fin de semana o festivo" for x in laboral], index=indice)


def estacion(indice: pd.DatetimeIndex, day: pd.DataFrame) -> pd.Series:
    nombres = {1: "invierno", 2: "primavera", 3: "verano", 4: "otoño"}
    return pd.Series(day["season"].reindex(indice.normalize()).map(nombres).to_numpy(), index=indice)


def main() -> None:
    RESULTADOS.mkdir(exist_ok=True)
    inicio = time.time()
    day, h = datos.serie_diaria(), datos.serie_horaria()
    day_tr, _ = datos.particion(day)

    # ------------------------------------------------ modelos estadísticos (serie diaria)
    auto = estadisticos.auto_arima(day_tr["cnt"])
    ajustes, diagnosticos, coefs = {}, {}, {}
    for e in estadisticos.especificaciones(auto.order):
        exog = variables.exogenas_diarias(day_tr, e.exogenas) if e.exogenas else None
        res = estadisticos.ajustar(day_tr["cnt"], e.orden, e.orden_estacional, exog)
        ajustes[e.nombre] = (res, e.exogenas)
        diagnosticos[e.nombre] = {"orden": e.orden, "orden_estacional": e.orden_estacional,
                                  **estadisticos.diagnostico(res)}
        coefs[e.nombre] = como_dict(estadisticos.coeficientes(res))
    guardar_json("estadisticos.json", {"orden_auto_arima": list(auto.order),
                                       "diagnostico": diagnosticos, "coeficientes": coefs})
    print(f"Estadísticos ajustados ({time.time() - inicio:.0f} s)")

    # ------------------------------------------------ ML: validación temporal y ajuste
    modelos, cv_json, oof = {}, {}, {}
    for esc, construir in (("A", variables.variables_a), ("B", variables.variables_b)):
        X_tr, y_tr, _, _ = escenarios.conjunto_ml(construir(h), h["cnt"])
        cv, oof_esc = ml.validacion_cruzada(ml.modelos_base(), X_tr, y_tr)
        mejor = cv.index[0]
        ajustado, params, mae_busqueda = ml.buscar_hiperparametros(mejor, X_tr, y_tr)
        nombre_ajustado = f"{mejor} ajustado"
        cv_aj, oof_aj = ml.validacion_cruzada({nombre_ajustado: ajustado}, X_tr, y_tr)
        cv = pd.concat([cv, cv_aj]).sort_values("MAE_media")
        modelos[esc] = {**ml.modelos_base(), nombre_ajustado: ajustado}
        oof[esc] = pd.concat([y_tr.rename("real"), oof_esc, oof_aj], axis=1).dropna()
        cv_json[esc] = {"metricas": como_dict(cv), "mejor_base": mejor,
                        "hiperparametros": params, "MAE_busqueda": mae_busqueda}
        print(f"Validación cruzada {esc}: mejor {mejor} ({time.time() - inicio:.0f} s)")
    guardar_json("validacion_cruzada.json", cv_json)

    # ------------------------------------------------ escenario A: hora siguiente
    pred_a, ajustados_a = escenarios.escenario_a(h, modelos["A"])
    tabla_a = evaluacion.tabla(pred_a["real"], pred_a.drop(columns="real"))
    guardar_json("escenario_a.json", {"nivel": "hora", "metricas": como_dict(tabla_a)})
    print(f"Escenario A ({time.time() - inicio:.0f} s)")

    # ------------------------------------------------ escenario B: día siguiente
    pred_b_h = escenarios.escenario_b_ml(h, modelos["B"])
    est_b = escenarios.escenario_b_estadisticos(day, ajustes)
    pred_b_d = escenarios.a_diario(pred_b_h)
    for nombre, p in est_b.items():
        pred_b_d[nombre] = p["media"].clip(lower=0)
    guardar_json("escenario_b.json", {
        "diario": como_dict(evaluacion.tabla(pred_b_d["real"], pred_b_d.drop(columns="real"))),
        "horario": como_dict(evaluacion.tabla(pred_b_h["real"], pred_b_h.drop(columns="real"))),
    })
    print(f"Escenario B ({time.time() - inicio:.0f} s)")

    # ------------------------------------------------ escenario C: mes completo
    pred_c_h = escenarios.escenario_c_ml(h, ajustados_a)
    est_c = escenarios.escenario_c_estadisticos(day, ajustes)
    pred_c_d = escenarios.a_diario(pred_c_h)
    for nombre, p in est_c.items():
        pred_c_d[nombre] = p["media"].clip(lower=0)
    guardar_json("escenario_c.json", {
        "diario": como_dict(evaluacion.tabla(pred_c_d["real"], pred_c_d.drop(columns="real"))),
        "horario": como_dict(evaluacion.tabla(pred_c_h["real"], pred_c_h.drop(columns="real"))),
    })
    print(f"Escenario C ({time.time() - inicio:.0f} s)")

    # ------------------------------------------------ predicciones para los notebooks
    pred_a.to_csv(RESULTADOS / "predicciones_a_horaria.csv")
    pred_b_h.to_csv(RESULTADOS / "predicciones_b_horaria.csv")
    pred_b_d.to_csv(RESULTADOS / "predicciones_b_diaria.csv")
    pred_c_h.to_csv(RESULTADOS / "predicciones_c_horaria.csv")
    pred_c_d.to_csv(RESULTADOS / "predicciones_c_diaria.csv")
    for esc, est in (("b", est_b), ("c", est_c)):
        pd.concat(est, axis=1).to_csv(RESULTADOS / f"intervalos_{esc}_estadisticos.csv")
    for esc, o in oof.items():
        o.to_csv(RESULTADOS / f"oof_{esc.lower()}.csv")
    resumir_por_grupo(day)
    print(f"Listo en {time.time() - inicio:.0f} s")


def periodo(indice: pd.DatetimeIndex, _day=None) -> pd.Series:
    """Prueba antes de las fiestas y semana de fiestas (desde el sábado 22 de diciembre)."""
    fiestas = indice >= pd.Timestamp("2012-12-22")
    return pd.Series(["22 a 31 dic" if x else "2 a 21 dic" for x in fiestas], index=indice)


def resumir_por_grupo(day: pd.DataFrame) -> None:
    """Error por tipo de día, estación y periodo a partir de las predicciones guardadas."""
    leer = lambda n: pd.read_csv(RESULTADOS / n, index_col=0, parse_dates=True)
    grupos = {}
    for esc, archivo in (("A", "predicciones_a_horaria.csv"), ("B", "predicciones_b_diaria.csv"),
                         ("C", "predicciones_c_diaria.csv")):
        pred = leer(archivo)
        for nombre_grupo, f in (("tipo_de_dia", tipo_de_dia), ("periodo", periodo)):
            grupos[f"prueba_{esc}_{nombre_grupo}"] = {
                m: como_dict(evaluacion.por_grupo(pred["real"], pred[m], f(pred.index, day)))
                for m in pred.columns if m != "real"}
    for esc in ("A", "B"):
        o = leer(f"oof_{esc.lower()}.csv")
        for nombre_grupo, f in (("tipo_de_dia", tipo_de_dia), ("estacion", estacion)):
            grupos[f"validacion_{esc}_{nombre_grupo}"] = {
                m: como_dict(evaluacion.por_grupo(o["real"], o[m], f(o.index, day)))
                for m in o.columns if m != "real"}
    guardar_json("por_grupo.json", grupos)


if __name__ == "__main__":
    import sys

    if "--solo-resumen" in sys.argv:
        resumir_por_grupo(datos.serie_diaria())
    else:
        main()
