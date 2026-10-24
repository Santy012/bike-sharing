"""Descarga, verificaciÃ³n, integridad e imputaciÃ³n del dataset Bike Sharing de UCI.

Produce dos series limpias: la diaria (``day.csv``) y la horaria sobre una rejilla
completa de 17.544 horas (``hour.csv`` trae 17.379; las 165 que faltan son horas
sin alquileres).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL_UCI = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = {
    "day.csv": "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be",
    "hour.csv": "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f",
}
RAIZ = Path(__file__).resolve().parents[2]
DIRECTORIO_DATOS = RAIZ / "data"
# Copia local de respaldo; solo existe en el equipo de trabajo (ignorada por Git).
DIRECTORIO_RESPALDO = RAIZ / "_privado" / "originales" / "bike+sharing+dataset"

# ParticiÃ³n temporal: los Ãºltimos 30 dÃ­as son la prueba de todos los escenarios.
INICIO_PRUEBA = pd.Timestamp("2012-12-02")
FIN_PRUEBA = pd.Timestamp("2012-12-31")

# DesnormalizaciÃ³n segÃºn el Readme de UCI: valor = normalizado Â· factor + desfase.
DESNORMALIZACION = {
    "temp": ("temp_c", 47.0, -8.0),
    "atemp": ("atemp_c", 66.0, -16.0),
    "hum": ("hum_pct", 100.0, 0.0),
    "windspeed": ("viento_kmh", 67.0, 0.0),
}
CLIMA_CONTINUO = list(DESNORMALIZACION)
DEMANDA = ["casual", "registered", "cnt"]
FECHA_HUMEDAD_CERO = pd.Timestamp("2011-03-10")


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _verificados(directorio: Path) -> bool:
    return all((directorio / n).exists() and _sha256(directorio / n) == h for n, h in SHA256.items())


def descargar_datos(directorio: Path = DIRECTORIO_DATOS) -> Path:
    """Deja ``day.csv`` y ``hour.csv`` verificados en ``directorio``.

    Descarga el zip de UCI; si falla, copia los CSV de respaldo locales.
    """
    directorio = Path(directorio)
    if _verificados(directorio):
        return directorio
    directorio.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL_UCI, timeout=60) as r:
            contenido = r.read()
        with zipfile.ZipFile(io.BytesIO(contenido)) as z:
            for nombre in SHA256:
                (directorio / nombre).write_bytes(z.read(nombre))
    except OSError:
        if not _verificados(DIRECTORIO_RESPALDO):
            raise
        for nombre in SHA256:
            shutil.copyfile(DIRECTORIO_RESPALDO / nombre, directorio / nombre)
    if not _verificados(directorio):
        raise ValueError(f"SHA-256 inesperado en {directorio}; borre los CSV y vuelva a descargarlos.")
    return directorio


def cargar_crudos(directorio: Path = DIRECTORIO_DATOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve ``(day, hour)`` sin modificar, indexados por fecha y por fecha-hora."""
    directorio = descargar_datos(directorio)
    day = pd.read_csv(directorio / "day.csv", parse_dates=["dteday"]).set_index("dteday")
    hour = pd.read_csv(directorio / "hour.csv", parse_dates=["dteday"])
    hour.index = pd.DatetimeIndex(hour["dteday"] + pd.to_timedelta(hour["hr"], unit="h"), name="timestamp")
    return day, hour


def revisar_integridad(day: pd.DataFrame, hour: pd.DataFrame) -> dict:
    """Resume la integridad temporal y la coherencia entre los dos archivos."""
    dias = pd.date_range(day.index.min(), day.index.max(), freq="D")
    horas = pd.date_range(hour.index.min().normalize(), day.index.max() + pd.Timedelta(hours=23), freq="h")
    faltantes = horas.difference(hour.index)
    suma_horaria = hour.groupby("dteday")["cnt"].sum().reindex(day.index, fill_value=0)

    def calendario(df: pd.DataFrame, fechas: pd.DatetimeIndex) -> dict:
        return {
            "yr": bool((df["yr"].to_numpy() == fechas.year - 2011).all()),
            "mnth": bool((df["mnth"].to_numpy() == fechas.month).all()),
            # ConvenciÃ³n del dataset: 0 = domingo.
            "weekday": bool((df["weekday"].to_numpy() == (fechas.day_of_week + 1) % 7).all()),
        }

    return {
        "dias_esperados": len(dias),
        "dias_presentes": len(day),
        "dias_faltantes": len(dias.difference(day.index)),
        "dias_duplicados": int(day.index.duplicated().sum()),
        "horas_esperadas": len(horas),
        "horas_presentes": len(hour),
        "horas_faltantes": faltantes,
        "horas_duplicadas": int(hour.index.duplicated().sum()),
        "nulos_day": int(day.isna().sum().sum()),
        "nulos_hour": int(hour.isna().sum().sum()),
        "dias_cnt_distinto": int((suma_horaria != day["cnt"]).sum()),
        "casual_mas_registered": bool((day["casual"] + day["registered"] == day["cnt"]).all()
                                      and (hour["casual"] + hour["registered"] == hour["cnt"]).all()),
        "calendario_day": calendario(day, day.index),
        "calendario_hour": calendario(hour, pd.DatetimeIndex(hour["dteday"])),
    }


def desnormalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas en unidades fÃ­sicas sin tocar las normalizadas."""
    df = df.copy()
    for col, (nueva, factor, desfase) in DESNORMALIZACION.items():
        df[nueva] = df[col] * factor + desfase
    return df


def serie_diaria(directorio: Path = DIRECTORIO_DATOS) -> pd.DataFrame:
    """Tabla diaria con frecuencia ``D``, humedad corregida y variables desnormalizadas.

    El sensor de humedad marcÃ³ 0 todo el 2011-03-10, un valor imposible; se imputa
    con la media de los dÃ­as vecinos.
    """
    day, _ = cargar_crudos(directorio)
    day = day.drop(columns="instant").asfreq("D")
    vecinos = [FECHA_HUMEDAD_CERO - pd.Timedelta(days=1), FECHA_HUMEDAD_CERO + pd.Timedelta(days=1)]
    day.loc[FECHA_HUMEDAD_CERO, "hum"] = day.loc[vecinos, "hum"].mean()
    return desnormalizar(day)


def serie_horaria(directorio: Path = DIRECTORIO_DATOS) -> pd.DataFrame:
    """Rejilla horaria completa con imputaciÃ³n diferenciada por tipo de variable.

    - Demanda (``casual``, ``registered``, ``cnt``): 0. Las horas ausentes son horas sin
      alquileres (madrugadas, enero de 2011 y el cierre por el huracÃ¡n Sandy); asÃ­ la
      suma diaria sigue coincidiendo con ``day.csv``.
    - Clima continuo: interpolaciÃ³n lineal. Las 22 horas con humedad 0 del 2011-03-10
      se tratan como faltantes antes de interpolar.
    - ``weathersit``: se arrastra el Ãºltimo valor observado.
    - ``season``, ``holiday``, ``workingday``: se toman de la tabla diaria.
    - ``hr``, ``weekday``, ``mnth``, ``yr``: se reconstruyen desde el Ã­ndice.
    """
    day, hour = cargar_crudos(directorio)
    rejilla = pd.date_range(hour.index.min().normalize(), day.index.max() + pd.Timedelta(hours=23),
                            freq="h", name="timestamp")
    h = hour.drop(columns=["instant", "dteday"]).reindex(rejilla)
    h["imputada"] = h["cnt"].isna()

    h[DEMANDA] = h[DEMANDA].fillna(0).astype(int)
    h.loc[h["hum"] == 0, "hum"] = np.nan
    h[CLIMA_CONTINUO] = h[CLIMA_CONTINUO].interpolate(method="linear", limit_direction="both")
    h["weathersit"] = h["weathersit"].ffill().astype(int)

    fechas = rejilla.normalize()
    for col in ["season", "holiday", "workingday"]:
        h[col] = day[col].reindex(fechas).to_numpy()
    h["hr"] = rejilla.hour
    h["weekday"] = (rejilla.day_of_week + 1) % 7
    h["mnth"] = rejilla.month
    h["yr"] = rejilla.year - 2011
    return desnormalizar(h)


def particion(serie: pd.DataFrame | pd.Series) -> tuple:
    """Divide una serie diaria u horaria en entrenamiento (< 2012-12-02) y prueba."""
    return serie[serie.index < INICIO_PRUEBA], serie[serie.index >= INICIO_PRUEBA]
