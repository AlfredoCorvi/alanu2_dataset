"""
Limpieza del dataset "Calificaciones de un grupo escolar".

  E  Extracción     : lee data/raw/calificaciones_raw.csv
  T  Transformación : limpieza, estandarización, validación, imputación y variables derivadas
  L  Carga          : guarda data/clean/calificaciones_limpio.csv

Uso:  python scripts/limpiar_dataset.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

CONFIG = {
    "raw": BASE / "data/raw/calificaciones_raw.csv",
    "clean_csv": BASE / "data/clean/calificaciones_limpio.csv",
    "nota_min": 0, "nota_max": 100, "nota_aprobatoria": 70,
    "asistencia_min": 0, "asistencia_max": 100,
}

SEXO_MAP = {"m": "Masculino", "masculino": "Masculino", "h": "Masculino",
            "f": "Femenino", "femenino": "Femenino", "mujer": "Femenino"}
CARRERA_MAP = {"sistemas": "Sistemas", "ing sistemas": "Sistemas",
               "administracion": "Administración", "admin": "Administración",
               "contaduria": "Contaduría"}
MATERIA_MAP = {"matematicas": "Matemáticas", "programacion": "Programación",
               "bases de datos": "Bases de Datos", "ingles": "Inglés"}
TOKENS_NULOS = {"", "na", "n/a", "s/c", "n/d", "-", "nan", "none", "null"}
LOG = {}


def norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower().replace(".", " ")).strip()


def colapsar(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def a_numero(serie):
    s = serie.astype(str).str.strip().str.replace("%", "", regex=False)
    low = s.str.lower()
    vacio = low.isin({"", "nan"})
    token = low.isin(TOKENS_NULOS) & ~vacio
    coma = s.str.contains(",", regex=False)
    num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
    return num, {"vacios": int(vacio.sum()), "tokens_texto": int(token.sum()), "coma_decimal": int(coma.sum())}


def extraer():
    raw = pd.read_csv(CONFIG["raw"], dtype=str, keep_default_na=False, encoding="utf-8")
    LOG["extraccion"] = {"filas_crudas": len(raw), "columnas": raw.shape[1], "fuentes": 1}
    return raw


def transformar(raw):
    df, t = raw.copy(), {}

    # 1) duplicados exactos
    t["duplicados_exactos"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2) texto: espacios y mayúsculas
    t["nombres_irregulares"] = int((raw["nombre"] != raw["nombre"].map(colapsar).str.title()).sum())
    df["nombre"] = df["nombre"].map(colapsar).str.title()
    df["periodo"] = df["periodo"].map(colapsar)

    # 3) categorías: sexo, carrera, materia
    t["sexo_variantes_distintas"] = int(raw["sexo"].map(norm).nunique())
    t["sexo_no_canonico"] = int((~df["sexo"].isin(["Masculino", "Femenino"])).sum())
    df["sexo"] = df["sexo"].map(norm).map(SEXO_MAP)
    t["carrera_variantes_distintas"] = int(raw["carrera"].nunique())
    t["carrera_no_canonica"] = int((~df["carrera"].isin(CARRERA_MAP.values())).sum())
    df["carrera"] = df["carrera"].map(norm).map(CARRERA_MAP)
    t["materia_variantes_distintas"] = int(raw["materia"].nunique())
    t["materia_no_canonica"] = int((~df["materia"].isin(MATERIA_MAP.values())).sum())
    df["materia"] = df["materia"].map(norm).map(MATERIA_MAP)
    t["categorias_sin_clasificar"] = int(df[["carrera", "materia"]].isna().sum().sum())

    # 4) tipos numéricos y rangos
    st_n, fr_n = {"vacios": 0, "tokens_texto": 0, "coma_decimal": 0}, 0
    for c in ["parcial1", "parcial2"]:
        num, st = a_numero(df[c])
        for k in st_n:
            st_n[k] += st[k]
        fr = (num < CONFIG["nota_min"]) | (num > CONFIG["nota_max"])
        fr_n += int(fr.sum())
        num[fr] = np.nan
        df[c] = num
    t["notas"], t["notas_fuera_de_rango"] = st_n, fr_n
    asis, st = a_numero(df["asistencia_pct"])
    t["asistencia"] = {**st, "con_simbolo_pct": int(raw["asistencia_pct"].str.contains("%").sum())}
    fr = (asis < CONFIG["asistencia_min"]) | (asis > CONFIG["asistencia_max"])
    t["asistencia_fuera_de_rango"] = int(fr.sum())
    asis[fr] = np.nan
    df["asistencia_pct"] = asis

    # 5) llave natural única
    antes = len(df)
    df = df.drop_duplicates(subset=["matricula", "materia", "periodo"], keep="first")
    t["duplicados_llave_natural"] = antes - len(df)

    # 6) imputación con la mediana de la materia
    imput = {}
    for c in ["parcial1", "parcial2", "asistencia_pct"]:
        imput[c] = int(df[c].isna().sum())
        df[c] = df[c].fillna(df.groupby("materia")[c].transform("median"))
        df[c] = df[c].fillna(df[c].median()).round(1)
    t["imputados"], t["total_valores_imputados"] = imput, int(sum(imput.values()))

    # 7) atributos del alumno consistentes entre sus registros
    al = df.groupby("matricula")[["nombre", "sexo", "carrera"]].first()
    df = df.drop(columns=["nombre", "sexo", "carrera"]).join(al, on="matricula")
    df["sexo"] = df["sexo"].fillna("No especificado")

    # 8) variables derivadas
    df["promedio_final"] = df[["parcial1", "parcial2"]].mean(axis=1).round(1)
    df["aprobado"] = (df["promedio_final"] >= CONFIG["nota_aprobatoria"]).astype(int)

    df = df[["matricula", "nombre", "sexo", "carrera", "materia", "periodo", "parcial1", "parcial2",
             "promedio_final", "asistencia_pct", "aprobado"]].sort_values(["matricula", "materia"]).reset_index(drop=True)
    t["filas_limpias"], t["alumnos"] = len(df), int(df["matricula"].nunique())
    LOG["transformacion"] = t
    return df


def cargar(df):
    CONFIG["clean_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONFIG["clean_csv"], index=False, encoding="utf-8")


def main():
    raw = extraer()
    df = transformar(raw)
    cargar(df)
    print(f"Limpio: {len(df)} filas -> {CONFIG['clean_csv']}")
    print("Resumen de la limpieza:")
    print(json.dumps(LOG["transformacion"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
