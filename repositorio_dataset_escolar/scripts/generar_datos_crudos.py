"""
Genera el dataset CRUDO (sin limpiar) del caso "Calificaciones de un grupo escolar".
Datos sintéticos, sencillos y con errores de captura intencionales para practicar limpieza.

Columnas: matricula, nombre, sexo, carrera, materia, periodo, parcial1, parcial2, asistencia_pct
Salida:   data/raw/calificaciones_raw.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

CARRERAS = {"Sistemas": ["SISTEMAS", "sistemas", "Ing. Sistemas"],
            "Administración": ["Administracion", "ADMINISTRACIÓN", "admin"],
            "Contaduría": ["Contaduria", "CONTADURIA"]}
MATERIAS = ["Matemáticas", "Programación", "Bases de Datos", "Inglés"]
EFECTO = {"Matemáticas": -6, "Programación": 0, "Bases de Datos": -2, "Inglés": 4}
PERIODOS = ["2024-1", "2024-2"]

H = ["Carlos", "Luis", "Miguel", "Jorge", "Diego", "Andrés", "Fernando", "Ricardo", "Daniel", "Alejandro",
     "Iván", "Raúl", "Hugo", "Pablo", "Adrián"]
M = ["María", "Ana", "Sofía", "Valeria", "Camila", "Fernanda", "Daniela", "Paola", "Karla", "Mariana",
     "Ximena", "Regina", "Lucía", "Andrea", "Diana"]
AP = ["García", "Hernández", "Martínez", "López", "González", "Rodríguez", "Pérez", "Sánchez", "Ramírez", "Flores",
      "Gómez", "Díaz", "Reyes", "Morales", "Cruz", "Ortiz", "Gutiérrez", "Chávez", "Ramos", "Mendoza",
      "Vázquez", "Castillo", "Jiménez", "Moreno", "Romero", "Herrera", "Medina", "Aguilar", "Garza", "Salinas"]

apellidos = list(rng.permutation(AP))
alumnos = []
for i in range(30):
    es_h = i < 15
    alumnos.append({"matricula": f"A{i+1:03d}", "nombre": f"{(H if es_h else M)[i % 15]} {apellidos[i]}",
                    "sexo": "Masculino" if es_h else "Femenino",
                    "carrera": rng.choice(list(CARRERAS), p=[0.4, 0.35, 0.25]),
                    "habilidad": rng.normal(75, 10)})


def nombre_sucio(n):
    u = rng.random()
    return n.upper() if u < 0.10 else n.lower() if u < 0.20 else f"  {n} " if u < 0.28 else n


def sexo_sucio(s):
    u = rng.random()
    if u < 0.04:
        return ""
    op = ["Masculino", "M", "masculino", "H"] if s == "Masculino" else ["Femenino", "F", "femenino", "Mujer"]
    return rng.choice(op, p=[0.4, 0.3, 0.15, 0.15])


def carrera_sucia(c):
    return rng.choice(CARRERAS[c]) if rng.random() < 0.30 else c


def materia_sucia(m):
    u = rng.random()
    return m.upper() if u < 0.08 else m.lower() + " " if u < 0.15 else m


def nota_sucia(v):
    u = rng.random()
    if u < 0.06:
        return ""
    if u < 0.09:
        return "NA"
    if u < 0.12:
        return rng.choice(["105", "150", "-5"])
    if u < 0.17:
        return f"{v},{int(rng.integers(0, 10))}"
    return str(v)


def asis_sucia(v):
    u = rng.random()
    if u < 0.08:
        return f"{v}%"
    if u < 0.12:
        return ""
    if u < 0.14:
        return "120"
    return str(v)


filas = []
for a in alumnos:
    periodo = rng.choice(PERIODOS)
    for mat in rng.choice(MATERIAS, 2, replace=False):
        base = a["habilidad"] + EFECTO[mat]
        p1, p2 = [int(np.clip(round(base + rng.normal(0, 8)), 30, 100)) for _ in range(2)]
        asis = int(np.clip(round(rng.normal(88, 8) + (a["habilidad"] - 75) * 0.3), 55, 100))
        filas.append({"matricula": a["matricula"], "nombre": nombre_sucio(a["nombre"]), "sexo": sexo_sucio(a["sexo"]),
                      "carrera": carrera_sucia(a["carrera"]), "materia": materia_sucia(mat), "periodo": periodo,
                      "parcial1": nota_sucia(p1), "parcial2": nota_sucia(p2), "asistencia_pct": asis_sucia(asis)})

df = pd.DataFrame(filas)
df = pd.concat([df, df.sample(n=3, random_state=7)], ignore_index=True)   # 3 duplicados exactos
df = df.sample(frac=1, random_state=11).reset_index(drop=True)
df.to_csv(OUT / "calificaciones_raw.csv", index=False, encoding="utf-8")
print(f"calificaciones_raw.csv: {len(df)} filas, {df.shape[1]} columnas")
