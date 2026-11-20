# %% [markdown]
# # Unidad 3 - Análisis supervisado: predicción de heladas (KNMI)
# Caso: cooperativa agrícola que necesita anticipar heladas y temperaturas bajas.
#
# * **Clasificación:** ¿habrá helada de aire mañana? (`helada_manana` = temperatura mínima de mañana < 0 °C)
# * **Regresión:** ¿cuál será la temperatura mínima de mañana? (`tn_manana`, en °C)
#
# Dataset: `dataset_agricola_diario_2016_2025.csv` (30 estaciones del KNMI, 2016-2025).
# Partición cronológica: entrenamiento 2016-2022 y prueba 2023-2025.
# Ejecutar:  python analisis_supervisado.py        (modo rápido de prueba: QUICK=1 python analisis_supervisado.py)

# %% 0. Configuración
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
QUICK = os.environ.get("QUICK") == "1"
SEED = 42
BASE = Path(__file__).resolve().parent
for cand in [BASE / "dataset_agricola_diario_2016_2025.csv", BASE.parent / "dataset_agricola_diario_2016_2025.csv"]:
    if cand.exists():
        RUTA = cand
        break
else:
    raise FileNotFoundError("No se encontró dataset_agricola_diario_2016_2025.csv")
OUT = BASE / ("resultados_quick" if QUICK else "resultados")
OUT.mkdir(exist_ok=True)

TRAIN_FIN = "2022-12-31"            # entrenamiento: 2016-2022
TEST_INI = "2023-01-01"             # prueba: 2023-2025
CV_ANIOS = [2021, 2022] if QUICK else [2019, 2020, 2021, 2022]   # cada año se valida con los años anteriores
N_ITER = {"rf": 2, "hgb": 2} if QUICK else {"rf": 5, "hgb": 6}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# %% 1. Carga y preparación de variables
df = pd.read_csv(RUTA, encoding="utf-8-sig", parse_dates=["fecha"])
df = df.rename(columns={"temperatura_media_c": "tg", "temperatura_minima_c": "tn", "temperatura_maxima_c": "tx",
                        "temperatura_minima_10cm_c": "t10n", "duracion_precipitacion_h": "dr",
                        "humedad_media_pct": "hum", "viento_medio_m_s": "viento",
                        "evapotranspiracion_potencial_mm": "ev", "precipitacion_mm": "rh",
                        "precipitacion_traza": "traza"})
df["traza"] = df["traza"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
if QUICK:
    df = df[df["estacion_id"].isin(sorted(df["estacion_id"].unique())[:4])]
df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)
g = df.groupby("estacion_id")


def rezago(col, k):
    """Valor de la variable k días antes, solo si ese día existe (fechas consecutivas)."""
    ok = (df["fecha"] - g["fecha"].shift(k)).dt.days == k
    return g[col].shift(k).where(ok)


# objetivos: el día siguiente (solo si la fila siguiente es realmente el día siguiente)
fecha_sig = g["fecha"].shift(-1)
sig_ok = (fecha_sig - df["fecha"]).dt.days == 1
df["tn_manana"] = g["tn"].shift(-1).where(sig_ok)
df["helada_manana"] = (df["tn_manana"] < 0).astype(float).where(df["tn_manana"].notna())
df["mes"] = df["fecha"].dt.month
df["mes_manana"] = (df["fecha"] + pd.Timedelta(days=1)).dt.month

# variables de entrada construidas solo con información de hoy y de días anteriores
for c in ["tn", "tg", "tx", "rh", "hum"]:
    df[f"{c}_l1"] = rezago(c, 1)
df["tn_l2"] = rezago("tn", 2)
df["dtn"] = df["tn"] - df["tn_l1"]
df["dtg"] = df["tg"] - df["tg_l1"]
df["tn_min3"] = df[["tn", "tn_l1", "tn_l2"]].min(axis=1)
df["rango"] = df["tx"] - df["tn"]
doy = df["fecha"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

FEATURES = ["tg", "tn", "tx", "t10n", "rh", "traza", "dr", "hum", "viento", "ev",
            "tn_l1", "tn_l2", "tg_l1", "tx_l1", "rh_l1", "hum_l1", "dtn", "dtg", "tn_min3", "rango",
            "doy_sin", "doy_cos", "clim_tn", "anom_tn"]

elig = df["tn"].notna() & df["tn_manana"].notna()
log(f"filas totales: {len(df)}; elegibles (hoy y mañana con TN): {int(elig.sum())}")


def tabla_clim(d):
    """Climatología: media de TN por estación y mes (se calcula SOLO con el periodo de entrenamiento)."""
    return d.groupby(["estacion_id", "mes"])["tn"].mean().rename("clim_tn").reset_index()


def con_clim(tabla):
    m = df[["estacion_id", "mes", "mes_manana"]].merge(tabla, on=["estacion_id", "mes"], how="left")
    sig = tabla.rename(columns={"mes": "mes_manana", "clim_tn": "clim_tn_manana"})
    m = m.merge(sig, on=["estacion_id", "mes_manana"], how="left")
    F = df.copy()
    F["clim_tn"] = m["clim_tn"].values
    F["clim_tn_manana"] = m["clim_tn_manana"].values
    F["anom_tn"] = F["tn"] - F["clim_tn"]
    return F


# %% 2. Particiones: entrenamiento / prueba y validación cruzada con ventana creciente
tr_mask = elig & (df["fecha"] <= TRAIN_FIN)
te_mask = elig & (df["fecha"] >= TEST_INI)
log(f"entrenamiento: {int(tr_mask.sum())} filas; prueba: {int(te_mask.sum())} filas")

folds = []
for v in CV_ANIOS:
    tr = elig & (df["fecha"] < f"{v}-01-01")
    va = elig & (df["fecha"].dt.year == v)
    F = con_clim(tabla_clim(df[tr]))
    folds.append({"anio": v, "Xtr": F.loc[tr, FEATURES], "Xva": F.loc[va, FEATURES],
                  "ytr_c": F.loc[tr, "helada_manana"].values, "yva_c": F.loc[va, "helada_manana"].values,
                  "ytr_r": F.loc[tr, "tn_manana"].values, "yva_r": F.loc[va, "tn_manana"].values})
Ffull = con_clim(tabla_clim(df[tr_mask]))
Xtr, Xte = Ffull.loc[tr_mask, FEATURES], Ffull.loc[te_mask, FEATURES]
ytr_c, yte_c = Ffull.loc[tr_mask, "helada_manana"].values, Ffull.loc[te_mask, "helada_manana"].values
ytr_r, yte_r = Ffull.loc[tr_mask, "tn_manana"].values, Ffull.loc[te_mask, "tn_manana"].values
log(f"tasa de helada mañana: entrenamiento {ytr_c.mean():.3f}, prueba {yte_c.mean():.3f}")


# %% 3. Modelos y espacios de búsqueda
def construir(tarea, nombre, params):
    p = dict(params)
    if tarea == "clf":
        if nombre == "Regresión logística":
            return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                             ("m", LogisticRegression(max_iter=2000, **p))])
        if nombre == "Random Forest":
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", RandomForestClassifier(n_estimators=100, max_samples=0.5, n_jobs=-1,
                                                          random_state=SEED, **p))])
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **p)
    if nombre == "Regresión lineal":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", Ridge(**p) if p else LinearRegression())])
    if nombre == "Random Forest":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestRegressor(n_estimators=100, max_samples=0.5, n_jobs=-1,
                                                     random_state=SEED, **p))])
    return HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **p)


ESPACIOS = {
    ("clf", "Regresión logística"): ({"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}, 8),
    ("clf", "Random Forest"): ({"max_depth": [6, 10, 14], "min_samples_leaf": [5, 20, 50], "max_features": [0.3, 0.5],
                                "class_weight": [None, "balanced_subsample"]}, N_ITER["rf"]),
    ("clf", "Gradient Boosting"): ({"learning_rate": [0.03, 0.06, 0.1], "max_iter": [150, 300],
                                    "max_depth": [3, 5, None], "min_samples_leaf": [20, 50, 100],
                                    "l2_regularization": [0, 1, 10]}, N_ITER["hgb"]),
    ("reg", "Regresión lineal"): ({"alpha": [0.01, 0.1, 1, 10, 100, 1000]}, 6),
    ("reg", "Random Forest"): ({"max_depth": [8, 12, None], "min_samples_leaf": [5, 20, 50],
                                "max_features": [0.3, 0.5, 1.0]}, N_ITER["rf"]),
    ("reg", "Gradient Boosting"): ({"learning_rate": [0.03, 0.06, 0.1], "max_iter": [150, 300],
                                    "max_depth": [3, 5, None], "min_samples_leaf": [20, 50, 100],
                                    "l2_regularization": [0, 1, 10]}, N_ITER["hgb"]),
}


# %% 4. Métricas
def m_clf(y, p, umbral=0.5):
    yh = (p >= umbral).astype(int)
    return {"Accuracy": accuracy_score(y, yh), "Precision": precision_score(y, yh, zero_division=0),
            "Recall": recall_score(y, yh), "F1": f1_score(y, yh), "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p)}


def m_reg(y, yh):
    return {"MAE": mean_absolute_error(y, yh), "RMSE": mean_squared_error(y, yh) ** 0.5, "R2": r2_score(y, yh),
            "Sesgo": float(np.mean(yh - y))}


def pred_clf(modelo, X):
    return modelo.predict_proba(X)[:, 1]


# %% 5. Optimización: búsqueda aleatoria con validación cruzada temporal (ventana creciente)
mejores, cv_filas, oof_clf = {}, [], {}
for (tarea, nombre), (espacio, n_iter) in ESPACIOS.items():
    cands = list(ParameterSampler(espacio, n_iter=min(n_iter, int(np.prod([len(v) for v in espacio.values()]))),
                                  random_state=SEED))
    mejor = None
    for params in cands:
        sc, oof = [], []
        for f in folds:
            mod = construir(tarea, nombre, params)
            if tarea == "clf":
                mod.fit(f["Xtr"], f["ytr_c"])
                p = pred_clf(mod, f["Xva"])
                sc.append(average_precision_score(f["yva_c"], p))
                oof.append(p)
            else:
                mod.fit(f["Xtr"], f["ytr_r"])
                yh = mod.predict(f["Xva"])
                sc.append(mean_squared_error(f["yva_r"], yh) ** 0.5)
        media = float(np.mean(sc))
        cv_filas.append({"Tarea": "Clasificación" if tarea == "clf" else "Regresión", "Modelo": nombre,
                         "Hiperparámetros": json.dumps(params), "Métrica CV": "PR-AUC" if tarea == "clf" else "RMSE",
                         "Media": media, "Desv. est.": float(np.std(sc))})
        mejor_es = mejor is None or (media > mejor[0] if tarea == "clf" else media < mejor[0])
        if mejor_es:
            mejor = (media, params, oof)
        log(f"{tarea} {nombre} {params} -> {media:.4f}")
    mejores[(tarea, nombre)] = mejor[1]
    if tarea == "clf":
        oof_clf[nombre] = np.concatenate(mejor[2])
pd.DataFrame(cv_filas).to_csv(OUT / "cv_resultados.csv", index=False, encoding="utf-8")
(OUT / "mejores_parametros.json").write_text(json.dumps(
    {f"{t}|{n}": p for (t, n), p in mejores.items()}, indent=2, ensure_ascii=False), encoding="utf-8")

# umbral de decisión: se elige con las predicciones fuera de muestra de la validación (no con la prueba)
y_oof = np.concatenate([f["yva_c"] for f in folds])
umbrales = {}
for nombre, p in oof_clf.items():
    pr, rc, th = precision_recall_curve(y_oof, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    umbrales[nombre] = {"umbral_f1": float(th[np.argmax(f1)])}
    ok = np.where(rc[:-1] >= 0.90)[0]
    umbrales[nombre]["umbral_recall90"] = float(th[ok[-1]]) if len(ok) else float(th[0])
(OUT / "umbrales.json").write_text(json.dumps(umbrales, indent=2), encoding="utf-8")

# %% 6. Entrenamiento final (2016-2022) y evaluación en prueba (2023-2025)
def ajustar_final(tarea, nombre, params):
    mod = construir(tarea, nombre, params)
    mod.fit(Xtr, ytr_c if tarea == "clf" else ytr_r)
    return mod


pred = Ffull.loc[te_mask, ["estacion_id", "fecha", "tn", "tn_manana", "helada_manana", "mes_manana",
                           "clim_tn_manana"]].copy()
pred["helada_hoy"] = (pred["tn"] < 0).astype(int)
filas_c, filas_r, modelos = [], [], {}

# clasificación
mayoria = int(round(ytr_c.mean()))
filas_c.append({"Modelo": "Línea base: clase mayoritaria", "Versión": "-",
                **m_clf(yte_c, np.full(len(yte_c), float(mayoria)))})
filas_c.append({"Modelo": "Línea base: persistencia (helada hoy)", "Versión": "-",
                **m_clf(yte_c, pred["helada_hoy"].values.astype(float))})
for nombre in ["Regresión logística", "Random Forest", "Gradient Boosting"]:
    for version, params in [("Por defecto", {}), ("Optimizado", mejores[("clf", nombre)])]:
        mod = ajustar_final("clf", nombre, params)
        p = pred_clf(mod, Xte)
        modelos[("clf", nombre, version)] = mod
        u = umbrales[nombre]["umbral_f1"] if version == "Optimizado" else 0.5
        filas_c.append({"Modelo": nombre, "Versión": version, "Umbral": u, **m_clf(yte_c, p, u)})
        if version == "Optimizado":
            pred[f"p_{nombre}"] = p
            filas_c.append({"Modelo": nombre, "Versión": "Optimizado + umbral recall 90 %",
                            "Umbral": umbrales[nombre]["umbral_recall90"],
                            **m_clf(yte_c, p, umbrales[nombre]["umbral_recall90"])})
        log(f"clf {nombre} {version} listo")

# regresión
filas_r.append({"Modelo": "Línea base: persistencia (TN de hoy)", "Versión": "-", **m_reg(yte_r, Xte["tn"].values)})
filas_r.append({"Modelo": "Línea base: climatología (estación y mes)", "Versión": "-",
                **m_reg(yte_r, Ffull.loc[te_mask, "clim_tn_manana"].values)})
pred["reg_persistencia"] = Xte["tn"].values
for nombre in ["Regresión lineal", "Random Forest", "Gradient Boosting"]:
    for version, params in [("Por defecto", {}), ("Optimizado", mejores[("reg", nombre)])]:
        if nombre == "Regresión lineal" and version == "Por defecto":
            params = {}
        mod = ajustar_final("reg", nombre, params)
        yh = mod.predict(Xte)
        modelos[("reg", nombre, version)] = mod
        filas_r.append({"Modelo": nombre, "Versión": version, **m_reg(yte_r, yh)})
        if version == "Optimizado":
            pred[f"y_{nombre}"] = yh
        log(f"reg {nombre} {version} listo")

dc, dr_ = pd.DataFrame(filas_c), pd.DataFrame(filas_r)
dc.to_csv(OUT / "test_clasificacion.csv", index=False, encoding="utf-8")
dr_.to_csv(OUT / "test_regresion.csv", index=False, encoding="utf-8")
pred.to_csv(OUT / "predicciones_test.csv", index=False, encoding="utf-8")

# %% 7. Análisis del mejor modelo: por mes e importancia de variables
mejor_c = dc[(dc["Versión"] == "Optimizado")].sort_values("PR-AUC", ascending=False).iloc[0]["Modelo"]
mejor_r = dr_[(dr_["Versión"] == "Optimizado")].sort_values("RMSE").iloc[0]["Modelo"]
log(f"mejor clasificador: {mejor_c}; mejor regresor: {mejor_r}")
pj = {"mejor_clasificador": mejor_c, "mejor_regresor": mejor_r}
(OUT / "mejores_modelos.json").write_text(json.dumps(pj, ensure_ascii=False), encoding="utf-8")

u = umbrales[mejor_c]["umbral_f1"]
pred["yh_c"] = (pred[f"p_{mejor_c}"] >= u).astype(int)
pred["err_abs"] = (pred[f"y_{mejor_r}"] - pred["tn_manana"]).abs()
por_mes = []
for mes, d in pred.groupby("mes_manana"):
    fila = {"mes": int(mes), "dias": len(d), "heladas_reales": int(d["helada_manana"].sum()),
            "MAE_reg": float(d["err_abs"].mean())}
    fila["recall"] = float(recall_score(d["helada_manana"], d["yh_c"])) if d["helada_manana"].sum() > 0 else np.nan
    fila["precision"] = float(precision_score(d["helada_manana"], d["yh_c"], zero_division=0)) if d["yh_c"].sum() > 0 else np.nan
    por_mes.append(fila)
pd.DataFrame(por_mes).to_csv(OUT / "por_mes.csv", index=False, encoding="utf-8")
tn_, fp_, fn_, tp_ = confusion_matrix(pred["helada_manana"], pred["yh_c"]).ravel()
(OUT / "matriz_confusion.json").write_text(json.dumps(
    {"modelo": mejor_c, "umbral": u, "TN": int(tn_), "FP": int(fp_), "FN": int(fn_), "TP": int(tp_)}), encoding="utf-8")

rs = np.random.default_rng(SEED)
idx = rs.choice(len(Xte), size=min(12000, len(Xte)), replace=False)
for tarea, nombre, y, scoring, archivo in [("clf", mejor_c, yte_c, "average_precision", "importancias_clf.csv"),
                                           ("reg", mejor_r, yte_r, "neg_root_mean_squared_error", "importancias_reg.csv")]:
    mod = modelos[(tarea, nombre, "Optimizado")]
    imp = permutation_importance(mod, Xte.iloc[idx], y[idx], scoring=scoring, n_repeats=3, random_state=SEED, n_jobs=1)
    pd.DataFrame({"variable": FEATURES, "importancia": imp.importances_mean, "desv": imp.importances_std}) \
        .sort_values("importancia", ascending=False).to_csv(OUT / archivo, index=False, encoding="utf-8")
    log(f"importancias {tarea} listas")

resumen = {"filas_entrenamiento": int(tr_mask.sum()), "filas_prueba": int(te_mask.sum()),
           "tasa_helada_train": float(ytr_c.mean()), "tasa_helada_test": float(yte_c.mean()),
           "features": FEATURES, "cv_anios": CV_ANIOS, "tiempo_s": round(time.time() - T0)}
(OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
for tarea, nombre in [("clf", mejor_c), ("reg", mejor_r)]:
    if nombre != "Random Forest":
        joblib.dump(modelos[(tarea, nombre, "Optimizado")], OUT / f"modelo_{tarea}_{nombre.replace(' ', '_')}.joblib", compress=3)
log("terminado")
