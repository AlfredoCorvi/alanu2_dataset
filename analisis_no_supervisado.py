# %% [markdown]
# # Unidad 4 - Análisis no supervisado: tipos de día y regiones climáticas (KNMI)
# Caso: cooperativa agrícola que necesita identificar periodos de lluvia, heladas y temperaturas extremas.
#
# * **Parte A (días):** reducción de dimensionalidad (PCA) + clustering (K-Means; comparado con jerárquico Ward y DBSCAN)
#   para descubrir *tipos de día* a partir de 9 variables meteorológicas, sin usar etiquetas.
# * **Parte B (estaciones):** clustering jerárquico y K-Means sobre el perfil climático de las 30 estaciones.
#
# Dataset: `dataset_agricola_diario_2016_2025.csv` (limpio, de la Unidad 2).
# Ejecutar:  python analisis_no_supervisado.py        (prueba rápida: QUICK=1 python analisis_no_supervisado.py)

# %% 0. Configuración
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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
N_SIL = 5000 if QUICK else 10000          # tamaño de muestra para el coeficiente de silueta
N_WARD = 1500 if QUICK else 6000          # muestra para el clustering jerárquico (costo cuadrático)
N_DB = 5000 if QUICK else 20000           # muestra para DBSCAN
N_TSNE = 1500 if QUICK else 5000          # muestra para t-SNE
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def metricas(X, etiquetas, seed=SEED):
    """Tres métricas internas, calculadas SIEMPRE en el mismo espacio (variables estandarizadas)."""
    n = len(X)
    sil = silhouette_score(X, etiquetas, sample_size=min(N_SIL, n), random_state=seed)
    return {"Silueta": float(sil), "Calinski-Harabasz": float(calinski_harabasz_score(X, etiquetas)),
            "Davies-Bouldin": float(davies_bouldin_score(X, etiquetas))}


# %% 1. Carga y selección de variables
df = pd.read_csv(RUTA, encoding="utf-8-sig", parse_dates=["fecha"])
df = df.rename(columns={"temperatura_media_c": "tg", "temperatura_minima_c": "tn", "temperatura_maxima_c": "tx",
                        "temperatura_minima_10cm_c": "t10n", "duracion_precipitacion_h": "dr",
                        "humedad_media_pct": "hum", "viento_medio_m_s": "viento",
                        "evapotranspiracion_potencial_mm": "ev", "precipitacion_mm": "rh"})
if QUICK:
    df = df[df["estacion_id"].isin(sorted(df["estacion_id"].unique())[:6])]
VARS = ["tg", "tn", "tx", "t10n", "rh", "dr", "hum", "viento", "ev"]
d = df.dropna(subset=VARS).reset_index(drop=True)          # solo registros completos
log(f"registros: {len(df)}; completos (9 variables): {len(d)} ({len(d) / len(df):.1%})")

# %% 2. Transformación: logaritmo en variables muy sesgadas y estandarización
X = d[VARS].copy()
X["rh"] = np.log1p(X["rh"])
X["dr"] = np.log1p(X["dr"])
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# %% 3. Reducción de dimensionalidad: PCA
pca_full = PCA(random_state=SEED).fit(Xs)
var_exp = pca_full.explained_variance_ratio_
acum = np.cumsum(var_exp)
N_PC = int(np.argmax(acum >= 0.90) + 1)                     # componentes para conservar al menos 90 % de la varianza
pca = PCA(n_components=N_PC, random_state=SEED).fit(Xs)
Z = pca.transform(Xs)
pd.DataFrame({"componente": [f"PC{i + 1}" for i in range(len(var_exp))], "varianza_explicada": var_exp,
              "varianza_acumulada": acum}).to_csv(OUT / "pca_varianza.csv", index=False, encoding="utf-8")
pd.DataFrame(pca_full.components_[:4].T, index=VARS, columns=[f"PC{i + 1}" for i in range(4)]) \
    .to_csv(OUT / "pca_cargas.csv", encoding="utf-8")
log(f"PCA: {N_PC} componentes conservan {acum[N_PC - 1]:.1%} de la varianza; PC1 {var_exp[0]:.1%}, PC2 {var_exp[1]:.1%}")

# %% 4. Modelo "antes": K-Means con parámetros por defecto (k = 8) sobre los datos sin transformar ni escalar
km0 = KMeans(n_clusters=8, random_state=SEED).fit(d[VARS].values)
base = {"Configuración": "Antes: K-Means por defecto (k = 8), datos sin escalar", "k": 8,
        **metricas(Xs, km0.labels_)}
log("modelo base:", {k: round(v, 3) if isinstance(v, float) else v for k, v in base.items()})

# %% 5. Optimización: número de clústeres (K-Means sobre las componentes principales)
filas_k = []
modelos_k = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    modelos_k[k] = km
    filas_k.append({"k": k, "Inercia": float(km.inertia_), **metricas(Xs, km.labels_)})
    log(f"k = {k}: silueta {filas_k[-1]['Silueta']:.3f}, CH {filas_k[-1]['Calinski-Harabasz']:.0f}, DB {filas_k[-1]['Davies-Bouldin']:.3f}")
dk = pd.DataFrame(filas_k)
dk.to_csv(OUT / "k_means_busqueda_k.csv", index=False, encoding="utf-8")
# regla: k = 2 solo separa invierno de verano; se busca el mejor k entre 3 y 8 con la silueta
cand = dk[(dk["k"] >= 3) & (dk["k"] <= 8)]
K = int(cand.sort_values("Silueta", ascending=False).iloc[0]["k"])
km = modelos_k[K]
log(f"k elegido: {K}")

# %% 6. Alternativas: clustering jerárquico (Ward) y DBSCAN, sobre muestras
rng = np.random.default_rng(SEED)
alt = []
idx_w = rng.choice(len(Xs), size=min(N_WARD, len(Xs)), replace=False)
L = linkage(Xs[idx_w], method="ward")
lab_w = fcluster(L, t=K, criterion="maxclust") - 1
alt.append({"Algoritmo": f"Jerárquico Ward (k = {K}, muestra {len(idx_w):,})", "k": K, "Ruido (%)": 0.0,
            **metricas(Xs[idx_w], lab_w)})
lab_km_w = km.labels_[idx_w]
alt.append({"Algoritmo": f"K-Means optimizado (k = {K}, misma muestra)", "k": K, "Ruido (%)": 0.0,
            **metricas(Xs[idx_w], lab_km_w)})
idx_d = rng.choice(len(Xs), size=min(N_DB, len(Xs)), replace=False)
Xd = Xs[idx_d]
filas_db = []
for eps in [0.8, 1.0, 1.3, 1.6]:
    for ms in [10, 50]:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xd)
        ncl = len(set(lab)) - (1 if -1 in lab else 0)
        ruido = float((lab == -1).mean() * 100)
        fila = {"eps": eps, "min_samples": ms, "clústeres": ncl, "Ruido (%)": ruido, "Silueta": np.nan,
                "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan}
        if ncl >= 2:
            ok = lab != -1
            if len(set(lab[ok])) >= 2:
                fila.update(metricas(Xd[ok], lab[ok]))
        filas_db.append(fila)
        log(f"DBSCAN eps={eps} min_samples={ms}: {ncl} clústeres, ruido {ruido:.1f} %")
ddb = pd.DataFrame(filas_db)
ddb.to_csv(OUT / "dbscan_busqueda.csv", index=False, encoding="utf-8")
ok_db = ddb[(ddb["Ruido (%)"] < 15) & ddb["Silueta"].notna()]
if len(ok_db):
    b = ok_db.sort_values("Silueta", ascending=False).iloc[0]
    alt.append({"Algoritmo": f"DBSCAN (eps = {b['eps']}, min_samples = {int(b['min_samples'])}, muestra {len(idx_d):,})",
                "k": int(b["clústeres"]), "Ruido (%)": float(b["Ruido (%)"]), "Silueta": float(b["Silueta"]),
                "Calinski-Harabasz": float(b["Calinski-Harabasz"]), "Davies-Bouldin": float(b["Davies-Bouldin"])})
pd.DataFrame(alt).to_csv(OUT / "comparacion_algoritmos.csv", index=False, encoding="utf-8")

# %% 7. Resultado final y comparación antes vs. después
final = {"Configuración": f"Después: estandarizar + PCA ({N_PC} comp.) + K-Means optimizado (k = {K})", "k": K,
         **metricas(Xs, km.labels_)}
pd.DataFrame([base, final]).to_csv(OUT / "antes_vs_despues.csv", index=False, encoding="utf-8")
d["cluster"] = km.labels_

# perfil de cada clúster en las variables originales
d["mes"] = d["fecha"].dt.month
d["helada"] = (d["tn"] < 0).astype(int)
d["dia_lluvioso"] = (d["rh"] >= 1.0).astype(int)
d["calor"] = (d["tx"] >= 30).astype(int)
perfil = d.groupby("cluster").agg(dias=("tg", "size"), tg=("tg", "mean"), tn=("tn", "mean"), tx=("tx", "mean"),
                                  t10n=("t10n", "mean"), rh=("rh", "mean"), dr=("dr", "mean"), hum=("hum", "mean"),
                                  viento=("viento", "mean"), ev=("ev", "mean"), pct_helada=("helada", "mean"),
                                  pct_lluvioso=("dia_lluvioso", "mean"), pct_calor=("calor", "mean")).reset_index()
perfil["pct_dias"] = perfil["dias"] / perfil["dias"].sum()
perfil.to_csv(OUT / "perfil_clusters.csv", index=False, encoding="utf-8")
mes_cl = pd.crosstab(d["mes"], d["cluster"], normalize="index")
mes_cl.to_csv(OUT / "clusters_por_mes.csv", encoding="utf-8")
# cuánto de cada fenómeno agrícola concentra cada clúster
conc = pd.DataFrame({"fenomeno": ["Helada (TN < 0 °C)", "Día lluvioso (>= 1 mm)", "Calor extremo (TX >= 30 °C)"],
                     **{f"c{c}": [d.loc[d.cluster == c, col].sum() / d[col].sum() for col in ["helada", "dia_lluvioso", "calor"]]
                        for c in sorted(d.cluster.unique())}})
conc.to_csv(OUT / "concentracion_fenomenos.csv", index=False, encoding="utf-8")
log("perfiles listos")

# %% 8. Visualización: t-SNE sobre una muestra (solo ilustrativa)
idx_t = rng.choice(len(Xs), size=min(N_TSNE, len(Xs)), replace=False)
ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Xs[idx_t])
pd.DataFrame({"x": ts[:, 0], "y": ts[:, 1], "cluster": km.labels_[idx_t]}).to_csv(OUT / "tsne_muestra.csv", index=False)
log("t-SNE listo")

# %% 9. Parte B: regiones climáticas (clustering de las 30 estaciones)
g = df.groupby("estacion_id")
est = pd.DataFrame({
    "tn_media": g["tn"].mean(), "tx_media": g["tx"].mean(), "tg_media": g["tg"].mean(), "hum_media": g["hum"].mean(),
    "viento_medio": g["viento"].mean(), "ev_media": g["ev"].mean(), "lluvia_mm_dia": g["rh"].mean(),
    "dias_helada_anio": g["tn"].apply(lambda s: (s.dropna() < 0).mean() * 365.25),
    "dias_lluvia_anio": g["rh"].apply(lambda s: (s.dropna() >= 1.0).mean() * 365.25),
    "dias_calor_anio": g["tx"].apply(lambda s: (s.dropna() >= 30).mean() * 365.25)})
est = est.dropna()
Xe = StandardScaler().fit_transform(est.values)
pe = PCA(random_state=SEED).fit(Xe)
Ze = pe.transform(Xe)[:, :2]
pd.DataFrame({"componente": [f"PC{i + 1}" for i in range(len(pe.explained_variance_ratio_))],
              "varianza_explicada": pe.explained_variance_ratio_}).to_csv(OUT / "estaciones_pca_varianza.csv", index=False)
pd.DataFrame(pe.components_[:2].T, index=est.columns, columns=["PC1", "PC2"]).to_csv(OUT / "estaciones_pca_cargas.csv")
filas_e = []
Lw = linkage(Xe, method="ward")
for k in range(2, min(6, len(est) - 1) + 1):
    lab_h = fcluster(Lw, t=k, criterion="maxclust") - 1
    lab_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xe)
    filas_e.append({"k": k, "Algoritmo": "Jerárquico Ward", **{m: v for m, v in {
        "Silueta": silhouette_score(Xe, lab_h), "Calinski-Harabasz": calinski_harabasz_score(Xe, lab_h),
        "Davies-Bouldin": davies_bouldin_score(Xe, lab_h)}.items()}})
    filas_e.append({"k": k, "Algoritmo": "K-Means", "Silueta": silhouette_score(Xe, lab_k),
                    "Calinski-Harabasz": calinski_harabasz_score(Xe, lab_k), "Davies-Bouldin": davies_bouldin_score(Xe, lab_k)})
de = pd.DataFrame(filas_e)
de.to_csv(OUT / "estaciones_metricas.csv", index=False, encoding="utf-8")
KE = int(de[(de["Algoritmo"] == "Jerárquico Ward") & (de["k"] >= 3)].sort_values("Silueta", ascending=False).iloc[0]["k"])
est["region"] = fcluster(Lw, t=KE, criterion="maxclust") - 1
est["pc1"], est["pc2"] = Ze[:, 0], Ze[:, 1]
est.reset_index().to_csv(OUT / "estaciones_regiones.csv", index=False, encoding="utf-8")
np.save(OUT / "estaciones_linkage.npy", Lw)
pd.Series(est.index.astype(int)).to_csv(OUT / "estaciones_orden.csv", index=False, header=["estacion_id"])
log(f"estaciones: {len(est)}; k elegido = {KE}")

# %% 10. Dataset transformado para el análisis no supervisado
sal = d[["estacion_id", "fecha"]].copy()
for i, v in enumerate(VARS):
    sal[f"z_{v}"] = np.round(Xs[:, i], 4)
for i in range(N_PC):
    sal[f"PC{i + 1}"] = np.round(Z[:, i], 4)
sal["cluster"] = km.labels_
sal.to_csv(OUT / "dataset_no_supervisado.csv", index=False, encoding="utf-8")
resumen = {"registros": len(df), "registros_completos": len(d), "n_componentes": N_PC,
           "varianza_pc": [float(v) for v in var_exp[:4]], "varianza_acum_n": float(acum[N_PC - 1]),
           "k_elegido": K, "k_estaciones": KE, "variables": VARS, "tiempo_s": round(time.time() - T0)}
(OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
log("terminado")
