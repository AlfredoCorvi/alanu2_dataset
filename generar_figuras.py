"""Genera las figuras del documento a partir de resultados/ (ejecutar después de analisis_no_supervisado.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
FIG = RES / "figuras"
FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold"})
rs = json.loads((RES / "resumen.json").read_text(encoding="utf-8"))
K, KE = rs["k_elegido"], rs["k_estaciones"]
VARS = rs["variables"]
ETQ = {"tg": "TG (media)", "tn": "TN (mínima)", "tx": "TX (máxima)", "t10n": "T10N (mín. 10 cm)", "rh": "Lluvia (log)",
       "dr": "Duración lluvia (log)", "hum": "Humedad", "viento": "Viento", "ev": "Evapotransp."}
# nombres de los grupos (se asignaron después de ver sus perfiles)
NOM = {0: "Día cálido y seco", 1: "Día frío con helada", 2: "Día lluvioso y ventoso"}
COL = {0: "#F28E2B", 1: "#4E79A7", 2: "#59A14F"}
COLR = {0: "#B07AA1", 1: "#E15759", 2: "#76B7B2"}


def caja(ax, x, y, w, h, t, fc, fs=7):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec="black", lw=1))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, linespacing=1.3)


# ------------------------------------------------------------- 1. flujo de trabajo
def flujo():
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")
    pasos = [(16, 62, "1. Datos limpios\n(30 estaciones, 2016-2025)", "#DCE6F2"),
             (50, 62, "2. Selección de variables\n9 variables meteorológicas;\nsolo días completos", "#E4F0DA"),
             (84, 62, "3. Transformación\nlogaritmo en lluvia y duración;\nestandarización", "#FBE8D0"),
             (84, 40, "4. Reducción: PCA\n4 componentes conservan\n95.6 % de la varianza", "#EBDDF0"),
             (50, 40, "5. Optimizar k\nK-Means con k de 2 a 10\n(silueta, Calinski-Harabasz, Davies-Bouldin)", "#F7E7A6"),
             (16, 40, "6. Comparar algoritmos\nK-Means, jerárquico Ward\ny DBSCAN", "#D5E8E6"),
             (16, 17, "7. Interpretar grupos\nperfil, calendario y\nfenómenos agrícolas", "#DCE6F2"),
             (50, 17, "8. Visualizar\nPCA, t-SNE, tablas\ny gráficas", "#E4F0DA"),
             (84, 17, "9. Regiones climáticas\ngrupos de estaciones\n(jerárquico + K-Means)", "#FBE8D0")]
    for x, y, t, c in pasos:
        caja(ax, x, y, 29, 15, t, c)
    for (x1, y1, x2, y2) in [(30.5, 62, 35.5, 62), (64.5, 62, 69.5, 62), (84, 54, 84, 47.5), (69.5, 40, 64.5, 40),
                             (35.5, 40, 30.5, 40), (16, 32.5, 16, 24.5), (30.5, 17, 35.5, 17), (64.5, 17, 69.5, 17)]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="black"))
    ax.text(50, 3, "No hay variable objetivo: los grupos se descubren solo con las mediciones, sin etiquetas",
            ha="center", fontsize=8, style="italic")
    fig.savefig(FIG / "flujo_trabajo.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 2. PCA: varianza y cargas
def pca_varianza():
    v = pd.read_csv(RES / "pca_varianza.csv")
    c = pd.read_csv(RES / "pca_cargas.csv", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.arange(len(v)) + 1
    axs[0].bar(x, v["varianza_explicada"] * 100, color="#9ECAE1", ec="black", lw=0.5, label="Individual")
    axs[0].plot(x, v["varianza_acumulada"] * 100, color="#E15759", marker="o", ms=3, lw=1.4, label="Acumulada")
    axs[0].axhline(90, color="#555555", ls="--", lw=0.8)
    axs[0].text(9.4, 91.5, "90 %", ha="right", fontsize=7)
    axs[0].set_xticks(x); axs[0].set_xlabel("componente principal"); axs[0].set_ylabel("% de la varianza")
    axs[0].legend(frameon=False, fontsize=7, loc="center right"); axs[0].set_title("Varianza explicada")
    im = axs[1].imshow(c.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(c.columns)
    axs[1].set_yticks(range(len(c))); axs[1].set_yticklabels([ETQ[i] for i in c.index], fontsize=7)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            axs[1].text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(c.values[i, j]) > 0.5 else "black")
    axs[1].set_title("Cargas de cada variable"); axs[1].spines[:].set_visible(False)
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "pca_varianza_cargas.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 3. PCA: dispersión coloreada por grupo
def pca_dispersion():
    d = pd.read_csv(RES / "dataset_no_supervisado.csv", usecols=["PC1", "PC2", "cluster"])
    s = d.sample(14000, random_state=42)
    cargas = pd.read_csv(RES / "pca_cargas.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c in sorted(s["cluster"].unique()):
        m = s["cluster"] == c
        ax.scatter(s.loc[m, "PC1"], s.loc[m, "PC2"], s=3, alpha=0.35, color=COL[c], label=NOM[c], rasterized=True)
    cen = d.groupby("cluster")[["PC1", "PC2"]].mean()
    ax.scatter(cen["PC1"], cen["PC2"], marker="X", s=110, color="black", ec="white", zorder=5, label="Centroides")
    esc = 4.0
    for v in ["tg", "tn", "rh", "hum", "viento", "ev"]:
        ax.annotate("", xy=(cargas.loc[v, "PC1"] * esc, cargas.loc[v, "PC2"] * esc), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))
        ax.text(cargas.loc[v, "PC1"] * esc * 1.12, cargas.loc[v, "PC2"] * esc * 1.12, ETQ[v].split(" (")[0], fontsize=7,
                fontweight="bold", ha="center")
    ax.set_xlabel("PC1 (50.0 %): intensidad térmica"); ax.set_ylabel("PC2 (27.1 %): humedad y lluvia")
    ax.set_title("Días en el plano de las dos primeras componentes (muestra de 14,000 días)")
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], marker="o", ls="", color=COL[c], markersize=6, label=NOM[c]) for c in sorted(COL)]
    hs.append(Line2D([0], [0], marker="X", ls="", color="black", markersize=8, label="Centroides"))
    ax.legend(handles=hs, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "pca_dispersion.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 4. métricas vs. k
def metricas_k():
    d = pd.read_csv(RES / "k_means_busqueda_k.csv")
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.7))
    for ax, (col, tit, mejor) in zip(axs, [("Inercia", "Inercia (codo)", None), ("Silueta", "Silueta (mayor es mejor)", "max"),
                                           ("Calinski-Harabasz", "Calinski-Harabasz (mayor)", "max"),
                                           ("Davies-Bouldin", "Davies-Bouldin (menor)", "min")]):
        ax.plot(d["k"], d[col], marker="o", ms=3.5, color="#4E79A7", lw=1.4)
        ax.axvline(K, color="#E15759", ls="--", lw=1.2)
        ax.set_xticks(d["k"]); ax.tick_params(labelsize=6.5); ax.set_xlabel("k"); ax.set_title(tit, fontsize=7.5)
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle(f"Búsqueda del número de grupos (la línea roja marca k = {K})", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "metricas_k.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 5. antes vs. después
def antes_despues():
    a = pd.read_csv(RES / "antes_vs_despues.csv")
    k8 = pd.read_csv(RES / "k_means_busqueda_k.csv").set_index("k").loc[8]
    filas = [("Antes\nk = 8, sin\nescalar", a.iloc[0]), ("Escalado\n+ PCA\nk = 8", k8), (f"Después\nescalado + PCA\nk = {K}", a.iloc[1])]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0))
    cols = ["#9AA5B1", "#9ECAE1", "#59A14F"]
    for ax, (m, tit) in zip(axs, [("Silueta", "Silueta (mayor es mejor)"), ("Calinski-Harabasz", "Calinski-Harabasz (mayor)"),
                                  ("Davies-Bouldin", "Davies-Bouldin (menor es mejor)")]):
        v = [f[1][m] for f in filas]
        ax.bar(range(3), v, color=cols, ec="black", lw=0.5)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:,.3f}" if m != "Calinski-Harabasz" else f"{x:,.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels([f[0] for f in filas], fontsize=6.5); ax.set_title(tit, fontsize=8)
        ax.set_ylim(0, max(v) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG / "antes_despues.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 6. perfil de los grupos
def perfil():
    d = pd.read_csv(RES / "dataset_no_supervisado.csv", usecols=["cluster"] + [f"z_{v}" for v in VARS])
    z = d.groupby("cluster")[[f"z_{v}" for v in VARS]].mean()
    p = pd.read_csv(RES / "perfil_clusters.csv").set_index("cluster")
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    im = axs[0].imshow(z.values.T, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    axs[0].set_xticks(range(len(z))); axs[0].set_xticklabels([NOM[c].replace(" y ", "\ny ").replace("Día ", "Día\n") for c in z.index], fontsize=7)
    axs[0].set_yticks(range(len(VARS))); axs[0].set_yticklabels([ETQ[v] for v in VARS], fontsize=7)
    for i in range(z.shape[1]):
        for j in range(z.shape[0]):
            axs[0].text(j, i, f"{z.values[j, i]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(z.values[j, i]) > 0.9 else "black")
    axs[0].set_title("Perfil de cada grupo (valor estandarizado promedio)"); axs[0].spines[:].set_visible(False)
    fig.colorbar(im, ax=axs[0], fraction=0.04)
    axs[1].bar([NOM[c].replace(" y ", "\ny ").replace("Día ", "Día\n") for c in p.index], p["pct_dias"] * 100,
               color=[COL[c] for c in p.index], ec="black", lw=0.5)
    for i, (c, v) in enumerate(p["pct_dias"].items()):
        axs[1].text(i, v * 100, f"{v * 100:.0f} %", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axs[1].set_ylabel("% de los días"); axs[1].set_title("Tamaño de cada grupo"); axs[1].tick_params(axis="x", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "perfil_grupos.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 7. calendario y fenómenos
def calendario():
    m = pd.read_csv(RES / "clusters_por_mes.csv", index_col=0)
    c = pd.read_csv(RES / "concentracion_fenomenos.csv")
    nom = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.3, 1]})
    base = np.zeros(len(m))
    for col in m.columns:
        axs[0].bar(m.index, m[col] * 100, bottom=base, color=COL[int(col)], ec="black", lw=0.3, label=NOM[int(col)])
        base += m[col].values * 100
    axs[0].set_xticks(range(1, 13)); axs[0].set_xticklabels(nom, fontsize=7); axs[0].set_ylabel("% de los días del mes")
    axs[0].set_title("Grupos a lo largo del año"); axs[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1,
                                                                  bbox_to_anchor=(0.5, -0.12))
    fen = ["Helada\n(TN < 0 °C)", "Día lluvioso\n(>= 1 mm)", "Calor extremo\n(TX >= 30 °C)"]
    base = np.zeros(3)
    for col in [c_ for c_ in c.columns if c_.startswith("c")]:
        k = int(col[1:])
        axs[1].barh(range(3), c[col] * 100, left=base, color=COL[k], ec="black", lw=0.3)
        for i, v in enumerate(c[col].values * 100):
            if v > 8:
                axs[1].text(base[i] + v / 2, i, f"{v:.0f} %", ha="center", va="center", fontsize=7, fontweight="bold", color="white")
        base += c[col].values * 100
    axs[1].set_yticks(range(3)); axs[1].set_yticklabels(fen, fontsize=7); axs[1].invert_yaxis()
    axs[1].set_xlabel("% de los días con el fenómeno"); axs[1].set_title("¿En qué grupo ocurre cada fenómeno?")
    fig.tight_layout()
    fig.savefig(FIG / "calendario_fenomenos.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 8. t-SNE
def tsne():
    t = pd.read_csv(RES / "tsne_muestra.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in sorted(t["cluster"].unique()):
        m = t["cluster"] == c
        ax.scatter(t.loc[m, "x"], t.loc[m, "y"], s=4, alpha=0.6, color=COL[c], label=NOM[c], rasterized=True)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE de una muestra de 5,000 días, coloreada por grupo de K-Means")
    ax.legend(frameon=False, fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / "tsne.png", dpi=220, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- 9. regiones climáticas (estaciones)
def estaciones():
    e = pd.read_csv(RES / "estaciones_regiones.csv")
    L = np.load(RES / "estaciones_linkage.npy")
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    labels = e["estacion_id"].astype(str).tolist()
    corte = L[-(KE - 1), 2] - 1e-9
    dendrogram(L, labels=labels, orientation="left", color_threshold=0, ax=axs[0], leaf_font_size=6.5,
               above_threshold_color="#555555")
    reg = dict(zip(e["estacion_id"].astype(str), e["region"]))
    for t in axs[0].get_ymajorticklabels():
        t.set_color(COLR[reg[t.get_text()]]); t.set_fontweight("bold")
    axs[0].axvline(corte, color="#E15759", ls="--", lw=1)
    axs[0].set_xlabel("distancia (Ward)"); axs[0].set_title("Dendrograma de estaciones")
    axs[0].spines[["left"]].set_visible(False)
    for r in sorted(e["region"].unique()):
        m = e["region"] == r
        axs[1].scatter(e.loc[m, "pc1"], e.loc[m, "pc2"], s=40, color=COLR[r], ec="black", lw=0.5, label=f"Región {r + 1} ({m.sum()})")
    for _, f in e.iterrows():
        axs[1].text(f["pc1"] + 0.07, f["pc2"] + 0.07, str(int(f["estacion_id"])), fontsize=6)
    pv = pd.read_csv(RES / "estaciones_pca_varianza.csv")
    axs[1].set_xlabel(f"PC1 ({pv['varianza_explicada'][0] * 100:.0f} %)"); axs[1].set_ylabel(f"PC2 ({pv['varianza_explicada'][1] * 100:.0f} %)")
    axs[1].set_title("Estaciones en el plano de PCA"); axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "estaciones.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flujo(); pca_varianza(); pca_dispersion(); metricas_k(); antes_despues(); perfil(); calendario(); tsne(); estaciones()
    print("Figuras en", FIG)
