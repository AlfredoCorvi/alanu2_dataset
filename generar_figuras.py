"""Genera las figuras del documento a partir de la carpeta resultados/ (ejecutar después de analisis_supervisado.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
FIG = RES / "figuras"
FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold"})
C = {"Regresión logística": "#4C78A8", "Random Forest": "#59A14F", "Gradient Boosting": "#E15759",
     "Regresión lineal": "#4C78A8", "base": "#9AA5B1"}

pred = pd.read_csv(RES / "predicciones_test.csv", parse_dates=["fecha"])
tc = pd.read_csv(RES / "test_clasificacion.csv")
tr = pd.read_csv(RES / "test_regresion.csv")
mm = json.loads((RES / "mejores_modelos.json").read_text(encoding="utf-8"))
umb = json.loads((RES / "umbrales.json").read_text(encoding="utf-8"))
cm = json.loads((RES / "matriz_confusion.json").read_text(encoding="utf-8"))
res = json.loads((RES / "resumen.json").read_text(encoding="utf-8"))
mc, mr = mm["mejor_clasificador"], mm["mejor_regresor"]
CLF = ["Regresión logística", "Random Forest", "Gradient Boosting"]


def caja(ax, x, y, w, h, t, fc, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=fc, ec="black", lw=1))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


# ---------------------------------------------------------------- 1. flujo de trabajo
def flujo():
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")
    pasos = [
        (16, 62, "1. Datos limpios\n(30 estaciones, 2016-2025)", "#DCE6F2"),
        (50, 62, "2. Variables de entrada\n(hoy, días previos, climatología,\nestacionalidad)", "#E4F0DA"),
        (84, 62, "3. Objetivos del día siguiente\nhelada_manana (clasificación)\ntn_manana (regresión)", "#FBE8D0"),
        (84, 40, "4. Partición cronológica\nentrenamiento 2016-2022\nprueba 2023-2025", "#EBDDF0"),
        (50, 40, "5. Validación cruzada temporal\n(ventana creciente: 2019-2022)\ny búsqueda aleatoria", "#F7E7A6"),
        (16, 40, "6. Selección y umbral\nmejor configuración por modelo;\numbral con datos de validación", "#D5E8E6"),
        (16, 17, "7. Entrenamiento final\ncon todo 2016-2022", "#DCE6F2"),
        (50, 17, "8. Evaluación en prueba\nmétricas, tablas y gráficas", "#E4F0DA"),
        (84, 17, "9. Análisis\nerror por mes e importancia\nde variables", "#FBE8D0"),
    ]
    for x, y, t, c in pasos:
        caja(ax, x, y, 29, 15, t, c, fs=7)
    for (x1, y1, x2, y2) in [(29.5, 62, 36.5, 62), (63.5, 62, 70.5, 62), (84, 54, 84, 47.5), (70.5, 40, 63.5, 40),
                             (36.5, 40, 29.5, 40), (16, 32.5, 16, 24.5), (29.5, 17, 36.5, 17), (63.5, 17, 70.5, 17)]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="black"))
    # la flecha 3->4 pasa por abajo
    ax.add_patch(FancyArrowPatch((84, 54), (84, 47.5), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="black"))
    ax.text(50, 3, "Todas las decisiones (variables, hiperparámetros, umbral) se toman SIN usar los datos de prueba",
            ha="center", fontsize=8, style="italic")
    fig.savefig(FIG / "flujo_trabajo.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 2. partición temporal
def particion():
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.set_xlim(2016, 2026); ax.set_ylim(0, 5); ax.axis("off")
    ax.add_patch(Rectangle((2016, 3.2), 7, 0.9, fc="#9ECAE1", ec="black"))
    ax.text(2019.5, 3.65, "Entrenamiento 2016-2022", ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((2023, 3.2), 3, 0.9, fc="#FDBB84", ec="black"))
    ax.text(2024.5, 3.65, "Prueba 2023-2025", ha="center", va="center", fontweight="bold")
    for i, v in enumerate([2019, 2020, 2021, 2022]):
        y = 2.3 - i * 0.55
        ax.add_patch(Rectangle((2016, y), v - 2016, 0.4, fc="#C6DBEF", ec="black", lw=0.6))
        ax.add_patch(Rectangle((v, y), 1, 0.4, fc="#A1D99B", ec="black", lw=0.6))
        ax.text(2026.1, y + 0.2, f"Pliegue {i + 1}", va="center", fontsize=7)
    ax.text(2016.1, 0.05, "azul claro: entrena el pliegue   |   verde: valida el pliegue", fontsize=7, style="italic")
    for a in range(2016, 2027):
        ax.text(a, 4.45, str(a), ha="center", fontsize=7)
    fig.savefig(FIG / "particion_temporal.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 3. comparación de clasificadores
def comparacion_clf():
    d = tc[(tc["Versión"].isin(["-", "Optimizado"]))].copy()
    d = d[~d["Modelo"].str.contains("mayoritaria")]
    nombres = [m.replace("Línea base: ", "Base: ") for m in d["Modelo"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(d)); w = 0.2
    for i, (met, col) in enumerate([("Precision", "#4C78A8"), ("Recall", "#E15759"), ("F1", "#59A14F"), ("PR-AUC", "#B07AA1")]):
        ax.bar(x + (i - 1.5) * w, d[met], w, label=met, color=col, ec="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) if len(n) > 14 else n for n in nombres], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("valor (0 a 1)")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Clasificación de helada mañana: modelos optimizados y líneas base (prueba 2023-2025)")
    fig.savefig(FIG / "comparacion_clasificacion.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 4. curvas PR y ROC
def curvas():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
    y = pred["helada_manana"].values
    for m in CLF:
        p = pred[f"p_{m}"].values
        pr, rc, _ = precision_recall_curve(y, p)
        axs[0].plot(rc, pr, color=C[m], lw=1.6, label=m)
        fpr, tpr, _ = roc_curve(y, p)
        axs[1].plot(fpr, tpr, color=C[m], lw=1.6, label=m)
    hp = pred["helada_hoy"].values
    tp = ((hp == 1) & (y == 1)).sum(); fp = ((hp == 1) & (y == 0)).sum()
    axs[0].scatter([tp / y.sum()], [tp / max(1, tp + fp)], color="black", zorder=5, label="Persistencia")
    axs[1].scatter([fp / (y == 0).sum()], [tp / y.sum()], color="black", zorder=5, label="Persistencia")
    axs[0].axhline(y.mean(), color="#9AA5B1", ls="--", lw=1)
    axs[0].text(0.02, y.mean() + 0.02, "azar", color="#555555", fontsize=7)
    axs[1].plot([0, 1], [0, 1], color="#9AA5B1", ls="--", lw=1)
    axs[0].set_xlabel("Recall (heladas detectadas)"); axs[0].set_ylabel("Precision"); axs[0].set_title("Curva precisión-recall")
    axs[1].set_xlabel("Tasa de falsos positivos"); axs[1].set_ylabel("Recall"); axs[1].set_title("Curva ROC")
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "curvas_pr_roc.png", dpi=220, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 5. matriz de confusión
def matriz():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
    y = pred["helada_manana"].values.astype(int)
    for ax, (tit, umbral) in zip(axs, [(f"Umbral F1 ({umb[mc]['umbral_f1']:.2f})", umb[mc]["umbral_f1"]),
                                       (f"Umbral recall 90 % ({umb[mc]['umbral_recall90']:.2f})", umb[mc]["umbral_recall90"])]):
        yh = (pred[f"p_{mc}"].values >= umbral).astype(int)
        M = np.array([[((y == 0) & (yh == 0)).sum(), ((y == 0) & (yh == 1)).sum()],
                      [((y == 1) & (yh == 0)).sum(), ((y == 1) & (yh == 1)).sum()]])
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Sin helada", "Helada"]); ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sin helada", "Helada"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(tit, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(True)
    fig.suptitle(f"Matriz de confusión: {mc} (prueba 2023-2025)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "matriz_confusion.png", dpi=220, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 6. regresión: real vs predicho y residuos
def regresion():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
    yh = pred[f"y_{mr}"].values; y = pred["tn_manana"].values
    hb = axs[0].hexbin(y, yh, gridsize=45, cmap="Blues", mincnt=1, bins="log")
    lim = [min(y.min(), yh.min()), max(y.max(), yh.max())]
    axs[0].plot(lim, lim, color="#E15759", lw=1.2)
    axs[0].set_xlabel("TN real de mañana (°C)"); axs[0].set_ylabel("TN predicha (°C)")
    axs[0].set_title(f"Real vs. predicho ({mr})")
    fig.colorbar(hb, ax=axs[0], label="días (escala log)", fraction=0.046)
    r = yh - y
    axs[1].hist(r, bins=60, color="#9ECAE1", ec="black", lw=0.3)
    axs[1].axvline(0, color="#E15759", lw=1.2)
    axs[1].set_xlabel("Error = predicho - real (°C)"); axs[1].set_ylabel("días")
    axs[1].set_title(f"Distribución del error (MAE {np.abs(r).mean():.2f} °C)")
    fig.tight_layout()
    fig.savefig(FIG / "regresion_real_vs_predicho.png", dpi=220, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 7. comparación de regresores
def comparacion_reg():
    d = tr[tr["Versión"].isin(["-", "Optimizado"])].copy()
    nombres = [m.replace("Línea base: ", "Base: ") + ("" if v in ("-",) else "") for m, v in zip(d["Modelo"], d["Versión"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
    cols = [C["base"] if "Base" in n else C.get(m, "#4C78A8") for n, m in zip(nombres, d["Modelo"])]
    for ax, met in zip(axs, ["MAE", "RMSE"]):
        ax.barh(range(len(d)), d[met], color=cols, ec="black", lw=0.5)
        for i, v in enumerate(d[met]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(nombres if met == "MAE" else [], fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel(f"{met} (°C), menor es mejor"); ax.set_title(met)
    fig.tight_layout()
    fig.savefig(FIG / "comparacion_regresion.png", dpi=220, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 8. por mes
def por_mes():
    d = pd.read_csv(RES / "por_mes.csv")
    nom = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
    axs[0].bar(d["mes"], d["heladas_reales"], color="#9ECAE1", ec="black", lw=0.4)
    ax2 = axs[0].twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(d["mes"], d["recall"], color="#E15759", marker="o", lw=1.4, label="Recall")
    ax2.plot(d["mes"], d["precision"], color="#59A14F", marker="s", lw=1.4, label="Precision")
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=6.5, frameon=False, loc="lower left")
    axs[0].set_xticks(range(1, 13)); axs[0].set_xticklabels(nom, fontsize=7)
    axs[0].set_ylabel("días con helada real"); axs[0].set_title(f"Heladas y desempeño por mes ({mc})")
    axs[1].bar(d["mes"], d["MAE_reg"], color="#FDBB84", ec="black", lw=0.4)
    axs[1].set_xticks(range(1, 13)); axs[1].set_xticklabels(nom, fontsize=7)
    axs[1].set_ylabel("MAE (°C)"); axs[1].set_title(f"Error de TN por mes ({mr})")
    fig.tight_layout()
    fig.savefig(FIG / "desempeno_por_mes.png", dpi=220, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 9. importancias
def importancias():
    ETQ = {"tg": "TG hoy", "tn": "TN hoy", "tx": "TX hoy", "t10n": "T10N hoy", "rh": "Lluvia hoy", "traza": "Traza lluvia",
           "dr": "Duración lluvia", "hum": "Humedad", "viento": "Viento", "ev": "Evapotransp.", "tn_l1": "TN ayer",
           "tn_l2": "TN anteayer", "tg_l1": "TG ayer", "tx_l1": "TX ayer", "rh_l1": "Lluvia ayer", "hum_l1": "Humedad ayer",
           "dtn": "Cambio TN", "dtg": "Cambio TG", "tn_min3": "TN mín. 3 días", "rango": "Rango TX-TN",
           "doy_sin": "Estac. (seno)", "doy_cos": "Estac. (coseno)", "clim_tn": "Climatología TN", "anom_tn": "Anomalía TN"}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, archivo, tit in [(axs[0], "importancias_clf.csv", f"Clasificación ({mc})"),
                             (axs[1], "importancias_reg.csv", f"Regresión ({mr})")]:
        d = pd.read_csv(RES / archivo).head(10).iloc[::-1]
        ax.barh([ETQ.get(v, v) for v in d["variable"]], d["importancia"], xerr=d["desv"], color="#9ECAE1", ec="black", lw=0.4)
        ax.set_title(tit); ax.set_xlabel("pérdida de desempeño al permutar la variable")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "importancia_variables.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flujo(); particion(); comparacion_clf(); curvas(); matriz(); regresion(); comparacion_reg(); por_mes(); importancias()
    print("Figuras en", FIG)
