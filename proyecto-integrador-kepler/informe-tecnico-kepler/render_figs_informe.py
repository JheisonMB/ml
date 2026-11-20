"""Regenera las figuras del informe técnico (APA 7) a 200 dpi desde los artefactos
guardados de los notebooks NB3-NB5. Salida: informe-tecnico-kepler/assets/images/."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

ROOT = Path("/home/jheisonmblivecom/Academic/Deriverables/ML/proyecto-integrador-kepler")
DATA = ROOT / "data"
OUT = ROOT / "informe-tecnico-kepler" / "assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)

CONF, CAND, FP = "CONFIRMED", "CANDIDATE", "FALSE POSITIVE"
COLORES = {CONF: "#2a78d6", FP: "#eb6834", CAND: "#1baf7a"}
GRIS = "#8a8984"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "grid.alpha": 0.35, "savefig.bbox": "tight"})

base = json.loads((DATA / "koi_metricas_baseline.json").read_text())
avanz = json.loads((DATA / "koi_metricas_modelos.json").read_text())
splits = json.loads((DATA / "koi_splits.json").read_text())
df = pd.read_parquet(DATA / "koi_clean.parquet")
clasif = pd.read_parquet(DATA / "koi_candidatos_clasificados.parquet")
raw = pd.read_csv(DATA / "koi_cumulative_raw.csv", low_memory=False,
                  dtype={"koi_quarters": str, "kepoi_name": str, "kepler_name": str})
prob_cand = clasif["prob_planeta"].values
pred_cand = clasif["pred_planeta"].values
cand_mult = pd.read_parquet(DATA / "koi_candidatos.parquet")["sistema_multiple"].values
perm = pd.Series(avanz["permutacion_top15"])
mat = np.array([[avanz["matriz_confusion_prueba"]["tn"], avanz["matriz_confusion_prueba"]["fp"]],
                [avanz["matriz_confusion_prueba"]["fn"], avanz["matriz_confusion_prueba"]["tp"]]])
NOMBRE = avanz["ganador"]

# ---------------------------------------------------------------- fig_comparacion
nombres = ["Baseline\nreg. logística", "SVM (RBF)", "Bosque aleatorio", "Gradient\nboosting"]
f1v = [base["validacion"]["F1 macro"], avanz["modelos"]["SVM (RBF)"]["f1_macro_validacion"],
       avanz["modelos"]["Bosque aleatorio"]["f1_macro_validacion"],
       avanz["modelos"]["Gradient boosting"]["f1_macro_validacion"]]
auv = [base["validacion"]["AUC"], avanz["modelos"]["SVM (RBF)"]["auc_validacion"],
       avanz["modelos"]["Bosque aleatorio"]["auc_validacion"],
       avanz["modelos"]["Gradient boosting"]["auc_validacion"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, vals, titulo, lim in [(axes[0], f1v, "F1 macro en validación", (0.93, 0.995)),
                              (axes[1], auv, "AUC en validación", (0.988, 0.999))]:
    cols = [COLORES[CONF] if n == "Gradient\nboosting" else GRIS for n in nombres]
    barras = ax.bar(nombres, vals, color=cols, width=0.55)
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (lim[1] - lim[0]) * 0.06, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(*lim)
    ax.set_title(titulo)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.savefig(OUT / "fig_comparacion.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_matriz
fig, ax = plt.subplots(figsize=(5.2, 4.0))
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax,
            annot_kws={"size": 13},
            xticklabels=["Pred: FP", "Pred: planeta"],
            yticklabels=["Real: FP", "Real: planeta"])
ax.set_title(f"Matriz de confusión · {NOMBRE} (prueba, n = 1 518)")
plt.tight_layout()
plt.savefig(OUT / "fig_matriz.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_roc
X = df.drop(columns="es_planeta")
y = df["es_planeta"].astype(int)
X_test = X.loc[splits["test"]]
y_test = y.loc[splits["test"]]
modelo_final = joblib.load(DATA / "koi_modelo_final.joblib")
baseline_final = joblib.load(DATA / "koi_modelo_baseline.joblib")
fpr_g, tpr_g, _ = roc_curve(y_test, modelo_final.predict_proba(X_test)[:, 1])
fpr_b, tpr_b, _ = roc_curve(y_test, baseline_final.predict_proba(X_test)[:, 1])
fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.plot(fpr_b, tpr_b, color=GRIS, lw=1.6, ls="--",
        label=f"Baseline · reg. logística (AUC = {base['prueba']['AUC']:.3f})")
ax.plot(fpr_g, tpr_g, color=COLORES[CONF], lw=2.2,
        label=f"{NOMBRE} (AUC = {avanz['ganador_prueba']['AUC']:.3f})")
ax.plot([0, 1], [0, 1], color="#c4c3bf", lw=1, label="Azar")
ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de acierto (planetas)",
       title="Curva ROC en prueba")
ax.legend(frameon=True, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig_roc.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_importancia
p = perm.sort_values()
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(p.index[::-1], p.values[::-1], color=COLORES[CONF], height=0.6)
for y_, v in enumerate(p.values[::-1]):
    ax.text(v, y_, f"  {v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Caída del F1 macro al barajar la variable (validación)")
ax.set_title("Importancia por permutación (15 variables con más peso)")
ax.tick_params(axis="y", labelsize=8.5)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig(OUT / "fig_importancia.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_periodo_radio
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for clase, etiqueta in [(FP, "Falso positivo"), (CONF, "Confirmado")]:
    sub = raw.loc[raw["koi_disposition"] == clase]
    sub = sub[(sub["koi_prad"] > 0) & (sub["koi_period"] > 0)]
    ax.scatter(sub["koi_period"], sub["koi_prad"], s=5, alpha=0.35,
               color=COLORES[clase], label=etiqueta, rasterized=True)
sub_c = clasif.loc[clasif["pred_planeta"] == 1]
sub_c = sub_c[(sub_c["koi_prad"] > 0) & (sub_c["koi_period"] > 0)]
ax.scatter(sub_c["koi_period"], sub_c["koi_prad"], s=14, alpha=0.75, facecolors="none",
           edgecolors=COLORES[CAND], linewidths=0.9,
           label=f"Candidatos «planeta» ({len(sub_c):,})")
for r, nombre in [(1, "Tierra"), (3.9, "Neptuno"), (11.2, "Júpiter")]:
    ax.axhline(r, color=GRIS, lw=0.8, ls=":")
    ax.text(800, r * 1.09, nombre, color="#52514e", fontsize=8, ha="right")
ax.set(xscale="log", yscale="log", xlim=(0.25, 1000), ylim=(0.3, 300),
       xlabel="Periodo orbital [días]", ylabel="Radio [R⊕]",
       title="El mapa de los exoplanetas, con los candidatos clasificados")
ax.legend(markerscale=3, frameon=True, loc="upper left")

ax = axes[1]
cnt = df[["sistema_multiple", "es_planeta"]].copy()
cnt["n_senales"] = np.where(cnt["sistema_multiple"] == 1, "2+ señales", "1 señal")
tasa_real = cnt.groupby("n_senales")["es_planeta"].mean() * 100
tasa_pred = pd.Series(pred_cand).groupby(
    pd.Series(np.where(cand_mult == 1, "2+ señales", "1 señal"))).mean() * 100
x = np.arange(2)
ax.bar(x - 0.18, tasa_real.loc[["1 señal", "2+ señales"]], width=0.34,
       color=COLORES[CONF], label="% reales que son planetas")
ax.bar(x + 0.18, tasa_pred.reindex(["1 señal", "2+ señales"]), width=0.34,
       color=COLORES[CAND], label="% de candidatos que el modelo dice «planeta»")
for xi, v in zip(x - 0.18, tasa_real.loc[["1 señal", "2+ señales"]]):
    ax.text(xi, v + 1.5, f"{v:.0f} %", ha="center", fontsize=9)
for xi, v in zip(x + 0.18, tasa_pred.reindex(["1 señal", "2+ señales"])):
    ax.text(xi, v + 1.5, f"{v:.0f} %", ha="center", fontsize=9)
ax.set_xticks(x, ["1 señal", "2+ señales"])
ax.set_ylabel("%")
ax.set_ylim(0, 108)
ax.set_title("La multiplicidad confirma la coherencia del modelo")
ax.legend(frameon=True, fontsize=8, loc="upper left")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.savefig(OUT / "fig_periodo_radio.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_zonas
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.hist(prob_cand, bins=np.linspace(0, 1, 31), color=COLORES[CAND], alpha=0.9)
ax.axvline(0.5, color="#52514e", ls="--", lw=1.2)
ax.text(0.51, ax.get_ylim()[1] * 0.9, "umbral 0.5", color="#52514e", fontsize=9)
ax.set(xlabel="Probabilidad predicha de «planeta»", ylabel="Candidatos",
       title="La mayoría de los candidatos cae en una zona decidida")
ax = axes[1]
zonas = pd.cut(prob_cand, [0, 0.25, 0.5, 0.75, 1.0],
               labels=["0–25 %", "25–50 %", "50–75 %", "75–100 %"])
zona_n = zonas.value_counts().reindex(["0–25 %", "25–50 %", "50–75 %", "75–100 %"])
colores_zona = [COLORES[FP], GRIS, GRIS, COLORES[CAND]]
barras = ax.bar(zona_n.index, zona_n.values, color=colores_zona, width=0.6)
for b, n in zip(barras, zona_n.values):
    ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
ax.set(ylabel="Candidatos", title="Reparto por zona de confianza")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.savefig(OUT / "fig_zonas.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_tablero
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
ax = axes[0, 0]
conteo = raw["koi_disposition"].value_counts().reindex([CONF, FP, CAND])
NOMBRES = {CONF: "Confirmados", FP: "Falsos positivos", CAND: "Candidatos"}
ax.barh([NOMBRES[c] for c in conteo.index][::-1],
        conteo.values[::-1], color=[COLORES[c] for c in conteo.index[::-1]], height=0.55)
for y_, n in enumerate(conteo.values[::-1]):
    ax.text(n, y_, f"  {n:,}", va="center", fontsize=9)
ax.set_xlim(0, conteo.max() * 1.2)
ax.set_title(f"① El problema: {conteo.sum():,} señales, {conteo[CONF]:,} planetas confirmados")
ax.grid(axis="y", visible=False)

ax = axes[0, 1]
comp = [base["prueba_trivial"]["clase_mayoritaria"]["F1 macro"],
        base["prueba"]["F1 macro"], avanz["ganador_prueba"]["F1 macro"]]
ax.bar(["Trivial", "Baseline\n(NB3)", "Final\n(NB4)"], comp,
       color=[GRIS, COLORES[FP], COLORES[CONF]], width=0.5)
for i, v in enumerate(comp):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0.3, 1.02)
ax.set_title("② El resultado: F1 macro en prueba")
ax.grid(axis="x", visible=False)

ax = axes[1, 0]
top8 = p.tail(8)
ax.barh(top8.index[::-1], top8.values[::-1], color=COLORES[CONF], height=0.55)
ax.set_title("③ El método: importancia por permutación")
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="y", visible=False)

ax = axes[1, 1]
ax.hist(prob_cand, bins=np.linspace(0, 1, 25), color=COLORES[CAND])
ax.axvline(0.5, color="#52514e", ls="--", lw=1.1)
ax.set(xlabel="probabilidad de «planeta»", ylabel="candidatos")
ax.set_title(f"④ La propuesta: {int(pred_cand.sum()):,} de {len(clasif):,} candidatos como planetas")
ax.grid(axis="y", visible=False)
plt.suptitle("Proyecto Kepler KOI · panorama completo del proyecto",
             fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(OUT / "fig_tablero.png".replace("_", "-"))
plt.close()

# ---------------------------------------------------------------- fig_pipeline
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

pasos = [
    ("CSV crudo", "9 564 x 153", GRIS),
    ("NB1 · EDA", "diagnóstico", COLORES[FP]),
    ("NB2 · limpieza", "7 587 x 67", COLORES[FP]),
    ("NB3 · partición\ny baseline", "F1 0.946", COLORES[CONF]),
    ("NB4 · búsqueda\ny ganador", "F1 0.983", COLORES[CONF]),
    ("NB5 · producto", "653 / 1 977", COLORES[CAND]),
]
fig, ax = plt.subplots(figsize=(12, 2.1))
ax.set_xlim(0, len(pasos) * 2)
ax.set_ylim(0, 1.6)
ax.axis("off")
for i, (titulo, sub, color) in enumerate(pasos):
    x = i * 2 + 0.15
    caja = FancyBboxPatch((x, 0.55), 1.7, 0.75, boxstyle="round,pad=0.06",
                          fc="white", ec=color, lw=2)
    ax.add_patch(caja)
    ax.text(x + 0.85, 1.08, titulo, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="#3a3936")
    ax.text(x + 0.85, 0.72, sub, ha="center", va="center", fontsize=8.5, color="#8a8984")
    if i < len(pasos) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.88, 0.92), (x + 2.26, 0.92),
                                     arrowstyle="-|>", mutation_scale=22,
                                     color="#52514e", lw=2.2))
plt.tight_layout()
plt.savefig(OUT / "fig_pipeline.png".replace("_", "-"))
plt.close()

print("Figuras generadas:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
