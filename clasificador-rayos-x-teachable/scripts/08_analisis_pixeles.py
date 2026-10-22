"""Caracteriza píxeles, resolución y profundidad de color del dataset.

Dos niveles:
    1. Cabeceras de las 10 702 imágenes de entrenamiento (modo, tamaño, formato,
       software y resolución en dpi que declara el PNG).
    2. Estadísticas de píxeles sobre una muestra aleatoria de N_PIXELES imágenes
       (semilla SEMILLA): canales, si R, G y B son iguales, alfa, valores únicos,
       media, desviación, percentiles, fracción de píxeles negros y blancos y
       histograma de intensidad acumulado.

Salidas:
    resultados/analisis_cabeceras.csv   una fila por imagen
    resultados/analisis_pixeles.csv     estadísticas por imagen de la muestra
    documento/figuras/ejemplo_histograma.png   imagen de ejemplo e histograma
    resumen en consola

Ejecutar desde la raíz del proyecto:
    venv/bin/python -I scripts/08_analisis_pixeles.py
"""
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
TRAIN = RAIZ / "data" / "raw" / "kaggle" / "kaggle" / "train"
SALIDA = RAIZ / "resultados"
FIGURAS = RAIZ / "documento" / "figuras"
N_PIXELES = 500
SEMILLA = 42

SALIDA.mkdir(exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)
archivos = sorted(TRAIN.glob("*.png"))

cabeceras = []
for ruta in archivos:
    with Image.open(ruta) as im:
        cabeceras.append({
            "archivo": ruta.name,
            "formato": im.format,
            "modo": im.mode,
            "ancho": im.size[0],
            "alto": im.size[1],
            "bytes": ruta.stat().st_size,
            "software": im.info.get("Software"),
            "dpi": round(im.info.get("dpi", (0, 0))[0]),
        })
cabeceras = pd.DataFrame(cabeceras)
cabeceras.to_csv(SALIDA / "analisis_cabeceras.csv", index=False)

muestra = pd.Series(archivos).sample(N_PIXELES, random_state=SEMILLA)
pixeles = []
histograma = np.zeros(256, dtype=np.int64)
for ruta in muestra:
    arreglo = np.asarray(Image.open(ruta))
    rojo, verde, azul, alfa = (arreglo[..., i] for i in range(4))
    gris = rojo.astype(np.float32)
    histograma += np.bincount(rojo.ravel(), minlength=256)
    pixeles.append({
        "archivo": ruta.name,
        "alto": arreglo.shape[0],
        "ancho": arreglo.shape[1],
        "canales": arreglo.shape[2],
        "tipo": str(arreglo.dtype),
        "rgb_iguales": bool(np.array_equal(rojo, verde) and np.array_equal(verde, azul)),
        "alfa_min": int(alfa.min()),
        "alfa_max": int(alfa.max()),
        "valores_unicos_rojo": int(len(np.unique(rojo))),
        "media": round(float(gris.mean()), 2),
        "desviacion": round(float(gris.std()), 2),
        "p1": float(np.percentile(gris, 1)),
        "p50": float(np.percentile(gris, 50)),
        "p99": float(np.percentile(gris, 99)),
        "fraccion_negro": round(float((rojo == 0).mean()), 4),
        "fraccion_blanco": round(float((rojo == 255).mean()), 4),
    })
pixeles = pd.DataFrame(pixeles)
pixeles.to_csv(SALIDA / "analisis_pixeles.csv", index=False)

ejemplo = np.asarray(Image.open(muestra.iloc[0]))[..., 0]
figura, (eje_imagen, eje_histograma) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.4]})
eje_imagen.imshow(ejemplo, cmap="gray", vmin=0, vmax=255)
eje_imagen.set_title(f"Ejemplo ({muestra.iloc[0].name})")
eje_imagen.axis("off")
eje_histograma.bar(np.arange(256), histograma / histograma.sum() * 100, width=1.0, color="#984C3D")
eje_histograma.set_xlabel("Intensidad (0 a 255)")
eje_histograma.set_ylabel("Porcentaje de píxeles")
eje_histograma.set_title(f"Histograma de {N_PIXELES} imágenes")
figura.tight_layout()
figura.savefig(FIGURAS / "ejemplo_histograma.png", dpi=150)

print("Cabeceras:", len(cabeceras), "imágenes")
print("  modos:", Counter(cabeceras.modo).most_common())
print("  tamaños:", Counter(zip(cabeceras.ancho, cabeceras.alto)).most_common(3))
print("  software:", Counter(cabeceras.software).most_common())
print("  dpi:", Counter(cabeceras.dpi).most_common())
print("  bytes promedio:", round(cabeceras.bytes.mean()), "| total:", cabeceras.bytes.sum())
print("Muestra de píxeles:", len(pixeles), "imágenes")
print("  RGB iguales en todas:", bool(pixeles.rgb_iguales.all()))
print("  alfa min/max:", pixeles.alfa_min.min(), pixeles.alfa_max.max())
print("  valores únicos en R (mediana, mín, máx):", int(pixeles.valores_unicos_rojo.median()), pixeles.valores_unicos_rojo.min(), pixeles.valores_unicos_rojo.max())
print("  intensidad media / desviación:", round(pixeles.media.mean(), 2), round(pixeles.desviacion.mean(), 2))
print("  percentiles p1/p50/p99:", round(pixeles.p1.mean(), 1), round(pixeles.p50.mean(), 1), round(pixeles.p99.mean(), 1))
print("  píxeles negros (0):", round(pixeles.fraccion_negro.mean() * 100, 2), "% | blancos (255):", round(pixeles.fraccion_blanco.mean() * 100, 2), "%")
