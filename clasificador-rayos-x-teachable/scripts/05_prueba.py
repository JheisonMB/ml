"""Saca 10 imagenes aleatorias por grupo de edad para probar el modelo.

Solo usa imagenes que no estan en data/muestra (el entrenamiento).
Ejecutar con: venv/bin/python -I scripts/05_prueba.py
"""
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
TRAIN = RAW / "kaggle" / "kaggle" / "train"
MUESTRA = RAIZ / "data" / "muestra"
PRUEBA = RAIZ / "data" / "prueba"
N_POR_CLASE = 10
SEMILLA = 7
CLASES = ["joven", "adulto", "adulto_mayor"]

entrenamiento = pd.read_csv(MUESTRA / "trazabilidad.csv")
edades = pd.read_csv(RAW / "train_age.csv")
edades["clase"] = pd.cut(
    edades.age,
    [18, 30, 60, 200],
    right=False,
    labels=CLASES,
)
disponibles = edades[~edades.imageId.isin(entrenamiento.imageId)]

seleccion = []
for clase in CLASES:
    grupo = disponibles[disponibles.clase == clase].sample(N_POR_CLASE, random_state=SEMILLA)
    carpeta = PRUEBA / clase
    carpeta.mkdir(parents=True, exist_ok=True)
    for imagen_id in grupo.imageId:
        shutil.copy2(TRAIN / f"{imagen_id:06d}.png", carpeta / f"{imagen_id:06d}.png")
    seleccion.append(grupo)

pd.concat(seleccion).to_csv(PRUEBA / "trazabilidad.csv", index=False)
print("Prueba lista:", {c: len(list((PRUEBA / c).glob("*.png"))) for c in CLASES})
