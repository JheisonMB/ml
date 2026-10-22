"""Muestra de entrenamiento: N imagenes aleatorias por grupo de edad.

Grupos (edad en años, de data/raw/train_age.csv):
    joven         18 a 29
    adulto        30 a 59
    adulto_mayor  60 o mas

Método:
    1. Se asigna a cada imagen de entrenamiento su grupo según la edad.
    2. Se excluyen las imagenes de data/prueba, para que la prueba nunca
       se entrene.
    3. Por grupo se toman N imagenes al azar con semilla fija (SEMILLA),
       así la muestra se puede reproducir exactamente.
    4. Se copian a data/muestra/<grupo>/ y se guarda data/muestra/trazabilidad.csv
       con imageId, edad real y grupo asignado.

Parámetros:
    N_POR_CLASE   imagenes por grupo (600 en la corrida actual)
    SEMILLA       semilla del muestreo aleatorio (42)

Ejecutar desde la raíz del proyecto:
    venv/bin/python -I scripts/04_muestra_edad.py
"""
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
TRAIN = RAW / "kaggle" / "kaggle" / "train"
MUESTRA = RAIZ / "data" / "muestra"
PRUEBA = RAIZ / "data" / "prueba"
N_POR_CLASE = 600
SEMILLA = 42
CLASES = ["joven", "adulto", "adulto_mayor"]

edades = pd.read_csv(RAW / "train_age.csv")
edades["clase"] = pd.cut(
    edades.age,
    [18, 30, 60, 200],
    right=False,
    labels=CLASES,
)
en_prueba = pd.read_csv(PRUEBA / "trazabilidad.csv").imageId
disponibles = edades[~edades.imageId.isin(en_prueba)]

seleccion = []
for clase in CLASES:
    grupo = disponibles[disponibles.clase == clase].sample(N_POR_CLASE, random_state=SEMILLA)
    carpeta = MUESTRA / clase
    carpeta.mkdir(parents=True, exist_ok=True)
    for imagen_id in grupo.imageId:
        shutil.copy2(TRAIN / f"{imagen_id:06d}.png", carpeta / f"{imagen_id:06d}.png")
    seleccion.append(grupo)

pd.concat(seleccion).to_csv(MUESTRA / "trazabilidad.csv", index=False)
print("Muestra lista:", {c: len(list((MUESTRA / c).glob("*.png"))) for c in CLASES})
