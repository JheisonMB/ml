"""Muestras para clasificar género: entrenamiento y prueba.

Clases (etiqueta de data/raw/train_gender.csv):
    mujer   0
    hombre  1
    Se usa el mapeo confirmado a simple vista en las imágenes.

Método:
    1. Entrenamiento: N_ENTRENO imágenes por clase, al azar con semilla SEMILLA.
    2. Prueba: N_PRUEBA imágenes por clase, al azar con semilla SEMILLA_PRUEBA.
    3. Se excluyen de ambos conjuntos las imágenes de data/prueba (prueba de edad)
       y entre sí, para que ninguna imagen aparezca en dos conjuntos.
    4. Salidas:
       data/genero/<clase>/            entrenamiento
       data/prueba_genero/<clase>/     prueba
       trazabilidad.csv en cada carpeta, con imageId, edad, género y clase.

Ejecutar desde la raíz del proyecto:
    venv/bin/python -I scripts/06_muestra_genero.py
"""
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
TRAIN = RAW / "kaggle" / "kaggle" / "train"
PRUEBA_EDAD = RAIZ / "data" / "prueba"
ENTRENO = RAIZ / "data" / "genero"
PRUEBA = RAIZ / "data" / "prueba_genero"
N_ENTRENO = 600
N_PRUEBA = 10
SEMILLA = 42
SEMILLA_PRUEBA = 7
CLASES = {0: "mujer", 1: "hombre"}

datos = pd.read_csv(RAW / "train_gender.csv").merge(pd.read_csv(RAW / "train_age.csv"), on="imageId")
excluidas = set(pd.read_csv(PRUEBA_EDAD / "trazabilidad.csv").imageId)
disponibles = datos[~datos.imageId.isin(excluidas)]


def copiar(carpeta_raiz, clase, ids):
    carpeta = carpeta_raiz / clase
    carpeta.mkdir(parents=True, exist_ok=True)
    for imagen_id in ids:
        shutil.copy2(TRAIN / f"{imagen_id:06d}.png", carpeta / f"{imagen_id:06d}.png")


def muestrear(n, semilla, origen):
    partes = []
    for valor, clase in CLASES.items():
        grupo = origen[origen.gender == valor].sample(n, random_state=semilla).assign(clase=clase)
        partes.append(grupo)
    return pd.concat(partes)


entreno = muestrear(N_ENTRENO, SEMILLA, disponibles)
for clase in CLASES.values():
    grupo = entreno[entreno.clase == clase]
    copiar(ENTRENO, clase, grupo.imageId)
ENTRENO.mkdir(parents=True, exist_ok=True)
entreno.to_csv(ENTRENO / "trazabilidad.csv", index=False)

restantes = disponibles[~disponibles.imageId.isin(entreno.imageId)]
prueba = muestrear(N_PRUEBA, SEMILLA_PRUEBA, restantes)
for clase in CLASES.values():
    grupo = prueba[prueba.clase == clase]
    copiar(PRUEBA, clase, grupo.imageId)
PRUEBA.mkdir(parents=True, exist_ok=True)
prueba.to_csv(PRUEBA / "trazabilidad.csv", index=False)

print("Entrenamiento:", {c: len(list((ENTRENO / c).glob("*.png"))) for c in CLASES.values()})
print("Prueba:", {c: len(list((PRUEBA / c).glob("*.png"))) for c in CLASES.values()})
