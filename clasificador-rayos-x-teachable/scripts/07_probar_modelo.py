"""Prueba un modelo exportado de Teachable Machine (Keras) sobre una carpeta de imágenes.

Lee modelo/<modelo>/keras_model.h5 y modelo/<modelo>/labels.txt (el zip
converted-keras de Teachable Machine). Cada subcarpeta de la carpeta de prueba
es una clase esperada, y su nombre se traduce a la etiqueta del modelo con
CARPETA_A_CLASE.

Preprocesamiento igual al del ejemplo de Teachable Machine: ajustar a 224x224
con recorte central y normalizar a [-1, 1].

Salidas:
    resultados/<modelo>.csv   una fila por imagen: archivo, esperada, predicha, confianza, acierto
    resumen en consola        aciertos por clase y total

Ejecutar desde la raíz del proyecto:
    venv/bin/python -I scripts/07_probar_modelo.py genero
    venv/bin/python -I scripts/07_probar_modelo.py genero --prueba data/prueba_genero
"""
import argparse
import os
from pathlib import Path

# Teachable Machine exporta en formato Keras 2: se usa tf-keras (pip install tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
EXTENSIONES = {".png", ".jpg", ".jpeg"}
TAMANO = (224, 224)

# Carpeta de prueba -> nombre de la clase en labels.txt de Teachable Machine.
CARPETA_A_CLASE = {"mujer": "Female", "hombre": "Male"}


def cargar_etiquetas(ruta):
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    # Cada línea trae "indice nombre" (por ejemplo "0 Female").
    return [linea.split(" ", 1)[1].strip() for linea in lineas if linea.strip()]


def preparar(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen = ImageOps.fit(imagen, TAMANO, Image.Resampling.LANCZOS)
    arreglo = np.asarray(imagen).astype(np.float32) / 127.5 - 1
    return arreglo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelo", help="nombre de la carpeta dentro de modelo/")
    parser.add_argument("--prueba", default="data/prueba_genero")
    args = parser.parse_args()

    carpeta_modelo = RAIZ / "modelo" / args.modelo
    modelo = tf.keras.models.load_model(carpeta_modelo / "keras_model.h5", compile=False)
    etiquetas = cargar_etiquetas(carpeta_modelo / "labels.txt")

    filas = []
    for carpeta in sorted((RAIZ / args.prueba).iterdir()):
        if not carpeta.is_dir() or carpeta.name not in CARPETA_A_CLASE:
            continue
        esperada = CARPETA_A_CLASE[carpeta.name]
        for imagen in sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES):
            entrada = np.expand_dims(preparar(imagen), axis=0)
            probabilidades = modelo.predict(entrada, verbose=0)[0]
            indice = int(np.argmax(probabilidades))
            predicha = etiquetas[indice]
            filas.append({
                "archivo": f"{carpeta.name}/{imagen.name}",
                "esperada": esperada,
                "predicha": predicha,
                "confianza": round(float(probabilidades[indice]), 4),
                "acierto": predicha == esperada,
            })

    resultados = pd.DataFrame(filas)
    salida = RAIZ / "resultados"
    salida.mkdir(exist_ok=True)
    resultados.to_csv(salida / f"{args.modelo}.csv", index=False)

    print(resultados.to_string(index=False))
    print()
    print(resultados.groupby("esperada")["acierto"].agg(["sum", "count"]).to_string())
    print(f"Total: {resultados.acierto.sum()} de {len(resultados)} correctas")


if __name__ == "__main__":
    main()
