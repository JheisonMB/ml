"""Inventaria los archivos descargados y resume las etiquetas disponibles.

Ejecutar con: venv/bin/python -I scripts/02_inspeccionar.py
"""
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".dcm"}

imagenes = [p for p in RAW.rglob("*") if p.suffix.lower() in EXTENSIONES_IMAGEN]
print(f"Imagenes encontradas: {len(imagenes)}")
print("Carpetas de primer nivel:", sorted({p.name for p in RAW.iterdir() if p.is_dir()}))

tablas = sorted(RAW.rglob("*.csv"))
print(f"\nArchivos CSV: {len(tablas)}")
for csv in tablas:
    df = pd.read_csv(csv)
    print(f"\n--- {csv.relative_to(RAW)}  ({len(df)} filas)")
    print("columnas:", list(df.columns))
    print(df.head(3).to_string(index=False))
    for columna in df.columns:
        if df[columna].nunique() <= 20:
            print(f"  distribucion de '{columna}':")
            print(df[columna].value_counts(dropna=False).to_string())

if not imagenes:
    print("\nNo se encontraron imagenes. Revisa la salida de 01_descargar.sh.")
