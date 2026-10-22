#!/usr/bin/env bash
# Descarga el dataset SPR X-Ray Age and Gender desde Kaggle y lo descomprime en data/raw.
# Requiere el token de Kaggle en ~/.kaggle/kaggle.json (chmod 600).
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$RAIZ/venv/bin/kaggle"
DATASET="felipekitamura/spr-x-ray-age-and-gender-dataset"
DESTINO="$RAIZ/data/raw"

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "Falta ~/.kaggle/kaggle.json. Crea un token en https://www.kaggle.com/settings (API > Create New Token)." >&2
    exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

mkdir -p "$DESTINO"
# Pesa ~19.8 GB: el zip se guarda en data/raw y se descomprime en la misma carpeta.
"$KAGGLE" datasets download -d "$DATASET" -p "$DESTINO" --unzip

echo "Descarga completa en $DESTINO"
find "$DESTINO" -maxdepth 2 | head -n 40
