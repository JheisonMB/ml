import re
import sys
from pathlib import Path

import pymupdf

PDF = Path(sys.argv[1])
CAPITULO = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SALIDA = Path("data") / f"capitulo_{CAPITULO:02d}.txt"

ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
           "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]
X_PARRAFO = 60.0

doc = pymupdf.open(PDF)
inicio = ROMANOS[CAPITULO - 1]
fin = ROMANOS[CAPITULO] if CAPITULO < len(ROMANOS) else None

parrafos, actual, dentro = [], [], False
for pagina in doc:
    for bloque in pagina.get_text("dict")["blocks"]:
        for linea in bloque.get("lines", []):
            texto = "".join(s["text"] for s in linea["spans"]).strip()
            x0 = linea["bbox"][0]
            if not texto:
                continue
            if x0 > 250 and texto in ROMANOS:
                if texto == inicio:
                    dentro = True
                    continue
                if dentro and texto == fin:
                    dentro = False
                    break
            if not dentro:
                continue
            if texto.startswith("Cien años de soledad") or texto.startswith("Gabriel") or re.fullmatch(r"\d+", texto):
                continue
            if x0 > X_PARRAFO and actual:
                parrafos.append(" ".join(actual))
                actual = []
            actual.append(texto)
        if not dentro and parrafos:
            break
    if not dentro and parrafos:
        break
if actual:
    parrafos.append(" ".join(actual))

SALIDA.parent.mkdir(exist_ok=True)
SALIDA.write_text("\n\n".join(parrafos) + "\n", encoding="utf-8")
palabras = sum(len(p.split()) for p in parrafos)
print(f"capitulo {inicio}: {len(parrafos)} parrafos, {palabras} palabras -> {SALIDA}")
