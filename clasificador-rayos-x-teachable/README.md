# Clasificador de radiografías de tórax (Teachable Machine)

Modelo de clasificación de imágenes sin código con Teachable Machine, entrenado con el
dataset **SPR X-Ray Age and Gender** (Kaggle, `felipekitamura/spr-x-ray-age-and-gender-dataset`).
El informe está en `documento/` (compilar con `texforge build`).

## Scripts

Se ejecutan desde esta carpeta con `venv/bin/python -I scripts/<script>.py`.

| Script | Qué hace |
|---|---|
| `01_descargar.sh` | Descarga el dataset con la CLI de Kaggle (requiere `~/.kaggle/kaggle.json`) |
| `02_inspeccionar.py` | Inventario de archivos y distribución de etiquetas |
| `04_muestra_edad.py` | Muestra de entrenamiento por grupo de edad (joven, adulto, adulto mayor) |
| `05_prueba.py` | 10 imágenes de prueba por grupo de edad, fuera del entrenamiento |
| `06_muestra_genero.py` | Muestras de entrenamiento y prueba por género |
| `07_probar_modelo.py` | Prueba el modelo exportado de Teachable Machine (Keras) |
| `08_analisis_pixeles.py` | Resolución, píxeles y profundidad de color del dataset |

Dependencias: `kaggle`, `pandas`, `pillow`, `matplotlib`, `tensorflow-cpu` y `tf-keras`.

## Datos y licencia

- Licencia CC BY-NC-ND 4.0: uso no comercial y sin redistribuir los datos.
- Los términos de la competencia prohíben redistribuir los datos y reidentificar pacientes.
  Por eso `data/` y `modelo/` están en `.gitignore`.
- Las muestras se generan con semilla fija a partir de `data/raw/` (el zip de Kaggle
  descomprimido); la muestra de 200 imágenes por clase de la primera corrida se sobrescribió
  con la de 600.
