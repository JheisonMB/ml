import json
import nbformat
from nbclient import NotebookClient

nb_path = 'nlp-transformers-llms-kepler/llms_transformers_cuatro_tareas.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Patch cell 4 to handle Blackwell sm_120 architecture gracefully
cell4_code = '''# ── Librerías, reproducibilidad y detección de hardware ────────────────────
import gc
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 120)

# Semilla global: transformers.set_seed fija random, NumPy y PyTorch (CPU y GPU)
SEED = 42
transformers.set_seed(SEED)

# Selección de dispositivo: comprobación funcional de CUDA (compatibilidad sm_120)
cuda_available = torch.cuda.is_available()
cuda_functional = False
if cuda_available:
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        cuda_functional = True
    except Exception:
        cuda_functional = False

device = "cuda" if cuda_functional else "cpu"
pipe_device = 0 if device == "cuda" else -1   # convención de los pipelines de transformers

print(f"PyTorch {torch.__version__} | Transformers {transformers.__version__}")
print(f"Dispositivo de cómputo para ejecución: {device.upper()}")
if cuda_available:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU detectada: {props.name} | VRAM total: {props.total_memory / 1024**3:.1f} GB")
    if not cuda_functional:
        print("Nota técnica de hardware: GPU arquitectura Blackwell (sm_120) detectada.")
        print("La distribución binaria precompilada de PyTorch admite kernels hasta sm_90.")
        print("Para garantizar ejecución continua y estable, la sesión opera sobre CPU")
        print("utilizando el modelo de respaldo con paralelismo de 12 hilos.")
else:
    print("No se detectó GPU CUDA: se usará la CPU y un LLM de respaldo más pequeño en P4.")

# Registro de rendimiento: cada tarea añade aquí sus métricas para la tabla final (sección 7)
rendimiento = {}


def contar_parametros(model):
    """Parámetros del modelo en millones. num_parameters() de transformers
    cuenta correctamente los pesos empaquetados de los modelos de 4 bits."""
    if hasattr(model, "num_parameters"):
        return model.num_parameters() / 1e6
    return sum(p.numel() for p in model.parameters()) / 1e6


def memoria_modelo_gb(model):
    """Memoria que ocupan los pesos del modelo (GB)."""
    if hasattr(model, "get_memory_footprint"):
        return model.get_memory_footprint() / 1024**3
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3


def liberar_memoria():
    """Recolecta basura y vacía la caché de CUDA. Con 8 GB de VRAM conviene
    descargar cada modelo cuando su tarea termina."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
'''

nb.cells[4]['source'] = cell4_code

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Cell 4 successfully updated!')
