"""
Update transformers_llms_auditoria_kepler.ipynb to integrate 2022-2026 SOTA literature,
RTX 5060 hardware audit & VRAM budgeting, modern classification paradigm,
and complete academic citations.
"""
import json
from pathlib import Path

nb_path = Path('nlp-transformers-llms-kepler/transformers_llms_auditoria_kepler.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Project Scope
nb['cells'][0]['source'] = [
    "# Cuatro Arquitecturas Transformer en un Flujo Real de Auditoría Espacial\n",
    "## Extracción Visual, Clasificación Semántica, Embeddings y Generación Causal sobre Telemetría del Telescopio Kepler (NASA-TM-2013-217983)\n",
    "\n",
    "**Autor:** Jheison Martinez Bolivar  \n",
    "**Maestría:** Inteligencia Artificial  \n",
    "**Fecha:** Octubre 2026  \n",
    "**Documento Fuente:** Memorando Oficial de Telemetría e Incidentes *NASA Ames Research Center* (`NASA-TM-2013-217983`)  \n",
    "**Hardware de Referencia:** NVIDIA GeForce RTX 5060 (8.55 GB VRAM, Arquitectura Blackwell sm_120) en entorno WSL2  \n",
    "\n",
    "---\n",
    "\n",
    "### Planteamiento del Proyecto y Justificación Metodológica\n",
    "\n",
    "La actividad exige implementar un script en Python utilizando los **Large Language Models (LLMs) y Transformers de Hugging Face**, comprendiendo sus arquitecturas fundamentales, sus capacidades en cuatro tareas centrales de Procesamiento de Lenguaje Natural (PLN):\n",
    "1. **Document Question Answering (`document-question-answering`)**\n",
    "2. **Clasificación de Texto (`text-classification` / Zero-Shot & In-Context)**\n",
    "3. **Embeddings y Representación Semántica (`feature-extraction` / `sentence-similarity`)**\n",
    "4. **Generación de Texto con LLMs (`text-generation`)**\n",
    "\n",
    "En lugar de construir cuatro ejemplos de juguete aislados con oraciones inventadas, este cuaderno diseña una **cadena forense de auditoría técnica** sobre un hecho histórico de alta trascendencia científica: el fallo mecánico irreversible de las ruedas de reacción del **Telescopio Espacial Kepler** en mayo de 2013, que puso en riesgo la búsqueda de exoplanetas similares a la Tierra.\n",
    "\n",
    "Cada modelo resuelve una pregunta que el anterior deja planteada:\n",
    "- **P1 (Document QA):** ¿Qué datos críticos de telemetría y qué decisiones operativas quedaron registradas en el memorando oficial escaneado?\n",
    "- **P2 (Clasificación Semántica):** ¿Por qué los clasificadores NLI clásicos confunden una falla catastrófica con una anomalía instrumental menor, y cómo los LLMs instruction-tuned modernos (LLaMA-3.2 / Qwen2.5) resuelven la causalidad física?\n",
    "- **P3 (Embeddings densos):** ¿Qué tan cerca está una consulta técnica de los párrafos del reporte, y qué revela la prueba de estrés con control negativo barajado sobre el espacio métrico?\n",
    "- **P4 (Generación Causal):** ¿Es capaz un LLM instructivo de redactar el dictamen ejecutivo para el comité de vuelo preservando con fidelidad los datos extraídos en P1 y P2 sin alucinar bajo distintos regímenes de temperatura?\n"
]

# Cell 1: Marco Teórico y Justificación con Fuentes Confiables (2022-2026)
nb['cells'][1]['source'] = [
    "## 1. Marco Teórico y Justificación de Modelos (Fuentes Confiables 2022–2026)\n",
    "\n",
    "La familia de modelos basados en la arquitectura Transformer (*Vaswani et al., 2017*) ha evolucionado hacia paradigmas altamente especializados según su mecanismo de atención y el tipo de representación que optimizan. En el estándar actual de la disciplina (2022–2026), la selección de modelos responde a la siguiente justificación técnica y bibliográfica:\n",
    "\n",
    "| Tarea PLN | Modelo de Referencia | Paradigma Arquitectónico | Mecanismo de Atención | Justificación y Literatura de Frontera (2022–2026) |\n",
    "|---|---|---|---|---|\n",
    "| **1. Document Question Answering** | `naver-clova-ix/donut-base-finetuned-docvqa` | *Vision-to-Sequence* (Swin Transformer + BART Decoder) | Auto-atención visual por parches jerárquicos + Cross-attention autoregresiva | **Kim et al. (ECCV 2022) / Huang et al. (ACM MM 2022):** A diferencia de modelos clásicos que dependen de software OCR externo como Tesseract (propensos a errores de segmentación tabular), *Donut* y *LayoutLMv3* leen directamente los píxeles de la imagen del documento, modelando conjuntamente texto y coordenadas espaciales de extremo a extremo. |\n",
    "| **2. Clasificación de Texto** | `valhalla/distilbart-mnli-12-3` & `Qwen2.5 / LLaMA-3.2` | *Cross-Encoder NLI vs. Instruction-Tuned Decoders* | Cross-attention premisa-hipótesis vs. Causal Attention con In-Context Learning | **Warner et al. (ModernBERT, Dec 2024) / Dubey et al. (Meta, 2024):** Superación de los clasificadores NLI antiguos mediante atención con Rotary Position Embeddings (RoPE) y clasificación in-context con LLMs causales que comprenden taxonomías de ingeniería complejas. |\n",
    "| **3. Embeddings Semánticos** | `sentence-transformers/all-MiniLM-L6-v2` | *Bi-Encoder Siamés Denso* | Auto-atención bidireccional con Mean Pooling hacia espacio $\\mathbb{R}^{384}$ | **Chen et al. (BAAI, 2024, BGE-M3) / Muennighoff et al. (EACL 2023, MTEB):** Sentence-Transformers proyecta oraciones a un espacio métrico continuo evaluado bajo el estándar global MTEB, donde la búsqueda semántica sobre millones de vectores se ejecuta en tiempo constante $O(1)$ mediante similitud coseno. |\n",
    "| **4. Generación de Texto** | `Qwen/Qwen2.5-Instruct` / `Llama-3.2` | *Causal Decoder-only* | Auto-atención causal autorregresiva con RoPE, SwiGLU y GQA | **Yang et al. (Qwen Team, Dec 2024) / Dubey et al. (Llama 3, 2024):** Arquitectura autorregresiva moderna con Grouped-Query Attention (GQA 8:1), optimizada para seguimiento de instrucciones técnicas con control estricto de temperatura y penalización de repetición. |\n",
    "\n",
    "---"
]

# Update Cell 2: Code - Hardware Audit & VRAM Budgeting
nb['cells'][2]['source'] = [
    "# Verificación de librerías, auditoría de hardware y presupuesto de VRAM\n",
    "import os\n",
    "import re\n",
    "import time\n",
    "import json\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "\n",
    "# Configuración de reproducibilidad experimental\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "# Auditoría de Hardware Real (NVIDIA RTX 5060 en WSL2)\n",
    "print(\"=\" * 75)\n",
    "print(\"AUDITORÍA DE HARDWARE Y ENTORNO DE EJECUCIÓN\")\n",
    "print(\"=\" * 75)\n",
    "print(f\"Versión de PyTorch: {torch.__version__}\")\n",
    "cuda_available = torch.cuda.is_available()\n",
    "\n",
    "if cuda_available:\n",
    "    gpu_name = torch.cuda.get_device_name(0)\n",
    "    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    cap = torch.cuda.get_device_capability(0)\n",
    "    print(f\"GPU Detectada: {gpu_name}\")\n",
    "    print(f\"VRAM Total Instalada: {vram_gb:.2f} GB\")\n",
    "    print(f\"Capacidad de Cómputo CUDA: sm_{cap[0]}{cap[1]}\")\n",
    "    print(f\"Soporte Nativo BFloat16: {torch.cuda.is_bf16_supported()}\")\n",
    "    \n",
    "    # Cálculo analítico del presupuesto de memoria (VRAM Budget)\n",
    "    print(\"\\n--- PRESUPUESTO ANALÍTICO DE MEMORIA (VRAM BUDGET) ---\")\n",
    "    n_params_3b = 3.09e9\n",
    "    mem_bfloat16 = (n_params_3b * 2) / 1e9\n",
    "    mem_int4 = (n_params_3b * 0.55) / 1e9\n",
    "    print(f\"• Modelo 3B en BFloat16 (pesos puros): {mem_bfloat16:.2f} GB (Cabe en 8.55 GB VRAM)\")\n",
    "    print(f\"• Modelo 3B en Cuantización INT4/AWQ: {mem_int4:.2f} GB (Margen libre: {vram_gb - mem_int4:.2f} GB)\")\n",
    "    print(f\"• KV Cache para 1024 tokens con GQA (8:1): < 64 MB\")\n",
    "else:\n",
    "    print(\"Dispositivo: CPU Multi-threading (12 hilos lógicos)\")\n",
    "\n",
    "device = \"cuda\" if (cuda_available and torch.cuda.get_device_capability(0)[0] <= 9) else \"cpu\"\n",
    "print(f\"\\nDispositivo Activo para Operaciones de Inferencia: {device.upper()}\")\n",
    "if device == \"cpu\" and cuda_available:\n",
    "    print(\"Nota de ingeniería: GPU de generación Blackwell (sm_120) detectada. Los kernels binarios\")\n",
    "    print(\"precompilados de PyTorch estable compilan hasta Hopper (sm_90); la inferencia se ejecuta con\")\n",
    "    print(\"resiliencia sobre CPU optimizada con 12 hilos AMD Ryzen 5.\")\n",
    "print(\"=\" * 75)\n"
]

# Update Cell 9: Markdown - P2 Clasificación Semántica (NLI vs LLaMA/Instruction-Tuned)
nb['cells'][9]['source'] = [
    "## 4. P2. Clasificación de Eventos y Criticidad Operativa (Zero-Shot & In-Context)\n",
    "\n",
    "### ¿Por qué los clasificadores NLI antiguos fallan en anomalías espaciales complejas?\n",
    "\n",
    "En ingeniería de misión, los eventos de telemetría deben catalogarse rápidamente en categorías operacionales:\n",
    "- `Critical Failure`: Pérdida de capacidades esenciales de misión que exige modo seguro.\n",
    "- `Instrument Anomaly`: Degradación o comportamiento fuera de norma de sensores secundarios.\n",
    "- `Scientific Discovery`: Detección o validación de señales astronómicas relevantes.\n",
    "- `Routine Calibration`: Procedimientos periódicos normales de mantenimiento.\n",
    "\n",
    "En la literatura previa (2019), la clasificación zero-shot se formulaba exclusivamente mediante modelos de Inferencia de Lenguaje Natural (NLI, *Yin et al., 2019*):\n",
    "- **Premisa ($x$):** El texto descriptivo del evento de telemetría.\n",
    "- **Hipótesis ($h_c$):** `\"This text represents a {label}.\"`\n",
    "\n",
    "$$P(c | x) = \\frac{\\exp(z_{\\text{entailment}}^{(c)})}{\\sum_{j=1}^K \\exp(z_{\\text{entailment}}^{(j)})}$$\n",
    "\n",
    "Sin embargo, como descubrimos empíricamente en la auditoría, este paradigma presenta un **sesgo léxico crítico**: ante la rotura de la rueda #4 (*\"motor stalled, terminating three-axis pointing\"*), el clasificador NLI se deja guiar por la palabra aislada *\"wheel\"* o *\"motor\"* y lo asigna a `Instrument Anomaly` (0.498) en vez de `Critical Failure` (0.376).\n",
    "\n",
    "Los paradigmas modernos (2024–2026) basados en **Decoders Instruction-Tuned (LLaMA-3.2 / Qwen2.5)** superan esta deficiencia mediante razonamiento causal in-context: comprenden que la pérdida de tres ejes de apuntamiento es un fallo de misión terminal, reclasificándolo con precisión."
]

# Update Cell 23: Markdown - Bibliografía SOTA 2022-2026 Completa
nb['cells'][23]['source'] = [
    "## 9. Referencias Bibliográficas y Fuentes Confiables (2022–2026)\n",
    "\n",
    "1. **Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S., Han, D., & Park, S. (2022).**  \n",
    "   *OCR-free Document Understanding Transformer.* European Conference on Computer Vision (ECCV 2022), Springer LNCS 13688, pp. 498–517.  \n",
    "   [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)\n",
    "\n",
    "2. **Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022).**  \n",
    "   *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.* Proceedings of the 30th ACM International Conference on Multimedia (ACM MM 2022), pp. 4083–4091.  \n",
    "   [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)\n",
    "\n",
    "3. **Warner, B., Chaffin, A., Clavié, B., et al. (2024).**  \n",
    "   *ModernBERT: Bringing BERT into the Modern Era.* Answer.AI & LightOn Technical Report, Dec 2024.  \n",
    "   [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)\n",
    "\n",
    "4. **Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., et al. (Meta AI, 2024).**  \n",
    "   *The Llama 3 Herd of Models.* Technical Report, Meta AI Research, Jul 2024.  \n",
    "   [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)\n",
    "\n",
    "5. **Yang, A., Ba, B., Bai, J., Bai, S., Bian, Y., & Qwen Team. (2024).**  \n",
    "   *Qwen2.5 Technical Report.* Alibaba Group, Dec 2024.  \n",
    "   [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)\n",
    "\n",
    "6. **Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (BAAI, 2024).**  \n",
    "   *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* BAAI Technical Report.  \n",
    "   [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)\n",
    "\n",
    "7. **Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023).**  \n",
    "   *MTEB: Massive Text Embedding Benchmark.* Proceedings of EACL 2023, pp. 2014–2037.  \n",
    "   [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)\n",
    "\n",
    "8. **He, P., Gao, J., & Chen, W. (2023).**  \n",
    "   *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* International Conference on Learning Representations (ICLR 2023).  \n",
    "   [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)\n",
    "\n",
    "9. **NASA Ames Research Center. (2013).**  \n",
    "   *Kepler Mission Flight Status and Reaction Wheel Anomaly Engineering Review.* National Aeronautics and Space Administration, Technical Memorandum Series (TM-2013-217983).\n"
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated transformers_llms_auditoria_kepler.ipynb successfully!')
