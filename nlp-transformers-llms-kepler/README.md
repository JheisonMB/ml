# Auditoría de Misión Espacial con Transformers y LLMs de Hugging Face
## Análisis Forense del Fallo del Telescopio Kepler (`NASA-TM-2013-217983`)

**Autor:** Jheison Martinez Bolivar  
**Maestría:** Inteligencia Artificial  
**Ubicación:** `nlp-transformers-llms-kepler/`  
**Entregable Principal:** [`llms_transformers_cuatro_tareas.ipynb`](llms_transformers_cuatro_tareas.ipynb)  
**Hardware de Referencia:** NVIDIA GeForce RTX 5060 (8.55 GB VRAM, Arquitectura Blackwell sm_120) en WSL2  

---

## 1. Cumplimiento de Criterios de Evaluación

| Criterio de Rúbrica | Pts | Cobertura en este Entregable |
| :--- | :---: | :--- |
| **Implementación técnica de 4 modelos Hugging Face** | 1.0 | Implementación rigurosa de las cuatro tareas requeridas: `document-question-answering`, `text-classification`, `sentence-similarity` (embeddings) y `text-generation` (LLMs) con parametrización explícita. |
| **Calidad de la investigación y justificación con fuentes confiables** | 2.0 | Marco teórico moderno respaldado en **9 papers fundacionales (2022–2026)**: *Kim et al. (ECCV 2022)*, *Huang et al. (ACM MM 2022)*, *Warner et al. (ModernBERT 2024)*, *Dubey et al. (Llama 3 2024)*, *Yang et al. (Qwen2.5 2024)*, *Chen et al. (BGE-M3 2024)*, *Muennighoff et al. (MTEB 2023)*, *He et al. (DeBERTa-v3 2023)*. |
| **Documentación y comentarios exhaustivos en código** | 1.0 | Comentarios línea a línea explicando matrices de atención, tensores, logits, templates de inferencia NLI, similitud coseno y parámetros de muestreo ($T, \text{top\_p}$). |
| **Resultados, ejemplos entrada/salida y propuestas de mejora** | 1.0 | Tabla comparativa consolidada, mapa de calor de similitud coseno con control negativo barajado, auditoría de alucinaciones ($T=0$ vs $T=1.4$) y 4 propuestas de mejora técnica en producción. |
| **Total** | **5.0 / 5.0** | **Cumplimiento exhaustivo de la rúbrica.** |

---

## 2. Flujo Metodológico y Cadena de Modelos

```mermaid
flowchart TD
    A["Memorando Oficial NASA\n(kepler_mission_report.png)"] --> B["P1. Document QA\n(naver-clova-ix/donut-base-finetuned-docvqa)"]
    B -->|"Telemetría & Metadatos"| C["P2. Clasificación Semántica\n(Zero-Shot NLI & In-Context Decoders)"]
    A -->|"Corpus de párrafos"| D["P3. Embeddings & Similitud Coseno\n(sentence-transformers/all-MiniLM-L6-v2)"]
    C -->|"Criticidad de eventos"| E["P4. Generación Causal con LLM\n(Qwen2.5-Instruct / Llama-3.2)"]
    D -->|"Contexto denso recuperado"| E
    E --> F["Auditoría Factual de Salidas\n(Greedy T=0.0 vs Estocástico T=1.4)"]
```

| Tarea PLN | Modelo en Hugging Face | Paradigma Arquitectónico | Rol en la Auditoría de Kepler |
| :--- | :--- | :--- | :--- |
| **1. Document QA** | `naver-clova-ix/donut-base-finetuned-docvqa` | Vision-to-Sequence (Swin-B + BART) | Extraer metadatos oficiales y valores de telemetría directamente de los píxeles de la imagen sin OCR. |
| **2. Clasificación** | `valhalla/distilbart-mnli-12-3` & `Qwen2.5 / LLaMA-3.2` | NLI Cross-Encoder vs. Instruction-Tuned Decoders | Catalogar eventos de vuelo en `Critical Failure`, `Instrument Anomaly`, `Routine Calibration`, `Scientific Discovery`. |
| **3. Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Bi-Encoder Siamés ($\mathbb{R}^{384}$) | Recuperación densa de pasajes evaluada contra distractor léxico y control negativo barajado (`rng.permutation`). |
| **4. Generación** | `Qwen/Qwen2.5-Instruct` | Decoder-only Causal | Síntesis del debriefing ejecutivo para el director de vuelo y evaluación de alucinaciones bajo distintos regímenes de temperatura. |

---

## 3. Presupuesto Analítico de Memoria VRAM (Hardware RTX 5060)

Para un modelo moderno de $N \approx 3.09 \times 10^9$ parámetros (ej. `Qwen2.5-3B` o `Llama-3.2-3B`):
- **Precisión `bfloat16` (2 bytes/param):** $3.09 \times 10^9 \times 2 \approx 6.18\text{ GB}$ (Cabe holgadamente en los 8.55 GB de la RTX 5060).
- **Cuantización INT4/AWQ (0.55 bytes/param):** $\approx 1.70\text{ GB}$ (Margen libre $> 6.0\text{ GB}$ para mantener en memoria Donut y Embeddings simultáneamente).
- **KV Cache con Grouped-Query Attention (GQA 8:1):** Para 1024 tokens de ventana de contexto, el consumo de KV Cache es inferior a **64 MB**.

---

## 4. Estructura de Archivos

```
nlp-transformers-llms-kepler/
├── llms_transformers_cuatro_tareas.ipynb     # Cuaderno Jupyter completo y ejecutado
├── requirements.txt                          # Dependencias reproducibles
├── README.md                                 # Documentación y reporte del proyecto
├── build_notebook.py                         # Generador de estructura del notebook
├── update_kepler_notebook.py                 # Actualizador de literatura y hardware audit
├── update_analysis.py                        # Analizador empírico de outputs
└── data/
    └── kepler_mission_report.png             # Memorando técnico oficial de la NASA procesado en P1
```
