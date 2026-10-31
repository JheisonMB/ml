import json
from pathlib import Path

nb_path = Path('nlp-llms-cuatro-tareas/llms_transformers_cuatro_tareas.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 8: Análisis P1
nb['cells'][8]['source'] = [
    "### Análisis y Hallazgos de P1 (Extracción Visual de Documento)\n",
    "\n",
    "1. **Lectura tabular sin OCR tradicional:** Donut resolvió con éxito los campos estructurales del memorando: identificó el código oficial (`nasa-tm-2013-217983`), el investigador principal (`william j. borucki`) y la celda exacta de la tabla para la rueda #4 (`critical friction failure`), leyendo directamente los píxeles de la imagen.\n",
    "2. **Ambigüedad de columna en tablas:** Para la consulta *What is the measured RPM of Reaction Wheel #1?*, el modelo respondió `500 - 1200 rpm`, que corresponde a la columna contigua (*Nominal Limit*) en lugar del valor medido (*854.2 RPM*). Esto ilustra una limitación clásica de la atención espacial en transformers visuales: cuando dos columnas comparten proximidad geométrica y prefijos similares, la atención por parches puede colapsar hacia el rango más prominente.\n",
    "3. **El fallo de frontera de conocimiento (Pregunta control):** Cuando se le preguntó por la temperatura de Marte (*What is the surface temperature of Mars?*, dato inexistente en el informe), el decodificador autorregresivo emitió `maximum` en lugar de declarar ignorancia. Los modelos DocVQA carecen de una opción de abstención calibrada; su entrenamiento los fuerza a emitir el token de respuesta más probable condicionado a la imagen, generando un falso positivo por completitud.\n"
]

# Cell 12: Análisis P2
nb['cells'][12]['source'] = [
    "### Análisis y Hallazgos de P2 (Clasificación Zero-Shot con NLI)\n",
    "\n",
    "1. **Certeza categórica en eventos rutinarios:** La calibración del fotómetro (`Periodic focal plane photometer thermal bias offset...`) fue clasificada como `Routine Calibration` con una probabilidad de **0.8991** y un margen abrumador (**gap de 0.8480**) sobre el segundo candidato. La implicación lógica de NLI identifica inequívocamente términos como *offset*, *calibration* y *periodic*.\n",
    "2. **La frontera entre falla crítica y anomalía:** La falla de la rueda 4 clasificó como `Instrument Anomaly` (**0.4981**) con `Critical Failure` como runner-up (**0.3764**, margen de 0.1217). Para un modelo generalista sin ontología aeroespacial explícita, una rueda de reacción es primariamente un *instrumento*, por lo que agrupa el evento como anomalía instrumental antes que como fallo catastrófico de sistema general.\n",
    "3. **Sensibilidad de la formulación de hipótesis:** Al usar el template `This telemetry log indicates a {}.`, el clasificador distribuye la probabilidad entre las etiquetas candidatas. Para tareas aeroespaciales en producción, definir hipótesis contextualizadas (ej. *This log indicates a mission-terminating failure*) permite ensanchar el margen de discriminación.\n"
]

# Cell 16: Análisis P3
nb['cells'][16]['source'] = [
    "### Análisis y Hallazgos de P3 (Embeddings y Control Negativo Barajado)\n",
    "\n",
    "1. **Sensibilidad temática de la búsqueda:** La consulta sobre la orientación de paneles solares (`Q3`) encontró su máxima afinidad con el distractor léxico (**0.6452**), mientras que la consulta de exoplanetas (`Q2`) emparejó con el párrafo de descubrimientos planetarios (**0.4542**), mostrando que el espacio vectorial organiza adecuadamente los clústeres temáticos.\n",
    "2. **El resultado revelador de la línea base barajada:** La consulta sobre la pérdida de control (`Q1`) arrojó una similitud de **0.6095** con el párrafo objetivo y un sorprendente **0.6262** con el *Control Barajado* (las mismas palabras en orden aleatorio). Esto confirma un principio fundamental de la representación densa con Bi-Encoders: al promediar los vectores de los tokens (*mean pooling*), la similitud coseno mide fundamentalmente la densidad del vocabulario compartido en el espacio latente. La sintaxis gramatical aporta un ajuste fino, pero no domina la magnitud del vector medio.\n",
    "3. **Implicación para arquitecturas RAG:** En producción, confiar exclusivamente en similitud coseno de embeddings densos es insuficiente para filtrar textos incoherentes o alucinados que compartan léxico. Se requiere un esquema de re-ranking con Cross-Encoder o un enfoque híbrido con BM25.\n"
]

# Cell 20: Análisis P4
nb['cells'][20]['source'] = [
    "### Análisis y Hallazgos de P4 (Fidelidad Factual y Alucinaciones con LLMs)\n",
    "\n",
    "1. **Fidelidad y consistencia en Greedy (T = 0.0):** La decodificación determinista produjo una síntesis rigurosa: describió el fallo mecánico de la rueda #4, el cese de rotación, la imposibilidad de apuntamiento científico y la activación de modos de contingencia, sin añadir entidades ajenas.\n",
    "2. **El colapso de factualidad en alta temperatura (T = 1.4):** A temperatura 1.4, la distribución de salida se volvió caótica: el modelo inventó fechas específicas (*\"March 5th\", \"April 6th\"*), ubicaciones no mencionadas (*\"Kennedy Space Center\"*) y hasta un satélite inexistente en el contexto (*\"Space Station Unity satellite\"*). La puntuación de fidelidad automatizada cayó de 3 a 1.\n",
    "3. **Regla de ingeniería para LLMs en misión crítica:** En reportes técnicos, legales o biomédicos, nunca debe usarse muestreo con temperatura libre ($T > 0.3$). La decodificación Greedy ($T = 0.0$) o el muestreo con penalización de repetición y temperatura baja ($T \\le 0.2$) es obligatoria para garantizar la integridad de los datos.\n"
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cleanly updated analysis cells!')
