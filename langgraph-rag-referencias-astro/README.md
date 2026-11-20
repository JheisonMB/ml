# Pipeline RAG con LangGraph sobre la literatura de astro-denoise

Notebook `langgraph_rag_agente_referencias.ipynb`: grafo de LangGraph que reescribe la pregunta (traducción y variantes técnicas), recupera por similitud coseno, cuenta con una herramienta y genera con citas; memoria por `thread_id`. Se compara con un LLM solo y con un RAG fijo.

## Entorno

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m ipykernel install --user --name langgraph-rag-referencias-astro
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

## Índice

El corpus son los PDFs de `astro-denoise/referencias/asreview/pdfs`, que viven fuera de este repositorio. La ruta se cambia con la variable de entorno `ASTRO_DENOISE_PDFS`. El índice Chroma (`index/`) no se versiona; se construye con:

```bash
./venv/bin/python rag_pipeline.py
```

Es reanudable: el manifiesto `index/manifest.json` (sí versionado) registra los archivos procesados y la causa de cada fallo. La colección usa métrica coseno.

`evaluacion_cache.json` guarda las respuestas de la comparación; borrarlo regenera las respuestas con el modelo.
