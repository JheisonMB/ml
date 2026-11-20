# Agente RAG en produccion: Colab + Drive + Gradio + Meta API

El mismo sistema agentico de `langgraph-rag-referencias-astro` (reescribir -> recuperar/contar -> generar con citas), llevado a produccion de la forma mas simple: un notebook en Colab con interfaz Gradio live y streaming.

## Que cambia respecto a la base

| Aspecto | Base | Produccion |
|---|---|---|
| PDFs | checkout local fuera del repo | carpeta compartida de Drive (se ubica o se descarga sola) |
| Embeddings | Ollama `nomic-embed-text` | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (sin servidor en Colab) |
| LLM | Ollama `qwen3.5:4b` | Meta `muse-spark-1.3-contributor` por API (`https://api.meta.ai/v1`, OpenAI-compatible) |
| UI | notebook | Gradio `ChatInterface` con streaming + enlace publico (`share=True`) y API |

## Archivos

Dos cuadernos autocontenidos (todo el código va en las celdas):

- `langgraph_rag_build_index.ipynb`: cuaderno 1, arma y publica el índice (una sola vez).
- `langgraph_rag_gradio_colab.ipynb`: cuaderno 2, trae el índice listo, pregunta con el grafo y expone Gradio.
- `requirements.txt`: sin pines de numpy/torch para no romper el torch preinstalado de Colab.

## Controles de hiperparametros (panel lateral)

- Temperatura (0-1.5), esfuerzo de busqueda en variantes (1-5), fragmentos por consulta k (1-8), umbral de similitud (0-0.9) y maximo de tokens (128-2048).
- La reescritura (planner) queda fija en temperatura 0 para que el enrutamiento no varie.

## Uso en Colab

1. Deja `pdfs/` en la carpeta compartida (el `index/` lo crea el cuaderno 1).
2. Consigue la clave en la [Meta Model API](https://dev.meta.ai) y guardala como secreto `META_API_KEY` en Colab.
3. Abre el notebook en Colab (tras el push):
   `https://colab.research.google.com/github/JheisonMB/ml/blob/main/langgraph-rag-gradio-colab/langgraph_rag_gradio_colab.ipynb`
4. Ejecuta de arriba a abajo y abre el enlace `*.gradio.live` de la ultima celda.

## Cómo corre en Colab (cualquiera)

El cuaderno es autocontenido: trae todo el código en sus celdas y solo necesita los PDF.

1. La carpeta compartida solo debe tener `pdfs/` (el `index/` lo crea y publica el propio cuaderno).
2. Monta Drive y ubica `rag_corpus`, o descarga la copia pública con `gdown --folder`.
3. El índice se construye en disco local y se publica en Drive al terminar; lo ya indexado se salta.
4. Solo se necesita el secreto `META_API_KEY`.

Todo el código vive en las celdas de los cuadernos.

## Controles de hiperparametros (panel lateral)

- Temperatura (0-1.5), esfuerzo de busqueda en variantes (1-5), fragmentos por consulta k (1-8), umbral de similitud (0-0.9) y maximo de tokens (128-2048).
- La reescritura (planner) queda fija en temperatura 0 para que el enrutamiento no varie.

## Uso en Colab

1. Deja `pdfs/` en la carpeta compartida (el `index/` lo crea el cuaderno 1).
2. Consigue la clave en la [Meta Model API](https://dev.meta.ai) y guardala como secreto `META_API_KEY` en Colab.
3. Abre el notebook en Colab (tras el push):
   `https://colab.research.google.com/github/JheisonMB/ml/blob/main/langgraph-rag-gradio-colab/langgraph_rag_gradio_colab.ipynb`
4. Ejecuta de arriba a abajo y abre el enlace `*.gradio.live` de la ultima celda.

## Cómo corre en Colab (cualquiera)

1. El cuaderno monta Drive y ubica `rag_corpus` (unidad propia o acceso directo agregado); si no la encuentra, descarga la copia pública de solo lectura con `gdown --folder` a `/content/rag_corpus`.
2. El índice se construye en disco local (`/content/rag_index`), porque SQLite no funciona sobre el montaje de Drive. Es reanudable: lo ya indexado se salta.
3. Al terminar, el índice se publica en la carpeta compartida (`index/`), así la siguiente ejecución lo reutiliza sin reindexar.
4. Solo se necesita el secreto `META_API_KEY`.

## Variables de entorno

- `META_API_KEY` (obligatoria; en Colab va en Secrets con ese nombre, el notebook la lee con `userdata` y solo si no existe la pide con `getpass`; nunca queda escrita en el codigo), `META_MODEL` (defecto `muse-spark-1.3-contributor`), `META_BASE_URL` (defecto `https://api.meta.ai/v1`).
- `RAG_PDFS` y `RAG_INDEX` (los fija el cuaderno), `RAG_EMBED_MODEL`, `RAG_SIM_MIN` (defecto `0.35`).

## MCP de Colab (ya configurado en esta maquina)

En `~/.config/muse/settings.json`, servidor `colab-mcp`:

```json
"colab-mcp": {
  "transport": "stdio",
  "command": "uvx",
  "args": ["--from", "git+https://github.com/googlecolab/colab-mcp", "colab-mcp"]
}
```

Nota: el README oficial muestra `args: ["git+https://..."]`, pero `uvx` lo rechaza (`Not a valid package name`); la forma que verifica es `--from <url> colab-mcp`. Requiere reiniciar la sesion para que el servidor aparezca.
