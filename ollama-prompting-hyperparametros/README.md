# Laboratorio de LLM y prompt engineering

Actividad individual de despliegue local de un LLM con Ollama. El notebook se construye de forma incremental, celda por celda, siguiendo el roadmap de aprendizaje: anatomía de una petición, hiperparámetros, alineación de instrucciones, zero/one/few-shot y estructura de resultados.

## Entorno

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m ipykernel install --user --name ollama-prompting-hyperparametros --display-name 'Python (ollama-prompting-hyperparametros)'
```

## Ollama

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
ollama pull qwen3.5:9b
```

Solo ejecuta `ollama serve` si la API no responde. Si aparece `address already in use`, ya existe un proceso escuchando en el puerto 11434.
