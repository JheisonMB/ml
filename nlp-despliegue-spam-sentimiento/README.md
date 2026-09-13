# Actividad 6 — Despliegue de modelos de NLP: spam en SMS y sentimiento en tweets

Jheison Martinez Bolivar · Maestría en Inteligencia Artificial · septiembre 2026

## Contenido

| Ruta | Qué es |
|---|---|
| `despliegue_spam_sentimiento.ipynb` | Cuaderno principal: EDA, preprocesamiento, comparación de modelos, evaluación en test, empaquetado con skops y análisis de resultados. Ejecutado. |
| `prueba_despliegue.ipynb` | Cuaderno de prueba del despliegue: descarga los modelos publicados en Hugging Face y predice mensajes nuevos. Ejecutado. |
| `models/spam_sms/` | Tarjeta de modelo (`README.md`) del clasificador spam/ham. El `model.skops` (23 MB) no se versiona: está en el Hub. |
| `models/tweet_sentiment/` | Tarjeta de modelo del clasificador de sentimiento (4 clases). El `model.skops` (110 MB) no se versiona: está en el Hub. |
| `publish.py` | Script que publicó los dos modelos en el Hub (`python publish.py --user <usuario>`). |
| `requirements.txt` | Dependencias exactas del entorno (Python 3.12). |

## Modelos publicados

- https://huggingface.co/JheisonM/spam-sms-tfidf-logreg
- https://huggingface.co/JheisonM/tweet-sentiment-tfidf-logreg

## Reproducir

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/jupyter nbconvert --to notebook --execute despliegue_spam_sentimiento.ipynb
```

Los datasets se descargan solos con `kagglehub` (SMS Spam Collection y Twitter Entity Sentiment Analysis, de Kaggle).
