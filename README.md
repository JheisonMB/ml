# Machine Learning — entregas de la Maestría en Inteligencia Artificial

Jheison Martinez Bolivar. Un directorio por actividad, cada uno con su notebook ejecutado, su `requirements.txt` y, cuando la actividad lo pide, el informe IEEE en LaTeX (`main.tex`, compilado con [texforge](https://github.com/univerlab/texforge)) y su PDF.

## Convención de nombres

`<tecnica>-<dataset-o-tema>`, en kebab-case y español sin tildes. La librería aparece solo cuando es el tema de la actividad (`spacy`, `transformers`).

## Proyectos

| Directorio | Actividad | Entrega |
|---|---|---|
| `regresion-excentricidad-exoplanetas` | Regresión de la excentricidad orbital (NASA Exoplanet Archive) | notebook + informe IEEE |
| `nlp-preprocesamiento-hotel-reviews` | Preprocesamiento de texto sobre reviews de hoteles | notebook |
| `clasificacion-exoplanetas-kepler` | Clasificación de candidatos a exoplaneta (Kepler KOI) | notebook + informe IEEE |
| `nlp-spacy-peliculas-tmdb` | Análisis lingüístico con spaCy sobre sinopsis de TMDB | notebook |
| `nlp-ngramas-amazon-reviews` | Vectorización y n-gramas sobre reviews de Amazon | notebook |
| `clustering-estelar-sdss` | Aprendizaje no supervisado (PCA, K-means, GMM, información mutua) sobre SDSS DR17 | notebook + informe IEEE |
| `didactico-covarianza-mahalanobis` | Cuaderno didáctico de apoyo: gaussianas, covarianza y Mahalanobis | notebook |
| `nlp-transformers-capitulo-novela` | Cuatro transformers (resumen, NER, QA, similitud) sobre un capítulo de novela | notebook |
| `nlp-despliegue-spam-sentimiento` | Despliegue de modelos clásicos: spam en SMS y sentimiento en tweets, publicados en Hugging Face | dos notebooks + modelos en el Hub |

## Reproducir un proyecto

```bash
cd <directorio>
python -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Los datasets se descargan con `kagglehub` desde el propio notebook. No se versionan entornos virtuales, modelos serializados (`*.skops`, publicados en Hugging Face) ni textos con copyright.
