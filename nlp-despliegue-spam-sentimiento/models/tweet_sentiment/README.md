---
language: en
library_name: sklearn
license: mit
tags:
- text-classification
- sklearn
- skops
- tfidf
---


# Model description

Clasificador de sentimiento en 4 clases (Positive, Negative, Neutral, Irrelevant) para tweets en inglés: TF-IDF de unigramas y bigramas de palabras y regresión logística con `class_weight="balanced"`. Devuelve la etiqueta y la probabilidad de cada clase.

## Intended uses & limitations

Las métricas son sobre tweets cuyo id no aparece en el entrenamiento; con un split aleatorio el F1 sube a 0.955 porque el dataset trae seis paráfrasis de cada tweet. Irrelevant se detecta mal (F1 0.42): requiere saber de qué habla el tweet, no solo qué palabras trae. Dominio: 32 entidades, sobre todo videojuegos, en 2020. No usa la entidad como entrada.

## Training Procedure

69354 filas del train original tras quitar 686 sin texto, textos con etiqueta contradictoria y duplicados exactos. Split 80/20 por id de tweet (GroupShuffleSplit), selección con GroupKFold de 5 pliegues comparando MultinomialNB, LogisticRegression y LinearSVC sobre unigramas, bigramas y caracteres. El modelo publicado se reentrenó con el 100 %.

Datos: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

### Hyperparameters

<details>
<summary> Click to expand </summary>

|     Hyperparameter     |                                                                       Value                                                                        |
| :--------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
|         memory         |                                                                        None                                                                        |
|         steps          | [('tfidf', TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)), ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))] |
|    transform_input     |                                                                        None                                                                        |
|        verbose         |                                                                       False                                                                        |
|         tfidf          |                                          TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)                                          |
|          clf           |                                             LogisticRegression(class_weight='balanced', max_iter=3000)                                             |
|    tfidf__analyzer     |                                                                        word                                                                        |
|     tfidf__binary      |                                                                       False                                                                        |
|  tfidf__decode_error   |                                                                       strict                                                                       |
|      tfidf__dtype      |                                                              <class 'numpy.float64'>                                                               |
|    tfidf__encoding     |                                                                       utf-8                                                                        |
|      tfidf__input      |                                                                      content                                                                       |
|    tfidf__lowercase    |                                                                        True                                                                        |
|     tfidf__max_df      |                                                                        1.0                                                                         |
|  tfidf__max_features   |                                                                        None                                                                        |
|     tfidf__min_df      |                                                                         2                                                                          |
|   tfidf__ngram_range   |                                                                       (1, 2)                                                                       |
|      tfidf__norm       |                                                                         l2                                                                         |
|  tfidf__preprocessor   |                                                                        None                                                                        |
|   tfidf__smooth_idf    |                                                                        True                                                                        |
|   tfidf__stop_words    |                                                                        None                                                                        |
|  tfidf__strip_accents  |                                                                        None                                                                        |
|  tfidf__sublinear_tf   |                                                                        True                                                                        |
|  tfidf__token_pattern  |                                                                   (?u)\b\w\w+\b                                                                    |
|    tfidf__tokenizer    |                                                                        None                                                                        |
|     tfidf__use_idf     |                                                                        True                                                                        |
|   tfidf__vocabulary    |                                                                        None                                                                        |
|         clf__C         |                                                                        1.0                                                                         |
|   clf__class_weight    |                                                                      balanced                                                                      |
|       clf__dual        |                                                                       False                                                                        |
|   clf__fit_intercept   |                                                                        True                                                                        |
| clf__intercept_scaling |                                                                         1                                                                          |
|     clf__l1_ratio      |                                                                        0.0                                                                         |
|     clf__max_iter      |                                                                        3000                                                                        |
|      clf__n_jobs       |                                                                        None                                                                        |
|      clf__penalty      |                                                                     deprecated                                                                     |
|   clf__random_state    |                                                                        None                                                                        |
|      clf__solver       |                                                                       lbfgs                                                                        |
|        clf__tol        |                                                                       0.0001                                                                       |
|      clf__verbose      |                                                                         0                                                                          |
|    clf__warm_start     |                                                                       False                                                                        |

</details>

## Evaluation Results

|              Metric             | Value |
| :-----------------------------: | :---: |
| f1_macro (test por id de tweet) | 0.557 |
| accuracy (test por id de tweet) | 0.578 |
|           f1 Negative           | 0.663 |
|            f1 Neutral           | 0.526 |
|           f1 Positive           |  0.62 |
|          f1 Irrelevant          |  0.42 |

# How to Get Started with the Model

```python
import skops.io as sio
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="<usuario>/tweet-sentiment-tfidf-logreg", filename="model.skops")
model = sio.load(path)
model.predict(["texto de ejemplo"])
```

# Model Card Authors

Jheison Martinez Bolivar

# Model Card Contact

jhmartinez07@unisalle.edu.co
