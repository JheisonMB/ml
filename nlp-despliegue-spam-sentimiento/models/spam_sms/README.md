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

Clasificador binario spam/ham para SMS en inglés: TF-IDF de n-gramas de caracteres (2 a 5, dentro de palabra) y regresión logística con `class_weight="balanced"`. Devuelve la etiqueta y la probabilidad de spam.

## Intended uses & limitations

Corpus de SMS en inglés de la década de 2000. Los errores residuales son spam escrito como conversación personal, donde el tono pesa más que la URL o el número que a veces traen; el phishing moderno con tono oficial queda cerca del 50 %. Umbral por defecto 0.5; la probabilidad se devuelve para que quien lo use juzgue los casos cercanos.

## Training Procedure

5158 mensajes únicos (4516 ham, 642 spam) tras reconstruir 50 filas partidas por comas y quitar duplicados exactos. Configuración elegida por validación cruzada estratificada de 5 pliegues sobre el 80 % de train, comparando MultinomialNB, LogisticRegression y LinearSVC sobre n-gramas de palabras y de caracteres. El modelo publicado se reentrenó con el 100 %.

Datos: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Hyperparameters

<details>
<summary> Click to expand </summary>

|     Hyperparameter     |                                                                                            Value                                                                                            |
| :--------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         memory         |                                                                                             None                                                                                            |
|         steps          | [('tfidf', TfidfVectorizer(analyzer='char_wb', min_df=2, ngram_range=(2, 5),<br />                sublinear_tf=True)), ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))] |
|    transform_input     |                                                                                             None                                                                                            |
|        verbose         |                                                                                            False                                                                                            |
|         tfidf          |                                          TfidfVectorizer(analyzer='char_wb', min_df=2, ngram_range=(2, 5),<br />                sublinear_tf=True)                                          |
|          clf           |                                                                  LogisticRegression(class_weight='balanced', max_iter=2000)                                                                 |
|    tfidf__analyzer     |                                                                                           char_wb                                                                                           |
|     tfidf__binary      |                                                                                            False                                                                                            |
|  tfidf__decode_error   |                                                                                            strict                                                                                           |
|      tfidf__dtype      |                                                                                   <class 'numpy.float64'>                                                                                   |
|    tfidf__encoding     |                                                                                            utf-8                                                                                            |
|      tfidf__input      |                                                                                           content                                                                                           |
|    tfidf__lowercase    |                                                                                             True                                                                                            |
|     tfidf__max_df      |                                                                                             1.0                                                                                             |
|  tfidf__max_features   |                                                                                             None                                                                                            |
|     tfidf__min_df      |                                                                                              2                                                                                              |
|   tfidf__ngram_range   |                                                                                            (2, 5)                                                                                           |
|      tfidf__norm       |                                                                                              l2                                                                                             |
|  tfidf__preprocessor   |                                                                                             None                                                                                            |
|   tfidf__smooth_idf    |                                                                                             True                                                                                            |
|   tfidf__stop_words    |                                                                                             None                                                                                            |
|  tfidf__strip_accents  |                                                                                             None                                                                                            |
|  tfidf__sublinear_tf   |                                                                                             True                                                                                            |
|  tfidf__token_pattern  |                                                                                        (?u)\b\w\w+\b                                                                                        |
|    tfidf__tokenizer    |                                                                                             None                                                                                            |
|     tfidf__use_idf     |                                                                                             True                                                                                            |
|   tfidf__vocabulary    |                                                                                             None                                                                                            |
|         clf__C         |                                                                                             1.0                                                                                             |
|   clf__class_weight    |                                                                                           balanced                                                                                          |
|       clf__dual        |                                                                                            False                                                                                            |
|   clf__fit_intercept   |                                                                                             True                                                                                            |
| clf__intercept_scaling |                                                                                              1                                                                                              |
|     clf__l1_ratio      |                                                                                             0.0                                                                                             |
|     clf__max_iter      |                                                                                             2000                                                                                            |
|      clf__n_jobs       |                                                                                             None                                                                                            |
|      clf__penalty      |                                                                                          deprecated                                                                                         |
|   clf__random_state    |                                                                                             None                                                                                            |
|      clf__solver       |                                                                                            lbfgs                                                                                            |
|        clf__tol        |                                                                                            0.0001                                                                                           |
|      clf__verbose      |                                                                                              0                                                                                              |
|    clf__warm_start     |                                                                                            False                                                                                            |

</details>

## Evaluation Results

|                   Metric                  | Value |
| :---------------------------------------: | :---: |
|    accuracy (test, 20 % estratificado)    | 0.993 |
| precision_spam (test, 20 % estratificado) | 0.984 |
|   recall_spam (test, 20 % estratificado)  | 0.961 |
|     f1_spam (test, 20 % estratificado)    | 0.972 |

# How to Get Started with the Model

```python
import skops.io as sio
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="<usuario>/spam-sms-tfidf-logreg", filename="model.skops")
model = sio.load(path)
model.predict(["texto de ejemplo"])
```

# Model Card Authors

Jheison Martinez Bolivar

# Model Card Contact

jhmartinez07@unisalle.edu.co
