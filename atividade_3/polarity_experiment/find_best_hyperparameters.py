from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
import time
import os

corpus_path = "atividade_3/text_databases/review_polarity.csv"

df = pd.read_csv(corpus_path)

# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'],
                                                    test_size=0.2, random_state=14,
                                                    stratify=df['class'])

"""
Parameters detailment:

MultinomialNB:
    Here we only have alpha, the smoothing parameter.
I selected some numbers around the default value, 1.0.

LogisticRegression
    The most relevant parameters here are the solver
algorithm, the penalty, the tolerance for stopping (tol),
and the regularization factor (C).

LGBM
    We're going to analyse the boosting type,
    the number of leaves, and the learning rate
"""


model_parameters = {
    MultinomialNB(): {
        "alpha": [0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 5]
    },
    LogisticRegression(): {
        "solver": ["saga", "lbfgs"],
        "penalty": ["l1", "l2"],
        "tol": [1e-4, 1],
        "C": [1e-4, 1]
    },
    LGBMClassifier(objective="binary"): {
        "boosting_type": ["gbdt", "dart", "rf"],
        "num_leaves": [10, 20, 31, 40, 50],
        "learning_rate": [1e-3, 1e-2, 1e-1]
    }
}

for model, parameters in model_parameters.items():    
    pipe = Pipeline(
        steps=[
            ("vectorizer", CountVectorizer(dtype=np.float64)),
            ("classifier", model)
        ]
    )
    pipe_params = {
        # here we can consider bigrams or trigrams as 
        "vectorizer__ngram_range": [(1,1), (1,2), (1,3)],
    }
    pipe_params.update(
        {f"classifier__{key}": value for key, value in parameters.items()}
    )
    start_time = time.time()
    model_name = type(model).__name__

    # running jobs in parallel with all available processors
    # it's a resource intensive task, beware
    clf = GridSearchCV(pipe, pipe_params, n_jobs=-1)
    clf.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time of {model_name}: {elapsed_time:.2f}s")

    result_df = pd.DataFrame(clf.cv_results_)
    print(result_df)
    result_df.to_csv(f"{os.getcwd()}/outputs/{model_name}_polarity_grid_results.csv", index=False)