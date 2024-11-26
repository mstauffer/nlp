from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import pandas as pd
import time

corpus_path = "atividade_3/text_databases/Dmoz-Sports.csv"

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

SVC
    Kernel and regularization factor are the main parameters.
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
    SVC(): {
        "kernel": ["linear", "rbf", "poly"],
        "C": [1e-1, 1, 5, 10]
    }
}

tfidf = TfidfVectorizer(ngram_range=(1,3))
X_train_csr = tfidf.fit_transform(X_train)

for model, parameters in model_parameters.items():    
    start_time = time.time()
    model_name = type(model).__name__

    # running jobs in parallel with all available processors
    # it's a resource intensive task, beware
    clf = GridSearchCV(model, parameters, n_jobs=-1)
    clf.fit(X_train_csr, y_train)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time of {model_name}: {elapsed_time:.2f}s")

    result_df = pd.DataFrame(clf.cv_results_)
    print(result_df)
    result_df.to_csv(f"outputs/{model_name}_grid_results.csv", index=False)