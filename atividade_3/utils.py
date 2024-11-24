from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import time

def custom_cls(
    results: pd.DataFrame,
    model_label: str,
    split: List[pd.Series],
    model: Optional[BaseEstimator]=None,
    vectorizer: Optional[_VectorizerMixin]=None,
    pipe: Optional[Pipeline]=None
):

    X_train, X_test, y_train, y_test = split[0], split[1], split[2], split[3]
                                                        
    start_time = time.time()
    if pipe is not None:
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
    else:
        X_csr_train = vectorizer.fit_transform(X_train)
        X_csr_test = vectorizer.transform(X_test)

        model.fit(X_csr_train, y_train)
        predictions = model.predict(X_csr_test)
    
    row = pd.DataFrame(columns=['Precision', 'Recall', 'F1-Score', 'Time elapsed'])
    row.loc[model_label] = precision_recall_fscore_support(
        y_test,
        predictions,
        average = 'weighted',
        zero_division = 0
    )
    row.at[model_label, 'Time elapsed'] = time.time()-start_time
    df_concat = pd.concat([results, row])

    return df_concat, predictions

def get_cls_results(
    results: pd.DataFrame,
    model_label: str,
    split: List[pd.Series],
    show_results: bool=True,
    model: Optional[BaseEstimator]=None,
    vectorizer: Optional[_VectorizerMixin]=None,
    pipe: Optional[Pipeline]=None
):
    
    df_concat, preds = custom_cls(results, model_label, split, model, vectorizer, pipe)
    y_test = split[3]
    
    if show_results:
        print(classification_report(y_test, preds))

        ConfusionMatrixDisplay.from_predictions(
            y_true=y_test,
            y_pred=preds,
            xticks_rotation=90
        )

        plt.show()
    
    return df_concat.sort_values(by=['F1-Score'], ascending=False)
