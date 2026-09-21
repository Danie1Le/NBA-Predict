"""
Model definitions and the train/test split.

Every model is a pipeline, so imputation and scaling are fitted on the training
fold only and travel with the model. evaluate.py uses the same build_model(),
which is what makes its reported numbers numbers for the shipped models.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

MODEL_TYPES = ['logreg', 'rf', 'xgb']


def build_model(model_type='logreg', random_state=42):
    """An unfitted pipeline for one model type."""
    if model_type == 'logreg':
        return make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            # Strong regularisation: ~1000 training games and 11 correlated features.
            LogisticRegression(C=0.3, max_iter=5000, random_state=random_state),
        )
    if model_type == 'rf':
        return make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=20,
                                   random_state=random_state, n_jobs=-1),
        )
    if model_type == 'xgb':
        if XGBClassifier is None:
            raise ValueError('xgboost is not installed')
        return XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             eval_metric='logloss', random_state=random_state)
    raise ValueError('Unknown model_type: ' + str(model_type))


def date_ordered_split(X, y, test_size=0.2):
    """
    Split on time, not at random.

    Rows arrive sorted by game date, so the tail is the future. A random split
    would let the model train on games played after the ones it is tested on.
    """
    cut = int(len(X) * (1 - test_size))
    return X[:cut], X[cut:], y[:cut], y[cut:]


def train_model(X, y, test_size=0.2, random_state=42, model_type='logreg'):
    """Fit one model on the earlier games and return it with the held-out tail."""
    X_train, X_test, y_train, y_test = date_ordered_split(X, y, test_size)
    model = build_model(model_type, random_state)
    model.fit(X_train, y_train)
    print(f'Trained {model_type} on {len(X_train)} games, holding out {len(X_test)}')
    return model, X_test, y_test
