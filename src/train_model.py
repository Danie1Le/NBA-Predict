from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def train_model(X, y, test_size=0.2, random_state=42, model_type='logreg'):
    """
    Train a classifier (RandomForest, XGBoost, or Logistic Regression) with hyperparameter tuning to predict NBA game outcomes.
    Splits data into train/test and returns the best model and test data.
    model_type: 'rf' (RandomForest), 'xgb' (XGBoost), 'logreg' (Logistic Regression)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        print('Best params (RF):', grid.best_params_)
        model = grid.best_estimator_
    elif model_type == 'xgb' and XGBClassifier is not None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(XGBClassifier(random_state=random_state, eval_metric='logloss'), param_grid, cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        print('Best params (XGB):', grid.best_params_)
        model = grid.best_estimator_
    elif model_type == 'logreg':
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
        grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=random_state), param_grid, cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        print('Best params (LogReg):', grid.best_params_)
        model = grid.best_estimator_
    else:
        raise ValueError('Unknown or unavailable model_type: ' + str(model_type))
    return model, X_test, y_test 