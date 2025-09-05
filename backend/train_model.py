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
        # Fast Random Forest - no grid search
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
        model.fit(X_train, y_train)
        print('Trained Random Forest (fast mode)')
    elif model_type == 'xgb' and XGBClassifier is not None:
        # Fast XGBoost - no grid search
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state, eval_metric='logloss')
        model.fit(X_train, y_train)
        print('Trained XGBoost (fast mode)')
    elif model_type == 'logreg':
        # Fast Logistic Regression - no grid search, increased max_iter to prevent convergence warnings
        model = LogisticRegression(C=1, max_iter=2000, random_state=random_state, solver='liblinear')
        model.fit(X_train, y_train)
        print('Trained Logistic Regression (fast mode)')
    else:
        raise ValueError('Unknown or unavailable model_type: ' + str(model_type))
    return model, X_test, y_test 