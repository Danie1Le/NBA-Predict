import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    
    # Create scaler for models that need feature scaling
    scaler = StandardScaler()
    
    if model_type == 'rf':
        # Fast Random Forest - no grid search (tree-based, doesn't need scaling)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
        model.fit(X_train, y_train)
        print('Trained Random Forest (fast mode)')
        return model, X_test, y_test
    elif model_type == 'xgb' and XGBClassifier is not None:
        # Fast XGBoost - no grid search (tree-based, doesn't need scaling)
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state, eval_metric='logloss')
        model.fit(X_train, y_train)
        print('Trained XGBoost (fast mode)')
        return model, X_test, y_test
    elif model_type == 'logreg':
        # Logistic Regression NEEDS feature scaling for proper performance
        # Clip extreme values before scaling to prevent extreme predictions
        X_train_clipped = np.clip(X_train, -1000, 1000)  # Clip extreme values
        X_test_clipped = np.clip(X_test, -1000, 1000)
        
        X_train_scaled = scaler.fit_transform(X_train_clipped)
        X_test_scaled = scaler.transform(X_test_clipped)
        
        model = LogisticRegression(C=0.1, max_iter=2000, random_state=random_state, solver='liblinear')
        model.fit(X_train_scaled, y_train)
        print('Trained Logistic Regression with feature scaling and clipping (fast mode)')
        
        # Return both model and scaler for prediction
        return (model, scaler), X_test_scaled, y_test
    else:
        raise ValueError('Unknown or unavailable model_type: ' + str(model_type)) 