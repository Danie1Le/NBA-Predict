from src.preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.predict import predict_outcome
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# 1. Load and clean data
df = load_and_clean_data('Data/NBA_GAMES.csv')

# 2. Feature engineering
df = create_features(df)

# 3. Select features and target
features = [
    'HOME',
    # Team rolling stats
    'PTS_rolling5', 'FG_PCT_rolling5', 'FG3_PCT_rolling5', 'FT_PCT_rolling5',
    'REB_rolling5', 'AST_rolling5', 'TOV_rolling5',
    'PTS_rolling10', 'FG_PCT_rolling10', 'FG3_PCT_rolling10', 'FT_PCT_rolling10',
    'REB_rolling10', 'AST_rolling10', 'TOV_rolling10',
    'WIN_STREAK5',
    # Opponent rolling stats
    'OPP_PTS_rolling5', 'OPP_FG_PCT_rolling5', 'OPP_FG3_PCT_rolling5', 'OPP_FT_PCT_rolling5',
    'OPP_REB_rolling5', 'OPP_AST_rolling5', 'OPP_TOV_rolling5',
    'OPP_PTS_rolling10', 'OPP_FG_PCT_rolling10', 'OPP_FG3_PCT_rolling10', 'OPP_FT_PCT_rolling10',
    'OPP_REB_rolling10', 'OPP_AST_rolling10', 'OPP_TOV_rolling10',
    # Rest days
    'REST_DAYS', 'OPP_REST_DAYS'
]
X = df[features]
y = (df['WL'] == 'W').astype(int)  # 1 for win, 0 for loss

# 4. Train model
model_type = 'xgb'  # Change to 'rf' for RandomForest, 'xgb' for XGBoost, 'logreg' for Logistic Regression
model, X_test, y_test = train_model(X, y, model_type=model_type)

# 5. Predict and evaluate
y_pred = predict_outcome(model, X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
# ROC AUC (for binary classification)
try:
    y_proba = model.predict_proba(X_test)[:, 1]
    print('ROC AUC Score:', roc_auc_score(y_test, y_proba))
except AttributeError:
    print('ROC AUC Score: Not available for this model')

# 6. Feature importances
try:
    importances = model.feature_importances_
    plt.barh(features, importances)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()
except AttributeError:
    print('Feature importances not available for this model.') 