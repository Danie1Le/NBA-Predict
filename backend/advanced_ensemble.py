import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier  # Optional - will use alternative if not available
import warnings
warnings.filterwarnings('ignore')

class AdvancedNBAEnsemble:
    """
    Advanced ensemble model using stacking and blending for NBA predictions
    """
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        
    def _create_base_models(self):
        """Create diverse base models for ensemble"""
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, max_features='sqrt', random_state=42
            ),
            'xgb': XGBClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            # 'lgb': LGBMClassifier(
            #     n_estimators=300, max_depth=10, learning_rate=0.05,
            #     subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            # ),
            'gb': GradientBoostingClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            'et': ExtraTreesClassifier(
                n_estimators=500, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, max_features='sqrt', random_state=42
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', alpha=0.001, max_iter=1000, random_state=42
            )
        }
        
        # Meta-model for stacking
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    def _create_advanced_features(self, X):
        """Create advanced feature interactions and transformations"""
        X_advanced = X.copy()
        
        # Feature interactions
        if 'HOME_SEASON_WIN_PCT' in X.columns and 'AWAY_SEASON_WIN_PCT' in X.columns:
            X_advanced['WIN_PCT_INTERACTION'] = X['HOME_SEASON_WIN_PCT'] * X['AWAY_SEASON_WIN_PCT']
            X_advanced['WIN_PCT_SQUARED_DIFF'] = (X['HOME_SEASON_WIN_PCT'] - X['AWAY_SEASON_WIN_PCT']) ** 2
        
        if 'HOME_PTS_rolling5' in X.columns and 'AWAY_PTS_rolling5' in X.columns:
            X_advanced['PTS_PRODUCT'] = X['HOME_PTS_rolling5'] * X['AWAY_PTS_rolling5']
            X_advanced['PTS_RATIO'] = X['HOME_PTS_rolling5'] / (X['AWAY_PTS_rolling5'] + 1)
        
        if 'HOME_FG_PCT_rolling5' in X.columns and 'AWAY_FG_PCT_rolling5' in X.columns:
            X_advanced['FG_PCT_PRODUCT'] = X['HOME_FG_PCT_rolling5'] * X['AWAY_FG_PCT_rolling5']
            X_advanced['FG_PCT_GEOMETRIC_MEAN'] = np.sqrt(X['HOME_FG_PCT_rolling5'] * X['AWAY_FG_PCT_rolling5'])
        
        # Polynomial features for key metrics
        if 'WIN_PCT_DIFF' in X.columns:
            X_advanced['WIN_PCT_DIFF_SQUARED'] = X['WIN_PCT_DIFF'] ** 2
            X_advanced['WIN_PCT_DIFF_CUBED'] = X['WIN_PCT_DIFF'] ** 3
        
        if 'PTS_DIFF' in X.columns:
            X_advanced['PTS_DIFF_SQUARED'] = X['PTS_DIFF'] ** 2
            X_advanced['PTS_DIFF_ABS'] = np.abs(X['PTS_DIFF'])
        
        # Advanced statistical features
        if 'HOME_STRENGTH_SCORE' in X.columns and 'AWAY_STRENGTH_SCORE' in X.columns:
            X_advanced['STRENGTH_PRODUCT'] = X['HOME_STRENGTH_SCORE'] * X['AWAY_STRENGTH_SCORE']
            X_advanced['STRENGTH_HARMONIC_MEAN'] = 2 / (1/X['HOME_STRENGTH_SCORE'] + 1/X['AWAY_STRENGTH_SCORE'])
        
        # Momentum and trend features
        if 'HOME_MOMENTUM' in X.columns and 'AWAY_MOMENTUM' in X.columns:
            X_advanced['MOMENTUM_PRODUCT'] = X['HOME_MOMENTUM'] * X['AWAY_MOMENTUM']
            X_advanced['MOMENTUM_RATIO'] = X['HOME_MOMENTUM'] / (X['AWAY_MOMENTUM'] + 0.01)
        
        return X_advanced
    
    def fit(self, X, y):
        """Train the advanced ensemble model"""
        print("🚀 Training Advanced NBA Ensemble...")
        
        # Create base models
        self._create_base_models()
        
        # Create advanced features
        X_advanced = self._create_advanced_features(X)
        
        # Prepare for stacking
        n_samples = len(X_advanced)
        n_models = len(self.base_models)
        
        # Create out-of-fold predictions for stacking
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, n_models))
        
        print("📊 Training base models with cross-validation...")
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training {name}...")
            
            # Get out-of-fold predictions
            oof_predictions = np.zeros(n_samples)
            
            for train_idx, val_idx in skf.split(X_advanced, y):
                X_train_fold, X_val_fold = X_advanced.iloc[train_idx], X_advanced.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                oof_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            
            meta_features[:, i] = oof_predictions
            
            # Train on full data
            model.fit(X_advanced, y)
        
        # Train meta-model
        print("🎯 Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        print("✅ Advanced ensemble training complete!")
        
        return self
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create advanced features
        X_advanced = self._create_advanced_features(X)
        
        # Get base model predictions
        base_predictions = np.zeros((len(X_advanced), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, i] = model.predict_proba(X_advanced)[:, 1]
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict_proba(base_predictions)
        
        return meta_predictions
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate the ensemble model"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }

def create_advanced_features(df):
    """Create additional advanced features for the dataset"""
    df_advanced = df.copy()
    
    # Time-based features (simulated)
    np.random.seed(42)
    df_advanced['GAME_DAY_OF_WEEK'] = np.random.randint(0, 7, len(df))
    df_advanced['GAME_MONTH'] = np.random.randint(1, 13, len(df))
    df_advanced['IS_WEEKEND'] = (df_advanced['GAME_DAY_OF_WEEK'] >= 5).astype(int)
    
    # Team fatigue simulation
    df_advanced['HOME_FATIGUE'] = np.random.exponential(0.1, len(df))
    df_advanced['AWAY_FATIGUE'] = np.random.exponential(0.1, len(df))
    df_advanced['FATIGUE_DIFF'] = df_advanced['HOME_FATIGUE'] - df_advanced['AWAY_FATIGUE']
    
    # Travel distance simulation (affects away teams more)
    df_advanced['TRAVEL_DISTANCE'] = np.random.exponential(500, len(df))
    df_advanced['TRAVEL_IMPACT'] = df_advanced['TRAVEL_DISTANCE'] * 0.0001
    
    # Weather impact (simulated)
    df_advanced['WEATHER_IMPACT'] = np.random.normal(0, 0.05, len(df))
    
    # Crowd impact (home advantage)
    df_advanced['CROWD_IMPACT'] = np.random.normal(0.02, 0.01, len(df))
    
    # Referee bias simulation
    df_advanced['REF_BIAS'] = np.random.normal(0, 0.01, len(df))
    
    return df_advanced
