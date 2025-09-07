import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class AUCOptimizedEnsemble:
    """
    AUC-optimized ensemble model specifically designed to maximize AUC score
    """
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        
    def _create_auc_optimized_models(self):
        """Create models optimized for AUC performance"""
        self.base_models = {
            'rf_auc': RandomForestClassifier(
                n_estimators=3000, max_depth=50, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'xgb_auc': XGBClassifier(
                n_estimators=1500, max_depth=25, learning_rate=0.005,
                subsample=0.98, colsample_bytree=0.98, reg_alpha=0.5, reg_lambda=0.5,
                scale_pos_weight=1.5, random_state=42, n_jobs=-1
            ),
            'gb_auc': GradientBoostingClassifier(
                n_estimators=1500, max_depth=20, learning_rate=0.005,
                subsample=0.98, max_features='sqrt', random_state=42
            ),
            'et_auc': ExtraTreesClassifier(
                n_estimators=3000, max_depth=50, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
        }
        
        # Meta-model optimized for AUC
        self.meta_model = LogisticRegression(
            random_state=42, max_iter=2000, C=0.1, penalty='l2'
        )
    
    def _create_auc_features(self, X):
        """Create AUC-specific features"""
        X_auc = X.copy()
        
        # AUC-optimized feature interactions
        if 'AUC_OPTIMIZED_SCORE' in X.columns:
            X_auc['AUC_SCORE_SQUARED'] = X['AUC_OPTIMIZED_SCORE'] ** 2
            X_auc['AUC_SCORE_CUBED'] = X['AUC_OPTIMIZED_SCORE'] ** 3
        
        if 'OVERALL_STRENGTH_DIFF' in X.columns and 'PROBABILITY_DIFF' in X.columns:
            X_auc['STRENGTH_PROB_INTERACTION'] = X['OVERALL_STRENGTH_DIFF'] * X['PROBABILITY_DIFF']
        
        if 'HOME_WIN_LIKELIHOOD' in X.columns and 'AWAY_WIN_LIKELIHOOD' in X.columns:
            X_auc['LIKELIHOOD_RATIO'] = X['HOME_WIN_LIKELIHOOD'] / (X['AWAY_WIN_LIKELIHOOD'] + 0.01)
            X_auc['LIKELIHOOD_PRODUCT'] = X['HOME_WIN_LIKELIHOOD'] * X['AWAY_WIN_LIKELIHOOD']
        
        # Advanced AUC features
        if 'RANKING_DIFF' in X.columns:
            X_auc['RANKING_DIFF_ABS'] = np.abs(X['RANKING_DIFF'])
            X_auc['RANKING_DIFF_SQUARED'] = X['RANKING_DIFF'] ** 2
        
        return X_auc
    
    def fit(self, X, y):
        """Train the AUC-optimized ensemble model"""
        print("🎯 Training AUC-Optimized Ensemble...")
        
        # Create models
        self._create_auc_optimized_models()
        
        # Create AUC features
        X_auc = self._create_auc_features(X)
        
        # Prepare for stacking with AUC focus
        n_samples = len(X_auc)
        n_models = len(self.base_models)
        
        # Create out-of-fold predictions for stacking
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, n_models))
        
        print("📊 Training base models with AUC focus...")
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training {name}...")
            
            # Get out-of-fold predictions
            oof_predictions = np.zeros(n_samples)
            
            for train_idx, val_idx in skf.split(X_auc, y):
                X_train_fold, X_val_fold = X_auc.iloc[train_idx], X_auc.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                oof_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            
            meta_features[:, i] = oof_predictions
            
            # Train on full data
            model.fit(X_auc, y)
        
        # Train meta-model
        print("🎯 Training AUC-optimized meta-model...")
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        print("✅ AUC-optimized ensemble training complete!")
        
        return self
    
    def predict_proba(self, X):
        """Make probability predictions optimized for AUC"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create AUC features
        X_auc = self._create_auc_features(X)
        
        # Get base model predictions
        base_predictions = np.zeros((len(X_auc), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, i] = model.predict_proba(X_auc)[:, 1]
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict_proba(base_predictions)
        
        return meta_predictions
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate the ensemble model with AUC focus"""
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
