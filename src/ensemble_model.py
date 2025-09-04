import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import torch
import tensorflow as tf

# Import our custom models
try:
    from .train_model import train_model
    from .pytorch_model import train_pytorch_model, predict_pytorch
    from .tensorflow_model import train_tensorflow_model
except ImportError:
    # Fallback for when running as script
    from train_model import train_model
    from pytorch_model import train_pytorch_model, predict_pytorch
    from tensorflow_model import train_tensorflow_model

class NBAEnsemblePredictor:
    """
    Ensemble model that combines traditional ML (XGBoost, Random Forest) 
    with deep learning models (PyTorch, TensorFlow) for NBA game prediction.
    """
    
    def __init__(self, use_pytorch=True, use_tensorflow=True, use_traditional=True):
        self.use_pytorch = use_pytorch
        self.use_tensorflow = use_tensorflow
        self.use_traditional = use_traditional
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def train_ensemble(self, X, y, test_size=0.2, random_state=42):
        """
        Train all available models and create ensemble predictions.
        """
        print("Training NBA Prediction Ensemble Model...")
        print("="*60)
        
        # Split data for final evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train traditional ML models
        if self.use_traditional:
            print("\n1. Training Traditional ML Models")
            print("-" * 40)
            
            # XGBoost
            try:
                xgb_model, _, _ = train_model(X_train, y_train, model_type='xgb')
                self.models['xgboost'] = xgb_model
                print("✓ XGBoost trained successfully")
            except Exception as e:
                print(f"✗ XGBoost failed: {e}")
            
            # Random Forest
            try:
                rf_model, _, _ = train_model(X_train, y_train, model_type='rf')
                self.models['random_forest'] = rf_model
                print("✓ Random Forest trained successfully")
            except Exception as e:
                print(f"✗ Random Forest failed: {e}")
            
            # Logistic Regression
            try:
                lr_model, _, _ = train_model(X_train, y_train, model_type='logreg')
                self.models['logistic_regression'] = lr_model
                print("✓ Logistic Regression trained successfully")
            except Exception as e:
                print(f"✗ Logistic Regression failed: {e}")
        
        # Train PyTorch model
        if self.use_pytorch:
            print("\n2. Training PyTorch Deep Learning Model")
            print("-" * 40)
            try:
                pytorch_model, _, scaler_pytorch, _ = train_pytorch_model(
                    X_train, y_train, model_type='hybrid', epochs=50
                )
                self.models['pytorch'] = pytorch_model
                self.models['pytorch_scaler'] = scaler_pytorch
                print("✓ PyTorch model trained successfully")
            except Exception as e:
                print(f"✗ PyTorch model failed: {e}")
        
        # Train TensorFlow model
        if self.use_tensorflow:
            print("\n3. Training TensorFlow Deep Learning Model")
            print("-" * 40)
            try:
                tf_model, _, _ = train_tensorflow_model(
                    X_train, y_train, model_type='hybrid', epochs=50
                )
                self.models['tensorflow'] = tf_model
                print("✓ TensorFlow model trained successfully")
            except Exception as e:
                print(f"✗ TensorFlow model failed: {e}")
        
        # Calculate model weights based on individual performance
        print("\n4. Calculating Model Weights")
        print("-" * 40)
        self._calculate_weights(X_test, y_test)
        
        self.is_trained = True
        return X_test, y_test
    
    def _calculate_weights(self, X_test, y_test):
        """
        Calculate weights for each model based on their individual performance.
        """
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'pytorch_scaler':  # Skip scaler
                continue
                
            try:
                if name == 'pytorch':
                    y_pred, y_proba = predict_pytorch(model, X_test, self.models['pytorch_scaler'])
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                        score = (accuracy + auc) / 2  # Average of accuracy and AUC
                    except:
                        score = accuracy
                        
                elif name == 'tensorflow':
                    y_pred, y_proba = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                        score = (accuracy + auc) / 2
                    except:
                        score = accuracy
                        
                else:  # Traditional ML models
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_proba)
                        score = (accuracy + auc) / 2
                    except:
                        score = accuracy
                
                model_scores[name] = score
                print(f"{name}: {score:.4f}")
                
            except Exception as e:
                print(f"{name}: Failed to evaluate - {e}")
                model_scores[name] = 0.0
        
        # Normalize weights (softmax)
        if model_scores:
            scores_array = np.array(list(model_scores.values()))
            # Add small epsilon to avoid division by zero
            scores_array = scores_array + 1e-8
            weights = scores_array / np.sum(scores_array)
            
            for i, name in enumerate(model_scores.keys()):
                self.weights[name] = weights[i]
        
        print(f"\nModel Weights: {self.weights}")
    
    def predict_ensemble(self, X):
        """
        Make ensemble predictions by combining all trained models.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'pytorch_scaler':  # Skip scaler
                continue
                
            try:
                if name == 'pytorch':
                    y_pred, y_proba = predict_pytorch(model, X, self.models['pytorch_scaler'])
                    predictions[name] = y_pred.flatten()
                    probabilities[name] = y_proba.flatten()
                    
                elif name == 'tensorflow':
                    y_pred, y_proba = model.predict(X)
                    predictions[name] = y_pred.flatten()
                    probabilities[name] = y_proba.flatten()
                    
                else:  # Traditional ML models
                    y_pred = model.predict(X)
                    predictions[name] = y_pred
                    try:
                        y_proba = model.predict_proba(X)[:, 1]
                        probabilities[name] = y_proba
                    except:
                        probabilities[name] = y_pred
                        
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                continue
        
        # Weighted ensemble prediction
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros(len(X))
        total_weight = 0
        
        for name, proba in probabilities.items():
            if name in self.weights:
                weight = self.weights[name]
                ensemble_proba += weight * proba
                total_weight += weight
        
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        # Binary predictions
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate the ensemble model performance.
        """
        y_pred, y_proba = self.predict_ensemble(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        print("\n" + "="*60)
        print("ENSEMBLE MODEL EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def get_model_contributions(self, X_sample):
        """
        Analyze how much each model contributes to the final prediction.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing contributions")
        
        contributions = {}
        
        for name, model in self.models.items():
            if name == 'pytorch_scaler':
                continue
                
            try:
                if name == 'pytorch':
                    _, y_proba = predict_pytorch(model, X_sample, self.models['pytorch_scaler'])
                    contribution = y_proba.flatten() * self.weights.get(name, 0)
                    
                elif name == 'tensorflow':
                    _, y_proba = model.predict(X_sample)
                    contribution = y_proba.flatten() * self.weights.get(name, 0)
                    
                else:  # Traditional ML models
                    try:
                        y_proba = model.predict_proba(X_sample)[:, 1]
                    except:
                        y_proba = model.predict(X_sample)
                    contribution = y_proba * self.weights.get(name, 0)
                
                contributions[name] = contribution
                
            except Exception as e:
                print(f"Warning: Could not get contribution from {name}: {e}")
        
        return contributions
    
    def save_ensemble(self, filepath):
        """
        Save the ensemble model to disk.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save traditional ML models
        for name, model in self.models.items():
            if name not in ['pytorch', 'tensorflow', 'pytorch_scaler']:
                joblib.dump(model, f"{filepath}_{name}.pkl")
        
        # Save PyTorch model
        if 'pytorch' in self.models:
            torch.save(self.models['pytorch'].state_dict(), f"{filepath}_pytorch.pth")
            joblib.dump(self.models['pytorch_scaler'], f"{filepath}_pytorch_scaler.pkl")
        
        # Save TensorFlow model
        if 'tensorflow' in self.models:
            self.models['tensorflow'].save(f"{filepath}_tensorflow")
        
        # Save weights and metadata
        metadata = {
            'weights': self.weights,
            'use_pytorch': self.use_pytorch,
            'use_tensorflow': self.use_tensorflow,
            'use_traditional': self.use_traditional
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        print(f"Ensemble model saved to {filepath}")

def train_ensemble_model(X, y, use_pytorch=True, use_tensorflow=True, use_traditional=True):
    """
    Convenience function to train an ensemble model.
    """
    ensemble = NBAEnsemblePredictor(
        use_pytorch=use_pytorch,
        use_tensorflow=use_tensorflow,
        use_traditional=use_traditional
    )
    
    X_test, y_test = ensemble.train_ensemble(X, y)
    results = ensemble.evaluate_ensemble(X_test, y_test)
    
    return ensemble, results
