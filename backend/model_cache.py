"""
Model caching system to pre-train and store all models for fast switching
"""

import pickle
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_model import train_model

# Lazy imports for deep learning frameworks - KEPT FOR RESUME VALUE
# These will be imported only when needed to prevent startup delays

# Initialize as None - will be imported lazily when needed
train_pytorch_model = None
predict_pytorch = None
train_tensorflow_model = None
train_ensemble_model = None

# Availability flags - will be set when modules are actually imported
PYTORCH_AVAILABLE = None  # None means "not checked yet"
TENSORFLOW_AVAILABLE = None
ENSEMBLE_AVAILABLE = None

def _lazy_import_pytorch():
    """Lazy import PyTorch modules - only when actually needed"""
    global PYTORCH_AVAILABLE, train_pytorch_model, predict_pytorch
    if PYTORCH_AVAILABLE is None:  # Only check once
        try:
            from pytorch_model import train_pytorch_model, predict_pytorch
            PYTORCH_AVAILABLE = True
            print("✅ PyTorch modules loaded successfully")
        except ImportError as e:
            PYTORCH_AVAILABLE = False
            train_pytorch_model = None
            predict_pytorch = None
            print(f"⚠️ PyTorch not available: {e}")
    return PYTORCH_AVAILABLE

def _lazy_import_tensorflow():
    """Lazy import TensorFlow modules - only when actually needed"""
    global TENSORFLOW_AVAILABLE, train_tensorflow_model
    if TENSORFLOW_AVAILABLE is None:  # Only check once
        try:
            from tensorflow_model import train_tensorflow_model
            TENSORFLOW_AVAILABLE = True
            print("✅ TensorFlow modules loaded successfully")
        except ImportError as e:
            TENSORFLOW_AVAILABLE = False
            train_tensorflow_model = None
            print(f"⚠️ TensorFlow not available: {e}")
    return TENSORFLOW_AVAILABLE

def _lazy_import_ensemble():
    """Lazy import Ensemble modules - only when actually needed"""
    global ENSEMBLE_AVAILABLE, train_ensemble_model
    if ENSEMBLE_AVAILABLE is None:  # Only check once
        try:
            from ensemble_model import train_ensemble_model
            ENSEMBLE_AVAILABLE = True
            print("✅ Ensemble modules loaded successfully")
        except ImportError as e:
            ENSEMBLE_AVAILABLE = False
            train_ensemble_model = None
            print(f"⚠️ Ensemble not available: {e}")
    return ENSEMBLE_AVAILABLE

class ModelCache:
    """
    Cache system for pre-trained models to enable fast switching
    """
    
    def __init__(self, cache_dir="model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models = {}
        self.is_trained = False
    
    def train_all_models(self, X, y, quick_mode=True):
        """
        Train all models and cache them for fast access
        
        Args:
            X: Feature matrix
            y: Target variable
            quick_mode: If True, use fewer epochs for faster training
        """
        print("🚀 Training all models for fast switching...")
        print("="*60)
        
        # Training parameters based on mode
        if quick_mode:
            pytorch_epochs = 20
            tensorflow_epochs = 20
            print("⚡ Quick mode: Using fewer epochs for faster training")
        else:
            pytorch_epochs = 50
            tensorflow_epochs = 50
            print("🔥 Full mode: Using full epochs for best performance")
        
        # 1. Traditional ML Models (Fast)
        print("\n1. Training Traditional ML Models")
        print("-" * 40)
        
        traditional_models = ['xgb', 'rf', 'logreg']
        for model_type in traditional_models:
            try:
                print(f"Training {model_type.upper()}...")
                model, _, _ = train_model(X, y, model_type=model_type)
                self.models[model_type] = model
                print(f"✓ {model_type.upper()} trained successfully")
            except Exception as e:
                print(f"✗ {model_type.upper()} failed: {e}")
        
        # 2. PyTorch Model
        print("\n2. Training PyTorch Model")
        print("-" * 40)
        if _lazy_import_pytorch():
            try:
                print("Training PyTorch hybrid model...")
                model, test_data, scaler, train_losses = train_pytorch_model(
                    X, y, model_type='hybrid', epochs=pytorch_epochs
                )
                self.models['pytorch'] = model
                self.models['pytorch_scaler'] = scaler
                print("✓ PyTorch model trained successfully")
            except Exception as e:
                print(f"✗ PyTorch model failed: {e}")
        else:
            print("⚠️ PyTorch not available - skipping PyTorch model")
        
        # 3. TensorFlow Model
        print("\n3. Training TensorFlow Model")
        print("-" * 40)
        if _lazy_import_tensorflow():
            try:
                print("Training TensorFlow hybrid model...")
                model, test_data, history = train_tensorflow_model(
                    X, y, model_type='hybrid', epochs=tensorflow_epochs
                )
                self.models['tensorflow'] = model
                print("✓ TensorFlow model trained successfully")
            except Exception as e:
                print(f"✗ TensorFlow model failed: {e}")
        else:
            print("⚠️ TensorFlow not available - skipping TensorFlow model")
        
        # 4. Ensemble Model
        print("\n4. Training Ensemble Model")
        print("-" * 40)
        if _lazy_import_ensemble():
            try:
                print("Training ensemble model...")
                ensemble, results = train_ensemble_model(
                    X, y, 
                    use_pytorch=_lazy_import_pytorch(), 
                    use_tensorflow=_lazy_import_tensorflow(), 
                    use_traditional=True
                )
                self.models['ensemble'] = ensemble
                print("✓ Ensemble model trained successfully")
            except Exception as e:
                print(f"✗ Ensemble model failed: {e}")
        else:
            print("⚠️ Ensemble model not available - skipping ensemble model")
        
        self.is_trained = True
        print(f"\n🎉 All models trained! Ready for fast switching.")
        
        return self.models
    
    def get_model(self, model_type):
        """
        Get a pre-trained model by type
        
        Args:
            model_type: 'xgb', 'rf', 'logreg', 'pytorch', 'tensorflow', 'ensemble'
        
        Returns:
            model, scaler (if applicable)
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not available")
        
        if model_type == 'pytorch':
            return self.models['pytorch'], self.models.get('pytorch_scaler')
        else:
            return self.models[model_type], None
    
    def save_models(self, filename="cached_models.pkl"):
        """
        Save all trained models to disk
        """
        if not self.is_trained:
            raise ValueError("No models to save. Train models first.")
        
        cache_file = self.cache_dir / filename
        
        # Save models (excluding scaler which is handled separately)
        models_to_save = {k: v for k, v in self.models.items() if k != 'pytorch_scaler'}
        
        with open(cache_file, 'wb') as f:
            pickle.dump(models_to_save, f)
        
        # Save scaler separately if it exists
        if 'pytorch_scaler' in self.models:
            scaler_file = self.cache_dir / "pytorch_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.models['pytorch_scaler'], f)
        
        print(f"💾 Models saved to {cache_file}")
    
    def load_models(self, filename="cached_models.pkl"):
        """
        Load pre-trained models from disk
        """
        cache_file = self.cache_dir / filename
        scaler_file = self.cache_dir / "pytorch_scaler.pkl"
        
        if not cache_file.exists():
            print(f"❌ Cache file {cache_file} not found")
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                self.models = pickle.load(f)
            
            # Load scaler if it exists
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.models['pytorch_scaler'] = pickle.load(f)
            
            self.is_trained = True
            print(f"✅ Models loaded from {cache_file}")
            print(f"Available models: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_available_models(self):
        """
        Get list of available model types
        """
        if not self.is_trained:
            return []
        
        available = []
        for model_type in ['xgb', 'rf', 'logreg', 'pytorch', 'tensorflow', 'ensemble']:
            if model_type in self.models:
                available.append(model_type)
        
        return available
    
    def predict(self, model_type, X):
        """
        Make prediction using cached model
        
        Args:
            model_type: Type of model to use
            X: Feature matrix
        
        Returns:
            prediction, probability
        """
        model, scaler = self.get_model(model_type)
        
        if model_type == 'pytorch':
            if _lazy_import_pytorch() and predict_pytorch is not None:
                return predict_pytorch(model, X, scaler)
            else:
                raise ValueError("PyTorch model not available")
        elif model_type == 'tensorflow':
            if _lazy_import_tensorflow():
                y_pred, y_proba = model.predict(X)
                return y_pred, y_proba
            else:
                raise ValueError("TensorFlow model not available")
        elif model_type == 'ensemble':
            if _lazy_import_ensemble():
                return model.predict_ensemble(X)
            else:
                raise ValueError("Ensemble model not available")
        else:  # Traditional ML
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            return y_pred, y_proba

def create_model_cache(X, y, quick_mode=True, save_cache=True):
    """
    Convenience function to create and train model cache
    
    Args:
        X: Feature matrix
        y: Target variable
        quick_mode: Use fewer epochs for faster training
        save_cache: Save models to disk for future use
    
    Returns:
        ModelCache object
    """
    cache = ModelCache()
    
    # Try to load existing cache first
    if cache.load_models():
        print("🚀 Using cached models for instant switching!")
        return cache
    
    # Train new models if cache doesn't exist
    cache.train_all_models(X, y, quick_mode=quick_mode)
    
    if save_cache:
        cache.save_models()
    
    return cache
