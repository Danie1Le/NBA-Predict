#!/usr/bin/env python3
"""
Pre-build script to train and cache models during Docker build
This ensures models are ready immediately when the API starts
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Import our modules
from preprocessing import load_and_clean_data
from feature_engineering import create_features
from model_cache import ModelCache

# Import memory monitoring
try:
    from memory_monitor import log_memory_usage, memory_efficient_model_training
except ImportError:
    # Fallback if memory monitor not available
    def log_memory_usage(context=""):
        pass
    
    class memory_efficient_model_training:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

def prebuild_models():
    """Pre-train and cache models with staged approach for fast API startup"""
    traditional_only = os.getenv('PREBUILD_TRADITIONAL_ONLY', '0') == '1'
    
    if traditional_only:
        print("🚀 Pre-building traditional ML models only (memory efficient)...")
        print("="*60)
    else:
        print("🚀 Pre-building all models with staged approach...")
        print("="*60)
    
    try:
        # Data and cache directories
        data_dir = 'Data'
        cache_dir = 'model_cache'
        
        print(f"📁 Using data directory: {data_dir}")
        print(f"📁 Using cache directory: {cache_dir}")
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(exist_ok=True)
        
        # Load and process data
        print("📊 Loading and processing data...")
        games_df = load_and_clean_data(f'{data_dir}/NBA_GAMES.csv')
        games_df = create_features(games_df)
        
        # Define advanced features for improved predictions (same as in main.py)
        features = [
            # Original features
            'HOME_PTS_rolling5', 'HOME_FG_PCT_rolling5', 'HOME_FG3_PCT_rolling5', 'HOME_FT_PCT_rolling5',
            'HOME_REB_rolling5', 'HOME_AST_rolling5', 'HOME_TOV_rolling5',
            'AWAY_PTS_rolling5', 'AWAY_FG_PCT_rolling5', 'AWAY_FG3_PCT_rolling5', 'AWAY_FT_PCT_rolling5',
            'AWAY_REB_rolling5', 'AWAY_AST_rolling5', 'AWAY_TOV_rolling5',
            'HOME_SEASON_WIN_PCT', 'AWAY_SEASON_WIN_PCT',
            
            # Advanced difference features
            'WIN_PCT_DIFF', 'WIN_PCT_RATIO', 'STRENGTH_ADVANTAGE',
            'PTS_DIFF', 'FG_PCT_DIFF', 'FG3_PCT_DIFF', 'FT_PCT_DIFF',
            'REB_DIFF', 'AST_DIFF', 'TOV_DIFF',
            
            # Efficiency and momentum features
            'HOME_EFFICIENCY', 'AWAY_EFFICIENCY', 'EFFICIENCY_DIFF',
            'HOME_MOMENTUM', 'AWAY_MOMENTUM', 'MOMENTUM_DIFF',
            'HOME_COURT_ADVANTAGE', 'STATS_DOMINANCE', 'TIER_MATCHUP',
            'HOME_RECENT_FORM', 'AWAY_RECENT_FORM', 'FORM_DIFF', 'CLUTCH_FACTOR'
        ]
        
        # Prepare training data (game-level)
        X = games_df[features].fillna(0)
        y = games_df['HOME_WON'].astype(int)
        
        print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        
        # Create model cache
        model_cache = ModelCache(cache_dir=cache_dir)
        
        # Stage 1: Train fast traditional models (for immediate API startup)
        print("\n🏃‍♂️ STAGE 1: Training traditional ML models (fast startup)...")
        print("-" * 50)
        log_memory_usage("before traditional model training")
        
        traditional_models = ['xgb', 'rf', 'logreg']
        for model_type in traditional_models:
            try:
                with memory_efficient_model_training():
                    print(f"Training {model_type.upper()}...")
                    from train_model import train_model
                    model, _, _ = train_model(X, y, model_type=model_type)
                    model_cache.models[model_type] = model
                    print(f"✓ {model_type.upper()} trained successfully")
            except Exception as e:
                print(f"✗ {model_type.upper()} failed: {e}")
        
        # Save traditional models immediately for fast API startup
        model_cache.is_trained = True
        model_cache.save_models(filename="traditional_models.pkl")
        print("💾 Traditional models saved - API can start immediately!")
        
        # Skip deep learning models if in traditional-only mode (for memory efficiency)
        if traditional_only:
            print("\n⚡ Traditional-only mode: Skipping deep learning models")
            print("🧠 Deep learning models will be trained in background after API starts")
            print("\n" + "="*60)
            print("✅ Traditional models pre-built successfully!")
            print(f"📈 Available models: {list(model_cache.models.keys())}")
            print("🚀 API will start instantly with traditional models")
            return
        
        # Stage 2: Train deep learning models (for advanced predictions)
        print("\n🧠 STAGE 2: Training deep learning models (advanced features)...")
        print("-" * 50)
        
        # Check if deep learning frameworks are available
        try:
            from model_cache import PYTORCH_AVAILABLE, TENSORFLOW_AVAILABLE, ENSEMBLE_AVAILABLE
            
            # Train PyTorch model with reduced epochs for build efficiency
            if PYTORCH_AVAILABLE:
                try:
                    print("Training PyTorch model...")
                    from pytorch_model import train_pytorch_model
                    model, _, scaler, _ = train_pytorch_model(X, y, model_type='hybrid', epochs=25)
                    model_cache.models['pytorch'] = model
                    model_cache.models['pytorch_scaler'] = scaler
                    print("✓ PyTorch model trained successfully")
                except Exception as e:
                    print(f"✗ PyTorch model failed: {e}")
            else:
                print("⚠️ PyTorch not available - skipping")
            
            # Train TensorFlow model with reduced epochs for build efficiency
            if TENSORFLOW_AVAILABLE:
                try:
                    print("Training TensorFlow model...")
                    from tensorflow_model import train_tensorflow_model
                    model, _, _ = train_tensorflow_model(X, y, model_type='hybrid', epochs=25)
                    model_cache.models['tensorflow'] = model
                    print("✓ TensorFlow model trained successfully")
                except Exception as e:
                    print(f"✗ TensorFlow model failed: {e}")
            else:
                print("⚠️ TensorFlow not available - skipping")
            
            # Train Ensemble model
            if ENSEMBLE_AVAILABLE and (PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE):
                try:
                    print("Training Ensemble model...")
                    from ensemble_model import train_ensemble_model
                    ensemble, _ = train_ensemble_model(
                        X, y, 
                        use_pytorch=PYTORCH_AVAILABLE, 
                        use_tensorflow=TENSORFLOW_AVAILABLE, 
                        use_traditional=True
                    )
                    model_cache.models['ensemble'] = ensemble
                    print("✓ Ensemble model trained successfully")
                except Exception as e:
                    print(f"✗ Ensemble model failed: {e}")
            else:
                print("⚠️ Ensemble not available - skipping")
        
        except ImportError as e:
            print(f"⚠️ Some deep learning frameworks not available: {e}")
            print("📊 Only traditional ML models will be available")
        
        # Save all models (including deep learning)
        model_cache.save_models(filename="cached_models.pkl")
        
        print("\n" + "="*60)
        print("✅ All models pre-built successfully!")
        print(f"📈 Available models: {list(model_cache.models.keys())}")
        print("🚀 API will start instantly with traditional models")
        print("🧠 Deep learning models ready for advanced predictions")
        
    except Exception as e:
        print(f"❌ Error during model pre-building: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the build - API can still train models on first request
        print("⚠️ Continuing with build - models will train on first request")

if __name__ == "__main__":
    prebuild_models()
