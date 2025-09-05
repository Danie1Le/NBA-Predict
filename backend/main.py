#!/usr/bin/env python3
"""
Minimal FastAPI backend for NBA Game Predictor
Uses only already installed packages
"""

import sys
import os
from pathlib import Path

# Import modules directly from current directory (backend/)
# All required files are now copied to the backend directory

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import uvicorn

# Import our models
from model_cache import ModelCache
from preprocessing import load_and_clean_data
from feature_engineering import create_features

# Initialize FastAPI app
app = FastAPI(
    title="NBA Game Predictor API",
    description="Predict NBA game outcomes using machine learning",
    version="1.0.0"
)

# Enable CORS for React frontend
# Get CORS origins from environment variable or use defaults
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,https://nba-predict.vercel.app").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global variables for loaded data and models
model_cache = None
games_df = None
teams_df = None
team_map = None
features = None

class PredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    model_type: str = "ensemble"

class PredictionResponse(BaseModel):
    prediction: int
    home_win_probability: float
    away_win_probability: float
    confidence: str
    model_used: str

class TeamInfo(BaseModel):
    id: int
    abbreviation: str
    full_name: str

# Remove startup event to prevent blocking
# Models will be loaded on first request instead

async def ensure_models_loaded():
    """Ensure models are loaded, load them if not already loaded"""
    global model_cache, games_df, teams_df, team_map, features
    
    if model_cache is None or teams_df is None:
        print("📊 Loading models on first request...")
        await load_models_async()
    
async def load_models_async():
    """Load models asynchronously in background"""
    global model_cache, games_df, teams_df, team_map, features
    
    try:
        # Data and cache directories are now in the backend directory
        data_dir = 'Data'
        cache_dir = 'model_cache'
        
        print(f"📁 Using data directory: {data_dir}")
        print(f"📁 Using cache directory: {cache_dir}")
        
        # Load teams data
        
        teams_df = pd.read_csv(f'{data_dir}/NBA_TEAMS.csv')
        team_map = dict(zip(teams_df['id'], teams_df['abbreviation']))
        
        # Load and process games data
        games_df = load_and_clean_data(f'{data_dir}/NBA_GAMES.csv')
        games_df = create_features(games_df)
        
        # Define features
        features = [
            'HOME', 'PTS_rolling5', 'FG_PCT_rolling5', 'FG3_PCT_rolling5', 'FT_PCT_rolling5',
            'REB_rolling5', 'AST_rolling5', 'TOV_rolling5', 'PTS_rolling10', 'FG_PCT_rolling10', 
            'FG3_PCT_rolling10', 'FT_PCT_rolling10', 'REB_rolling10', 'AST_rolling10', 'TOV_rolling10',
            'WIN_STREAK5', 'SEASON_WIN_PCT', 'OPP_PTS_rolling5', 'OPP_FG_PCT_rolling5', 'OPP_FG3_PCT_rolling5', 
            'OPP_FT_PCT_rolling5', 'OPP_REB_rolling5', 'OPP_AST_rolling5', 'OPP_TOV_rolling5',
            'OPP_PTS_rolling10', 'OPP_FG_PCT_rolling10', 'OPP_FG3_PCT_rolling10', 'OPP_FT_PCT_rolling10',
            'OPP_REB_rolling10', 'OPP_AST_rolling10', 'OPP_TOV_rolling10', 'OPP_SEASON_WIN_PCT',
            'REST_DAYS', 'OPP_REST_DAYS'
        ]
        
        # Staged model loading: try full cache first, then traditional models
        try:
            model_cache = ModelCache(cache_dir=cache_dir)
            
            # Try to load full model cache (includes deep learning models)
            if model_cache.load_models(filename="cached_models.pkl"):
                print("✅ All pre-trained models loaded successfully!")
                print(f"📈 Available models: {model_cache.get_available_models()}")
                print("🚀 API ready with instant predictions (including deep learning)!")
            
            # Fallback to traditional models only
            elif model_cache.load_models(filename="traditional_models.pkl"):
                print("✅ Traditional ML models loaded successfully!")
                print(f"📈 Available models: {model_cache.get_available_models()}")
                print("🚀 API ready with fast predictions (traditional ML)!")
                print("💡 Deep learning models will be trained in background...")
                
                # Auto-upgrade to deep learning models in background
                import asyncio
                asyncio.create_task(upgrade_to_deep_learning_models())
            
            else:
                print("❌ No cached models found - will train on first request")
                model_cache = None
                
        except Exception as e:
            print(f"⚠️ Error loading cached models: {e}")
            print("📊 Will train models on first request")
            model_cache = None
        
        print("🎯 API ready for predictions!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - let the API start anyway
        print("⚠️ API starting without full functionality")

async def train_deep_learning_models_async(model_cache, X, y):
    """Train deep learning models in background to upgrade from traditional models"""
    try:
        print("🧠 Starting background training of deep learning models...")
        print("⏳ This will upgrade your API to include PyTorch, TensorFlow, and Ensemble models")
        
        # Import availability flags
        from model_cache import PYTORCH_AVAILABLE, TENSORFLOW_AVAILABLE, ENSEMBLE_AVAILABLE
        
        # Train PyTorch model
        if PYTORCH_AVAILABLE:
            try:
                print("🔥 Training PyTorch model in background...")
                from pytorch_model import train_pytorch_model
                model, _, scaler, _ = train_pytorch_model(X, y, model_type='hybrid', epochs=30)
                model_cache.models['pytorch'] = model
                model_cache.models['pytorch_scaler'] = scaler
                print("✅ PyTorch model trained and ready!")
            except Exception as e:
                print(f"❌ PyTorch background training failed: {e}")
        
        # Train TensorFlow model
        if TENSORFLOW_AVAILABLE:
            try:
                print("🤖 Training TensorFlow model in background...")
                from tensorflow_model import train_tensorflow_model
                model, _, _ = train_tensorflow_model(X, y, model_type='hybrid', epochs=30)
                model_cache.models['tensorflow'] = model
                print("✅ TensorFlow model trained and ready!")
            except Exception as e:
                print(f"❌ TensorFlow background training failed: {e}")
        
        # Train Ensemble model
        if ENSEMBLE_AVAILABLE and (PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE):
            try:
                print("🎯 Training Ensemble model in background...")
                from ensemble_model import train_ensemble_model
                ensemble, _ = train_ensemble_model(
                    X, y, 
                    use_pytorch=PYTORCH_AVAILABLE, 
                    use_tensorflow=TENSORFLOW_AVAILABLE, 
                    use_traditional=True
                )
                model_cache.models['ensemble'] = ensemble
                print("✅ Ensemble model trained and ready!")
            except Exception as e:
                print(f"❌ Ensemble background training failed: {e}")
        
        # Save the upgraded model cache
        try:
            model_cache.save_models(filename="cached_models.pkl")
            print("💾 Deep learning models cached for future deployments!")
        except Exception as e:
            print(f"⚠️ Failed to save upgraded models: {e}")
        
        print("🎉 Background training complete! All models now available.")
        print(f"📈 Available models: {model_cache.get_available_models()}")
        
    except Exception as e:
        print(f"❌ Background training error: {e}")

async def upgrade_to_deep_learning_models():
    """Upgrade from traditional models to include deep learning models"""
    global model_cache, games_df, features
    
    if model_cache is None or games_df is None:
        print("⚠️ Cannot upgrade models - base models or data not loaded")
        return
    
    # Check if we already have deep learning models
    available_models = model_cache.get_available_models()
    has_deep_learning = any(model in available_models for model in ['pytorch', 'tensorflow', 'ensemble'])
    
    if has_deep_learning:
        print("✅ Deep learning models already available!")
        return
    
    print("🚀 Upgrading to deep learning models in background...")
    
    # Prepare training data
    X = games_df[features].fillna(0)
    y = (games_df['WL'] == 'W').astype(int)
    
    # Start background training
    import asyncio
    asyncio.create_task(train_deep_learning_models_async(model_cache, X, y))

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NBA Game Predictor API",
        "status": "running",
        "models_loaded": model_cache is not None and hasattr(model_cache, 'is_trained') and model_cache.is_trained
    }

@app.get("/teams", response_model=List[TeamInfo])
async def get_teams():
    """Get list of all NBA teams"""
    await ensure_models_loaded()
    
    if teams_df is None:
        raise HTTPException(status_code=500, detail="Teams data not loaded")
    
    return [
        TeamInfo(
            id=int(row['id']),
            abbreviation=row['abbreviation'],
            full_name=row['full_name']
        )
        for _, row in teams_df.iterrows()
    ]

@app.get("/team-stats/{team_id}")
async def get_team_stats(team_id: int):
    """Get team statistics"""
    if games_df is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    try:
        # Get team games
        team_games = games_df[games_df['Team_ID'] == team_id].sort_values('GAME_DATE_REAL', ascending=False)
        team_games = team_games.drop_duplicates(subset=['Game_ID', 'GAME_DATE'])
        
        if len(team_games) == 0:
            raise HTTPException(status_code=404, detail="Team not found")
        
        # Calculate stats with error handling
        last_5_games = team_games.head(5)
        last_10_games = team_games.head(10)
        
        # Simple stats calculation
        def safe_mean(series):
            try:
                return float(series.mean()) if len(series) > 0 else 0.0
            except:
                return 0.0
        
        def safe_sum(series):
            try:
                return int(series.sum()) if len(series) > 0 else 0
            except:
                return 0
        
        # Last 5 games stats
        last_5_stats = {
            'wins': safe_sum(last_5_games['WL'].apply(lambda x: 1 if x == 'W' else 0)),
            'games': len(last_5_games),
            'PTS': safe_mean(last_5_games['PTS']),
            'FG_PCT': safe_mean(last_5_games['FG_PCT']),
            'FG3_PCT': safe_mean(last_5_games['FG3_PCT']),
            'FT_PCT': safe_mean(last_5_games['FT_PCT']),
            'REB': safe_mean(last_5_games['REB']),
            'AST': safe_mean(last_5_games['AST']),
            'TOV': safe_mean(last_5_games['TOV'])
        }
        
        # Last 10 games stats
        last_10_stats = {
            'wins': safe_sum(last_10_games['WL'].apply(lambda x: 1 if x == 'W' else 0)),
            'games': len(last_10_games)
        }
        
        # Season stats (regular season only)
        season_games = team_games[~team_games['Game_ID'].astype(str).str.startswith('4240')]
        season_stats = {
            'wins': safe_sum(season_games['WL'].apply(lambda x: 1 if x == 'W' else 0)),
            'games': len(season_games),
            'win_pct': safe_mean(season_games['WL'].apply(lambda x: 1 if x == 'W' else 0)),
            'PTS': safe_mean(season_games['PTS']),
            'FG_PCT': safe_mean(season_games['FG_PCT']),
            'FG3_PCT': safe_mean(season_games['FG3_PCT']),
            'FT_PCT': safe_mean(season_games['FT_PCT'])
        }
        
        # Playoff stats
        playoff_games = team_games[team_games['Game_ID'].astype(str).str.startswith('4240')]
        playoff_stats = {
            'wins': safe_sum(playoff_games['WL'].apply(lambda x: 1 if x == 'W' else 0)),
            'games': len(playoff_games)
        }
        
        return {
            'team_id': team_id,
            'last_5': last_5_stats,
            'last_10': last_10_stats,
            'season': season_stats,
            'playoffs': playoff_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting team stats: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """Predict the outcome of a game"""
    global model_cache
    await ensure_models_loaded()
    
    if model_cache is None:
        # Try to train models on first prediction request (fast fallback)
        print("🚀 Training minimal model on first prediction request...")
        try:
            X = games_df[features].fillna(0)
            y = (games_df['WL'] == 'W').astype(int)
            
            model_cache = ModelCache(cache_dir='model_cache')
            
            # Train only the fastest model (XGBoost) for immediate response
            print("⚡ Training XGBoost for immediate predictions...")
            try:
                from train_model import train_model
                model, _, _ = train_model(X, y, model_type='xgb')
                model_cache.models['xgb'] = model
                model_cache.is_trained = True
                print("✓ XGBoost trained successfully - API ready!")
                
                # Save this model for future use
                try:
                    model_cache.save_models()
                    print("💾 Model cached for future deployments")
                except:
                    pass  # Don't fail if saving doesn't work
                
            except Exception as e:
                print(f"❌ XGBoost training failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to train fallback model")
            
        except Exception as e:
            print(f"❌ Failed to train models: {e}")
            raise HTTPException(status_code=500, detail="Failed to train models")
    
    if not hasattr(model_cache, 'is_trained') or not model_cache.is_trained:
        raise HTTPException(status_code=500, detail="Models not trained")
    
    if games_df is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    try:
        print(f"🎯 Making prediction for {request.home_team_id} vs {request.away_team_id} using {request.model_type}")
        
        # Create prediction input
        input_data = create_prediction_input(
            request.home_team_id, 
            request.away_team_id, 
            games_df, 
            team_map
        )
        
        if input_data is None:
            print("❌ Failed to create prediction input")
            raise HTTPException(status_code=400, detail="Could not create prediction input")
        
        print(f"✅ Prediction input created with {len(input_data)} features")
        
        # Use the fastest available model if requested model not available
        available_models = model_cache.get_available_models()
        model_to_use = request.model_type
        
        if model_to_use not in available_models:
            # Fallback to fastest available model
            if 'xgb' in available_models:
                model_to_use = 'xgb'
            elif 'rf' in available_models:
                model_to_use = 'rf'
            elif 'logreg' in available_models:
                model_to_use = 'logreg'
            elif available_models:
                model_to_use = available_models[0]
            else:
                raise HTTPException(status_code=500, detail="No models available")
            print(f"⚠️ Requested model '{request.model_type}' not available, using '{model_to_use}'")
        
        # Make prediction
        X_input = pd.DataFrame([input_data])[features]
        print(f"📊 Input shape: {X_input.shape}")
        
        y_pred, y_proba = model_cache.predict(model_to_use, X_input)
        print(f"🎯 Prediction result: {y_pred}, probabilities: {y_proba}")
        
        # Convert to proper formats
        if model_to_use in ['pytorch', 'tensorflow', 'ensemble']:
            prediction = int(y_pred[0])
            home_win_prob = float(y_proba[0])
            away_win_prob = float(1 - y_proba[0])
        else:
            prediction = int(y_pred[0])
            home_win_prob = float(y_proba[0][1])
            away_win_prob = float(y_proba[0][0])
        
        # Determine confidence
        max_confidence = max(home_win_prob, away_win_prob)
        if max_confidence > 0.7:
            confidence = "HIGH"
        elif max_confidence > 0.6:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        return PredictionResponse(
            prediction=prediction,
            home_win_probability=home_win_prob,
            away_win_probability=away_win_prob,
            confidence=confidence,
            model_used=model_to_use
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def create_prediction_input(home_team_id: int, away_team_id: int, games_df: pd.DataFrame, team_map: Dict) -> Optional[Dict]:
    """Create prediction input from team IDs"""
    try:
        # Get recent games for both teams
        home_team_games = games_df[games_df['Team_ID'] == home_team_id].tail(10)
        away_team_games = games_df[games_df['Team_ID'] == away_team_id].tail(10)
        
        if home_team_games.empty or away_team_games.empty:
            return None
        
        # Use the most recent game data
        home_latest = home_team_games.iloc[-1]
        away_latest = away_team_games.iloc[-1]
        
        # Create input data
        input_data = {
            'HOME': 1,  # Home team
            'PTS_rolling5': home_latest['PTS_rolling5'],
            'FG_PCT_rolling5': home_latest['FG_PCT_rolling5'],
            'FG3_PCT_rolling5': home_latest['FG3_PCT_rolling5'],
            'FT_PCT_rolling5': home_latest['FT_PCT_rolling5'],
            'REB_rolling5': home_latest['REB_rolling5'],
            'AST_rolling5': home_latest['AST_rolling5'],
            'TOV_rolling5': home_latest['TOV_rolling5'],
            'PTS_rolling10': home_latest['PTS_rolling10'],
            'FG_PCT_rolling10': home_latest['FG_PCT_rolling10'],
            'FG3_PCT_rolling10': home_latest['FG3_PCT_rolling10'],
            'FT_PCT_rolling10': home_latest['FT_PCT_rolling10'],
            'REB_rolling10': home_latest['REB_rolling10'],
            'AST_rolling10': home_latest['AST_rolling10'],
            'TOV_rolling10': home_latest['TOV_rolling10'],
            'WIN_STREAK5': home_latest['WIN_STREAK5'],
            'SEASON_WIN_PCT': home_latest['SEASON_WIN_PCT'],
            'OPP_PTS_rolling5': away_latest['PTS_rolling5'],
            'OPP_FG_PCT_rolling5': away_latest['FG_PCT_rolling5'],
            'OPP_FG3_PCT_rolling5': away_latest['FG3_PCT_rolling5'],
            'OPP_FT_PCT_rolling5': away_latest['FT_PCT_rolling5'],
            'OPP_REB_rolling5': away_latest['REB_rolling5'],
            'OPP_AST_rolling5': away_latest['AST_rolling5'],
            'OPP_TOV_rolling5': away_latest['TOV_rolling5'],
            'OPP_PTS_rolling10': away_latest['PTS_rolling10'],
            'OPP_FG_PCT_rolling10': away_latest['FG_PCT_rolling10'],
            'OPP_FG3_PCT_rolling10': away_latest['FG3_PCT_rolling10'],
            'OPP_FT_PCT_rolling10': away_latest['FT_PCT_rolling10'],
            'OPP_REB_rolling10': away_latest['REB_rolling10'],
            'OPP_AST_rolling10': away_latest['AST_rolling10'],
            'OPP_TOV_rolling10': away_latest['TOV_rolling10'],
            'OPP_SEASON_WIN_PCT': away_latest['SEASON_WIN_PCT'],
            'REST_DAYS': home_latest['REST_DAYS'],
            'OPP_REST_DAYS': away_latest['REST_DAYS']
        }
        
        return input_data
        
    except Exception as e:
        print(f"Error creating prediction input: {e}")
        return None

@app.get("/models")
async def get_available_models():
    """Get list of available models and their status"""
    await ensure_models_loaded()
    
    if model_cache is None:
        return {
            "available_models": [],
            "model_descriptions": {},
            "status": "No models loaded - will train on first prediction request"
        }
    
    available = model_cache.get_available_models()
    has_deep_learning = any(model in available for model in ['pytorch', 'tensorflow', 'ensemble'])
    
    return {
        "available_models": available,
        "model_descriptions": {
            'xgb': 'XGBoost (Gradient Boosting) - Fast & Accurate',
            'rf': 'Random Forest - Robust & Interpretable',
            'logreg': 'Logistic Regression - Simple & Fast',
            'pytorch': 'PyTorch Neural Network - Advanced Deep Learning',
            'tensorflow': 'TensorFlow/Keras - Production-Ready Deep Learning',
            'ensemble': 'Ensemble (All Models) - Best Performance'
        },
        "status": "All models ready" if has_deep_learning else "Traditional ML ready, deep learning training in background",
        "deep_learning_available": has_deep_learning,
        "recommended_model": "ensemble" if "ensemble" in available else "xgb"
    }

@app.post("/models/upgrade")
async def upgrade_models():
    """Manually trigger upgrade to deep learning models"""
    await ensure_models_loaded()
    
    if model_cache is None:
        return {
            "success": False,
            "message": "No base models loaded - cannot upgrade"
        }
    
    available = model_cache.get_available_models()
    has_deep_learning = any(model in available for model in ['pytorch', 'tensorflow', 'ensemble'])
    
    if has_deep_learning:
        return {
            "success": True,
            "message": "Deep learning models already available",
            "available_models": available
        }
    
    # Trigger upgrade
    import asyncio
    asyncio.create_task(upgrade_to_deep_learning_models())
    
    return {
        "success": True,
        "message": "Deep learning model training started in background",
        "estimated_time": "5-10 minutes",
        "current_models": available
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render default port is 10000
    
    print(f"🚀 Starting NBA Game Predictor API on port {port}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📄 Script location: {__file__}")
    
    # Run the app - disable reload in production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
