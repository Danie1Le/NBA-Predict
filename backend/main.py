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

@app.on_event("startup")
async def startup_event():
    """Load data and models on startup"""
    global games_df, teams_df, team_map, features, model_cache
    
    try:
        print("🚀 Loading NBA data and models...")
        
        # Load teams data
        data_dir = 'Data'
        teams_df = pd.read_csv(f'{data_dir}/NBA_TEAMS.csv')
        team_map = dict(zip(teams_df['id'], teams_df['abbreviation']))
        
        # Load and process games data (now game-level)
        games_df = load_and_clean_data(f'{data_dir}/NBA_GAMES.csv')
        games_df = create_features(games_df)
        
        # Define advanced features for improved predictions
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
        
        # Load model cache
        model_cache = ModelCache(cache_dir="model_cache")
        if not model_cache.load_models():
            print("⚠️ No cached models found - models will be trained on first request")
        else:
            print("✅ Models loaded from cache")
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Games: {len(games_df)}")
        print(f"🏀 Teams: {len(teams_df)}")
        print(f"🧠 Models: {len(model_cache.get_available_models())}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

# Enable CORS for React frontend
# Get CORS origins from environment variable or use defaults
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,https://nba-predict.vercel.app,https://*.vercel.app").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now to fix the 404 issue
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
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
        # Load only essential data first (teams), models in background
        await load_essential_data_async()
        # Start model loading in background without blocking
        import asyncio
        asyncio.create_task(load_models_async())

async def load_essential_data_async():
    """Load only essential data (teams) quickly without heavy processing"""
    global teams_df, team_map
    
    try:
        print("⚡ Loading essential data (teams only)...")
        data_dir = 'Data'
        
        # Load only teams data (lightweight)
        teams_df = pd.read_csv(f'{data_dir}/NBA_TEAMS.csv')
        team_map = dict(zip(teams_df['id'], teams_df['abbreviation']))
        
        print("✅ Essential data loaded - API ready for team selection!")
        
    except Exception as e:
        print(f"❌ Error loading essential data: {e}")
        teams_df = None
        team_map = None

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
        
        # Load and process games data (now game-level)
        games_df = load_and_clean_data(f'{data_dir}/NBA_GAMES.csv')
        games_df = create_features(games_df)
        
        # Define advanced features for improved predictions
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
        print("📝 NOTE: Deep learning models are temporarily disabled for fast deployment")
        
        # Import lazy import functions for deep learning models
        from model_cache import _lazy_import_pytorch, _lazy_import_tensorflow, _lazy_import_ensemble
        
        # Train PyTorch model
        if _lazy_import_pytorch():
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
        if _lazy_import_tensorflow():
            try:
                print("🤖 Training TensorFlow model in background...")
                from tensorflow_model import train_tensorflow_model
                model, _, _ = train_tensorflow_model(X, y, model_type='hybrid', epochs=30)
                model_cache.models['tensorflow'] = model
                print("✅ TensorFlow model trained and ready!")
            except Exception as e:
                print(f"❌ TensorFlow background training failed: {e}")
        
        # Train Ensemble model
        if _lazy_import_ensemble() and (_lazy_import_pytorch() or _lazy_import_tensorflow()):
            try:
                print("🎯 Training Ensemble model in background...")
                from ensemble_model import train_ensemble_model
                ensemble, _ = train_ensemble_model(
                    X, y, 
                    use_pytorch=_lazy_import_pytorch(), 
                    use_tensorflow=_lazy_import_tensorflow(), 
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
    
    # Prepare training data (game-level)
    X = games_df[features].fillna(0)
    y = games_df['HOME_WON'].astype(int)
    
    # Start background training
    import asyncio
    asyncio.create_task(train_deep_learning_models_async(model_cache, X, y))

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NBA Game Predictor API",
        "status": "running",
        "teams_loaded": teams_df is not None,
        "models_loaded": model_cache is not None and hasattr(model_cache, 'is_trained') and model_cache.is_trained,
        "available_models": model_cache.get_available_models() if model_cache else ["xgb"],
        "port": os.getenv("PORT", 8000),
        "ready_for_predictions": teams_df is not None
    }

@app.get("/health")
async def health_check():
    """Lightweight health check for keeping service warm"""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/teams", response_model=List[TeamInfo])
async def get_teams():
    """Get list of all NBA teams"""
    # Load teams data if not already loaded (fast operation)
    if teams_df is None:
        await load_essential_data_async()
    
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
        # Get team games - team can be either home or away
        home_games = games_df[games_df['HOME_TEAM_ID'] == team_id].sort_values('GAME_DATE_REAL', ascending=False)
        away_games = games_df[games_df['AWAY_TEAM_ID'] == team_id].sort_values('GAME_DATE_REAL', ascending=False)
        
        if len(home_games) == 0 and len(away_games) == 0:
            raise HTTPException(status_code=404, detail="Team not found")
        
        # Combine all games for this team
        all_games = []
        
        # Process home games
        for _, game in home_games.iterrows():
            all_games.append({
                'GAME_DATE_REAL': game['GAME_DATE_REAL'],
                'Game_ID': game['Game_ID'],
                'WON': game['HOME_WON'],  # 1 if home team won, 0 if lost
                'PTS': game['HOME_PTS'],
                'FG_PCT': game['HOME_FG_PCT'],
                'FG3_PCT': game['HOME_FG3_PCT'],
                'FT_PCT': game['HOME_FT_PCT'],
                'REB': game['HOME_REB'],
                'AST': game['HOME_AST'],
                'TOV': game['HOME_TOV']
            })
        
        # Process away games
        for _, game in away_games.iterrows():
            all_games.append({
                'GAME_DATE_REAL': game['GAME_DATE_REAL'],
                'Game_ID': game['Game_ID'],
                'WON': 1 - game['HOME_WON'],  # 1 if away team won, 0 if lost
                'PTS': game['AWAY_PTS'],
                'FG_PCT': game['AWAY_FG_PCT'],
                'FG3_PCT': game['AWAY_FG3_PCT'],
                'FT_PCT': game['AWAY_FT_PCT'],
                'REB': game['AWAY_REB'],
                'AST': game['AWAY_AST'],
                'TOV': game['AWAY_TOV']
            })
        
        # Convert to DataFrame and sort by date
        team_games_df = pd.DataFrame(all_games).sort_values('GAME_DATE_REAL', ascending=False)
        
        # Remove duplicates
        team_games_df = team_games_df.drop_duplicates(subset=['Game_ID'])
        
        # Calculate stats with error handling
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
        
        # Get last 5 and 10 games
        last_5_games = team_games_df.head(5)
        last_10_games = team_games_df.head(10)
        
        # Last 5 games stats
        last_5_stats = {
            'wins': safe_sum(last_5_games['WON']),
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
            'wins': safe_sum(last_10_games['WON']),
            'games': len(last_10_games)
        }
        
        # Season stats (all games)
        season_stats = {
            'wins': safe_sum(team_games_df['WON']),
            'games': len(team_games_df),
            'win_pct': safe_mean(team_games_df['WON']),
            'PTS': safe_mean(team_games_df['PTS']),
            'FG_PCT': safe_mean(team_games_df['FG_PCT']),
            'FG3_PCT': safe_mean(team_games_df['FG3_PCT']),
            'FT_PCT': safe_mean(team_games_df['FT_PCT'])
        }
        
        # Playoff stats (placeholder - no playoff data in current dataset)
        playoff_stats = {
            'wins': 0,
            'games': 0
        }
        
        return {
            'team_id': team_id,
            'last_5': last_5_stats,
            'last_10': last_10_stats,
            'season': season_stats,
            'playoffs': playoff_stats
        }
        
    except Exception as e:
        print(f"Error in team stats endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting team stats: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """Predict the outcome of a game"""
    global model_cache, games_df, features
    
    # Ensure data is loaded
    if games_df is None or features is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    # Check if we have models, if not train them
    if model_cache is None or not hasattr(model_cache, 'is_trained') or not model_cache.is_trained:
        print("🚀 Training models on prediction request...")
        try:
            # Prepare training data
            X = games_df[features].fillna(0)
            y = games_df['HOME_WON'].astype(int)
            
            # Create model cache
            model_cache = ModelCache(cache_dir='model_cache')
            
            # Train all traditional models for better options
            print("⚡ Training traditional ML models...")
            from train_model import train_model
            
            # Train XGBoost (fastest)
            print("Training XGBoost...")
            model, _, _ = train_model(X, y, model_type='xgb')
            model_cache.models['xgb'] = model
            print("✓ XGBoost trained successfully!")
            
            # Train Random Forest
            print("Training Random Forest...")
            model, _, _ = train_model(X, y, model_type='rf')
            model_cache.models['rf'] = model
            print("✓ Random Forest trained successfully!")
            
            # Train Logistic Regression
            print("Training Logistic Regression...")
            model, _, _ = train_model(X, y, model_type='logreg')
            model_cache.models['logreg'] = model
            print("✓ Logistic Regression trained successfully!")
            
            model_cache.is_trained = True
            print("🎉 All traditional models trained successfully!")
            
            # Try to save model (don't fail if it doesn't work)
            try:
                model_cache.save_models()
                print("💾 Model saved for future use")
            except Exception as save_error:
                print(f"⚠️ Could not save model: {save_error}")
                
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
    
    # Verify we have trained models
    if not model_cache.is_trained or len(model_cache.models) == 0:
        raise HTTPException(status_code=500, detail="No trained models available")
    
    # Check if requested model is available
    if request.model_type not in model_cache.models:
        available_models = list(model_cache.models.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_type}' not available. Available models: {available_models}"
        )
    
    if games_df is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    try:
        # Create prediction input (optimized for speed)
        input_data = create_prediction_input(
            request.home_team_id, 
            request.away_team_id, 
            games_df, 
            team_map
        )
        
        if input_data is None:
            raise HTTPException(status_code=400, detail="Could not create prediction input")
        
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
        
        # Make prediction (optimized)
        X_input = pd.DataFrame([input_data])[features]
        y_pred, y_proba = model_cache.predict(model_to_use, X_input)
        
        # Convert to proper formats
        if model_to_use in ['pytorch', 'tensorflow', 'ensemble']:
            prediction = int(y_pred[0])
            home_win_prob = float(y_proba[0])
            away_win_prob = float(1 - y_proba[0])
        else:
            prediction = int(y_pred[0])
            # For traditional models, y_proba[0] is [prob_class_0, prob_class_1]
            # where class_0 = away team wins, class_1 = home team wins
            away_win_prob = float(y_proba[0][0])  # Class 0: away team wins
            home_win_prob = float(y_proba[0][1])  # Class 1: home team wins
        
        # Determine confidence (adjusted for more realistic thresholds)
        max_confidence = max(home_win_prob, away_win_prob)
        if max_confidence > 0.75:
            confidence = "HIGH"
        elif max_confidence > 0.65:
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
    """Create prediction input from team IDs using game-level data"""
    try:
        # Get recent games for both teams from the game-level dataset
        home_team_games = games_df[
            (games_df['HOME_TEAM_ID'] == home_team_id) | (games_df['AWAY_TEAM_ID'] == home_team_id)
        ].tail(10)
        away_team_games = games_df[
            (games_df['HOME_TEAM_ID'] == away_team_id) | (games_df['AWAY_TEAM_ID'] == away_team_id)
        ].tail(10)
        
        if home_team_games.empty or away_team_games.empty:
            return None
        
        # Use the most recent game data
        home_latest = home_team_games.iloc[-1]
        away_latest = away_team_games.iloc[-1]
        
        # Get the correct stats for each team based on their role in their most recent game
        # For home team: use their stats from their most recent game
        if home_latest['HOME_TEAM_ID'] == home_team_id:
            # Home team was home in their most recent game
            home_pts = home_latest['HOME_PTS_rolling5']
            home_fg_pct = home_latest['HOME_FG_PCT_rolling5']
            home_fg3_pct = home_latest['HOME_FG3_PCT_rolling5']
            home_ft_pct = home_latest['HOME_FT_PCT_rolling5']
            home_reb = home_latest['HOME_REB_rolling5']
            home_ast = home_latest['HOME_AST_rolling5']
            home_tov = home_latest['HOME_TOV_rolling5']
            home_win_pct = home_latest['HOME_SEASON_WIN_PCT']
        else:
            # Home team was away in their most recent game
            home_pts = home_latest['AWAY_PTS_rolling5']
            home_fg_pct = home_latest['AWAY_FG_PCT_rolling5']
            home_fg3_pct = home_latest['AWAY_FG3_PCT_rolling5']
            home_ft_pct = home_latest['AWAY_FT_PCT_rolling5']
            home_reb = home_latest['AWAY_REB_rolling5']
            home_ast = home_latest['AWAY_AST_rolling5']
            home_tov = home_latest['AWAY_TOV_rolling5']
            home_win_pct = home_latest['AWAY_SEASON_WIN_PCT']
        
        # For away team: use their stats from their most recent game
        if away_latest['HOME_TEAM_ID'] == away_team_id:
            # Away team was home in their most recent game
            away_pts = away_latest['HOME_PTS_rolling5']
            away_fg_pct = away_latest['HOME_FG_PCT_rolling5']
            away_fg3_pct = away_latest['HOME_FG3_PCT_rolling5']
            away_ft_pct = away_latest['HOME_FT_PCT_rolling5']
            away_reb = away_latest['HOME_REB_rolling5']
            away_ast = away_latest['HOME_AST_rolling5']
            away_tov = away_latest['HOME_TOV_rolling5']
            away_win_pct = away_latest['HOME_SEASON_WIN_PCT']
        else:
            # Away team was away in their most recent game
            away_pts = away_latest['AWAY_PTS_rolling5']
            away_fg_pct = away_latest['AWAY_FG_PCT_rolling5']
            away_fg3_pct = away_latest['AWAY_FG3_PCT_rolling5']
            away_ft_pct = away_latest['AWAY_FT_PCT_rolling5']
            away_reb = away_latest['AWAY_REB_rolling5']
            away_ast = away_latest['AWAY_AST_rolling5']
            away_tov = away_latest['AWAY_TOV_rolling5']
            away_win_pct = away_latest['AWAY_SEASON_WIN_PCT']
        
        # Create input data for game-level prediction with advanced features
        input_data = {
            # Original features
            'HOME_PTS_rolling5': home_pts,
            'HOME_FG_PCT_rolling5': home_fg_pct,
            'HOME_FG3_PCT_rolling5': home_fg3_pct,
            'HOME_FT_PCT_rolling5': home_ft_pct,
            'HOME_REB_rolling5': home_reb,
            'HOME_AST_rolling5': home_ast,
            'HOME_TOV_rolling5': home_tov,
            'AWAY_PTS_rolling5': away_pts,
            'AWAY_FG_PCT_rolling5': away_fg_pct,
            'AWAY_FG3_PCT_rolling5': away_fg3_pct,
            'AWAY_FT_PCT_rolling5': away_ft_pct,
            'AWAY_REB_rolling5': away_reb,
            'AWAY_AST_rolling5': away_ast,
            'AWAY_TOV_rolling5': away_tov,
            'HOME_SEASON_WIN_PCT': home_win_pct,
            'AWAY_SEASON_WIN_PCT': away_win_pct,
            
            # Advanced difference features
            'WIN_PCT_DIFF': home_win_pct - away_win_pct,
            'WIN_PCT_RATIO': home_win_pct / (away_win_pct + 0.01),
            'STRENGTH_ADVANTAGE': 1 if (home_win_pct - away_win_pct) > 0.1 else (-1 if (home_win_pct - away_win_pct) < -0.1 else 0),
            'PTS_DIFF': home_pts - away_pts,
            'FG_PCT_DIFF': home_fg_pct - away_fg_pct,
            'FG3_PCT_DIFF': home_fg3_pct - away_fg3_pct,
            'FT_PCT_DIFF': home_ft_pct - away_ft_pct,
            'REB_DIFF': home_reb - away_reb,
            'AST_DIFF': home_ast - away_ast,
            'TOV_DIFF': away_tov - home_tov,
            
            # Efficiency and momentum features
            'HOME_EFFICIENCY': (home_pts * home_fg_pct * home_ast / (home_tov + 1)),
            'AWAY_EFFICIENCY': (away_pts * away_fg_pct * away_ast / (away_tov + 1)),
            'EFFICIENCY_DIFF': (home_pts * home_fg_pct * home_ast / (home_tov + 1)) - (away_pts * away_fg_pct * away_ast / (away_tov + 1)),
            'HOME_MOMENTUM': home_win_pct * home_pts / 100,
            'AWAY_MOMENTUM': away_win_pct * away_pts / 100,
            'MOMENTUM_DIFF': (home_win_pct * home_pts / 100) - (away_win_pct * away_pts / 100),
            'HOME_COURT_ADVANTAGE': (home_win_pct - away_win_pct) * 0.1,
            'STATS_DOMINANCE': sum([
                (home_pts - away_pts) > 5,
                (home_fg_pct - away_fg_pct) > 0.05,
                (home_reb - away_reb) > 2,
                (home_ast - away_ast) > 2,
                (away_tov - home_tov) > 1
            ]),
            'TIER_MATCHUP': int(pd.cut([home_win_pct], bins=3, labels=[1, 2, 3])[0]) - int(pd.cut([away_win_pct], bins=3, labels=[1, 2, 3])[0]),
            'HOME_RECENT_FORM': home_pts / (home_win_pct + 0.01),
            'AWAY_RECENT_FORM': away_pts / (away_win_pct + 0.01),
            'FORM_DIFF': (home_pts / (home_win_pct + 0.01)) - (away_pts / (away_win_pct + 0.01)),
            'CLUTCH_FACTOR': home_ft_pct - away_ft_pct
        }
        
        return input_data
        
    except Exception as e:
        print(f"Error creating prediction input: {e}")
        return None

@app.get("/models")
async def get_available_models():
    """Get list of available models and their status"""
    # Always return XGBoost as available since it can be trained on demand
    return {
        "available_models": ["xgb", "rf", "logreg"],  # All traditional models available
        "model_descriptions": {
            'xgb': 'XGBoost (Gradient Boosting) - Fast & Accurate',
            'rf': 'Random Forest - Robust & Interpretable',
            'logreg': 'Logistic Regression - Simple & Fast',
            'pytorch': 'PyTorch Neural Network - Advanced Deep Learning',
            'tensorflow': 'TensorFlow/Keras - Production-Ready Deep Learning',
            'ensemble': 'Ensemble (All Models) - Best Performance'
        },
        "status": "Traditional models available - will train on first prediction",
        "deep_learning_available": False,
        "recommended_model": "xgb"
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
    port = int(os.getenv("PORT", 8000))  # Render default port is 8000
    
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
