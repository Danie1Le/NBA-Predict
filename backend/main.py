#!/usr/bin/env python3
"""
FastAPI backend for NBA Game Predictor
Always running with pre-loaded models
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class GameStats(BaseModel):
    team_id: int
    team_name: str
    recent_stats: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global model_cache, games_df, teams_df, team_map, features
    
    print("🚀 Starting NBA Game Predictor API...")
    print("📊 Loading data and models...")
    
    try:
        # Load teams data
        teams_df = pd.read_csv('Data/NBA_TEAMS.csv')
        team_map = dict(zip(teams_df['id'], teams_df['abbreviation']))
        
        # Load and process games data
        games_df = load_and_clean_data('Data/NBA_GAMES.csv')
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
        
        # Load model cache
        X = games_df[features].fillna(0)
        y = (games_df['WL'] == 'W').astype(int)
        
        model_cache = ModelCache()
        if not model_cache.load_models():
            print("❌ Failed to load cached models")
            return
        
        print("✅ All models loaded successfully!")
        print(f"📈 Available models: {model_cache.get_available_models()}")
        print("🎯 API ready for predictions!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NBA Game Predictor API",
        "status": "running",
        "models_loaded": model_cache is not None and model_cache.is_trained
    }

@app.get("/teams", response_model=List[TeamInfo])
async def get_teams():
    """Get list of all NBA teams"""
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

@app.get("/teams/{team_id}/stats", response_model=GameStats)
async def get_team_stats(team_id: int):
    """Get recent statistics for a team"""
    if games_df is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    # Get recent games for the team
    team_games = games_df[games_df['Team_ID'] == team_id].tail(5)
    
    if team_games.empty:
        raise HTTPException(status_code=404, detail="No games found for team")
    
    # Calculate recent averages
    recent_stats = {
        'points': float(team_games['PTS'].mean()),
        'field_goal_pct': float(team_games['FG_PCT'].mean()),
        'three_point_pct': float(team_games['FG3_PCT'].mean()),
        'free_throw_pct': float(team_games['FT_PCT'].mean()),
        'rebounds': float(team_games['REB'].mean()),
        'assists': float(team_games['AST'].mean()),
        'turnovers': float(team_games['TOV'].mean()),
        'win_percentage': float(team_games['SEASON_WIN_PCT'].iloc[-1])
    }
    
    team_name = team_map.get(team_id, f"Team {team_id}")
    
    return GameStats(
        team_id=team_id,
        team_name=team_name,
        recent_stats=recent_stats
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """Predict the outcome of a game"""
    if model_cache is None or not model_cache.is_trained:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if games_df is None:
        raise HTTPException(status_code=500, detail="Games data not loaded")
    
    try:
        # Create prediction input
        input_data = create_prediction_input(
            request.home_team_id, 
            request.away_team_id, 
            games_df, 
            team_map
        )
        
        if input_data is None:
            raise HTTPException(status_code=400, detail="Could not create prediction input")
        
        # Make prediction
        X_input = pd.DataFrame([input_data])[features]
        y_pred, y_proba = model_cache.predict(request.model_type, X_input)
        
        # Convert to proper formats
        if request.model_type in ['pytorch', 'tensorflow', 'ensemble']:
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
            model_used=request.model_type
        )
        
    except Exception as e:
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
    """Get list of available models"""
    if model_cache is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "available_models": model_cache.get_available_models(),
        "model_descriptions": {
            'xgb': 'XGBoost (Gradient Boosting)',
            'rf': 'Random Forest',
            'logreg': 'Logistic Regression',
            'pytorch': 'PyTorch Neural Network',
            'tensorflow': 'TensorFlow/Keras',
            'ensemble': 'Ensemble (All Models)'
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
