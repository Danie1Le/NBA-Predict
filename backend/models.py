"""
Pydantic models for NBA Game Predictor API
"""

from pydantic import BaseModel
from typing import List, Optional


class PredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    model_type: str = "ensemble"


class PredictionResponse(BaseModel):
    prediction: int
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_win_probability: float
    away_win_probability: float
    confidence: str
    model_used: str


class TeamInfo(BaseModel):
    id: int
    abbreviation: str
    name: str


class TeamStats(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str
    
    # Season stats
    season_wins: int
    season_games: int
    season_win_pct: float
    season_pts: float
    season_fg_pct: float
    season_fg3_pct: float
    season_ft_pct: float
    
    # Last 5 games
    last_5_wins: int
    last_5_games: int
    last_5_pts: float
    last_5_fg_pct: float
    last_5_fg3_pct: float
    last_5_ft_pct: float
    last_5_reb: float
    last_5_ast: float
    last_5_tov: float
    
    # Last 10 games
    last_10_wins: int
    last_10_games: int
