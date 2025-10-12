"""
Prediction service for NBA Game Predictor - Fixed version
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from train_model import train_model


class PredictionService:
    """Handles all prediction-related logic"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def create_prediction_input(self, home_team_id: int, away_team_id: int) -> Optional[Dict]:
        """Create prediction input from team IDs using existing game data"""
        try:
            games_df = self.data_loader.games_df
            team_map = self.data_loader.team_map
            
            if games_df is None or team_map is None:
                return None
            
            # Find a recent game between these teams or use their most recent games
            # First try to find a game between these teams
            matchup_games = games_df[
                ((games_df['HOME_TEAM_ID'] == home_team_id) & (games_df['AWAY_TEAM_ID'] == away_team_id)) |
                ((games_df['HOME_TEAM_ID'] == away_team_id) & (games_df['AWAY_TEAM_ID'] == home_team_id))
            ].sort_values('GAME_DATE_REAL', ascending=False)
            
            if len(matchup_games) > 0:
                # Use the most recent matchup
                game_data = matchup_games.iloc[0]
                # Ensure home team is actually the home team
                if game_data['HOME_TEAM_ID'] == home_team_id:
                    # Home team is home, away team is away
                    input_data = self._extract_features_from_game(game_data, home_team_id, away_team_id)
                else:
                    # Teams are swapped, need to flip the data
                    input_data = self._extract_features_from_game(game_data, away_team_id, home_team_id)
                    # Swap home and away features
                    input_data = self._swap_home_away_features(input_data)
            else:
                # No direct matchup, use most recent games for each team
                home_games = games_df[games_df['HOME_TEAM_ID'] == home_team_id].sort_values('GAME_DATE_REAL', ascending=False)
                away_games = games_df[games_df['AWAY_TEAM_ID'] == away_team_id].sort_values('GAME_DATE_REAL', ascending=False)
                
                if len(home_games) == 0 or len(away_games) == 0:
                    print(f"Warning: No recent games found for teams {home_team_id} or {away_team_id}")
                    return None
                
                # Use most recent game for each team
                home_latest = home_games.iloc[0]
                away_latest = away_games.iloc[0]
                
                # Create synthetic game data
                input_data = self._create_synthetic_game_features(home_latest, away_latest, home_team_id, away_team_id)
            
            return input_data
            
        except Exception as e:
            print(f"Error creating prediction input: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_features_from_game(self, game_data, home_team_id, away_team_id):
        """Extract features from an actual game record"""
        # Get all features that exist in the games_df
        features = {}
        for feature in self.data_loader.features:
            if feature in game_data.index:
                features[feature] = game_data[feature]
            else:
                # Set missing features to 0
                features[feature] = 0.0
        
        return features
    
    def _swap_home_away_features(self, input_data):
        """Swap home and away features in the input data"""
        # Features that should NOT be swapped (they are calculated differently)
        no_swap_features = {'HOME_COURT_ADVANTAGE', 'HOME_ADVANTAGE_RATIO', 'AWAY_DISADVANTAGE_RATIO'}
        
        swapped = {}
        for key, value in input_data.items():
            # Skip features that should not be swapped
            if key in no_swap_features:
                swapped[key] = value
            elif key.startswith('HOME_'):
                new_key = key.replace('HOME_', 'AWAY_')
                swapped[new_key] = value
            elif key.startswith('AWAY_'):
                new_key = key.replace('AWAY_', 'HOME_')
                swapped[new_key] = value
            else:
                # For difference features, negate the value
                if 'DIFF' in key or 'RATIO' in key:
                    swapped[key] = -value
                else:
                    swapped[key] = value
        
        return swapped
    
    def _create_synthetic_game_features(self, home_game, away_game, home_team_id, away_team_id):
        """Create synthetic game features from individual team games"""
        # This is a simplified approach - just use the features that exist
        features = {}
        
        # Get basic stats for home team
        if home_game['HOME_TEAM_ID'] == home_team_id:
            home_pts = home_game['HOME_PTS_rolling5']
            home_fg_pct = home_game['HOME_FG_PCT_rolling5']
            home_fg3_pct = home_game['HOME_FG3_PCT_rolling5']
            home_ft_pct = home_game['HOME_FT_PCT_rolling5']
            home_reb = home_game['HOME_REB_rolling5']
            home_ast = home_game['HOME_AST_rolling5']
            home_tov = home_game['HOME_TOV_rolling5']
            home_win_pct = home_game['HOME_TEAM_ID_WIN_PCT']
        else:
            home_pts = home_game['AWAY_PTS_rolling5']
            home_fg_pct = home_game['AWAY_FG_PCT_rolling5']
            home_fg3_pct = home_game['AWAY_FG3_PCT_rolling5']
            home_ft_pct = home_game['AWAY_FT_PCT_rolling5']
            home_reb = home_game['AWAY_REB_rolling5']
            home_ast = home_game['AWAY_AST_rolling5']
            home_tov = home_game['AWAY_TOV_rolling5']
            home_win_pct = home_game['AWAY_TEAM_ID_WIN_PCT']
        
        # Get basic stats for away team
        if away_game['HOME_TEAM_ID'] == away_team_id:
            away_pts = away_game['HOME_PTS_rolling5']
            away_fg_pct = away_game['HOME_FG_PCT_rolling5']
            away_fg3_pct = away_game['HOME_FG3_PCT_rolling5']
            away_ft_pct = away_game['HOME_FT_PCT_rolling5']
            away_reb = away_game['HOME_REB_rolling5']
            away_ast = away_game['HOME_AST_rolling5']
            away_tov = away_game['HOME_TOV_rolling5']
            away_win_pct = away_game['HOME_TEAM_ID_WIN_PCT']
        else:
            away_pts = away_game['AWAY_PTS_rolling5']
            away_fg_pct = away_game['AWAY_FG_PCT_rolling5']
            away_fg3_pct = away_game['AWAY_FG3_PCT_rolling5']
            away_ft_pct = away_game['AWAY_FT_PCT_rolling5']
            away_reb = away_game['AWAY_REB_rolling5']
            away_ast = away_game['AWAY_AST_rolling5']
            away_tov = away_game['AWAY_TOV_rolling5']
            away_win_pct = away_game['AWAY_TEAM_ID_WIN_PCT']
        
        # Create a minimal feature set with only basic features
        features = {
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
            'HOME_TEAM_ID_WIN_PCT': home_win_pct,
            'AWAY_TEAM_ID_WIN_PCT': away_win_pct,
        }
        
        # Add basic calculated features
        features.update({
            'WIN_PCT_DIFF': home_win_pct - away_win_pct,
            'WIN_PCT_RATIO': home_win_pct / (away_win_pct + 0.01),
            'PTS_DIFF': home_pts - away_pts,
            'FG_PCT_DIFF': home_fg_pct - away_fg_pct,
            'FG3_PCT_DIFF': home_fg3_pct - away_fg3_pct,
            'FT_PCT_DIFF': home_ft_pct - away_ft_pct,
            'REB_DIFF': home_reb - away_reb,
            'AST_DIFF': home_ast - away_ast,
            'TOV_DIFF': away_tov - home_tov,
        })
        
        # Add missing features that exist in games_df
        features.update({
            'HOME_COURT_ADVANTAGE': 0.05,  # Standard home court advantage
            'HOME_ADVANTAGE_RATIO': 0.05 * home_win_pct * home_pts / 100,
            'AWAY_DISADVANTAGE_RATIO': 0.8 * away_win_pct * away_pts / 100,
        })
        
        # Set all other features to 0 (they'll be ignored by the model)
        for feature in self.data_loader.features:
            if feature not in features:
                features[feature] = 0.0
        
        # Ensure all required features are present
        required_features = ['HOME_COURT_ADVANTAGE', 'HOME_ADVANTAGE_RATIO', 'AWAY_DISADVANTAGE_RATIO']
        for feature in required_features:
            if feature not in features:
                features[feature] = 0.0
        
        return features
    
    async def train_models_if_needed(self) -> bool:
        """Train models if they don't exist"""
        try:
            if self.data_loader.model_cache is None:
                return False
            
            # Check if models are already trained
            available_models = self.data_loader.model_cache.get_available_models()
            if len(available_models) > 0:
                return True
            
            print("🚀 Training models on prediction request...")
            
            # Get the training data
            games_df = self.data_loader.games_df
            features = self.data_loader.features
            
            if games_df is None or features is None:
                return False
            
            # Prepare training data
            X = games_df[features].fillna(0)
            y = games_df['HOME_WON']
            
            # Train models
            success = self.data_loader.model_cache.train_all_models(X, y)
            
            if success:
                print("✅ Models trained successfully!")
                return True
            else:
                print("❌ Model training failed")
                return False
                
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def make_prediction(self, home_team_id: int, away_team_id: int, model_type: str = "ensemble"):
        """Make a prediction for a game"""
        try:
            # Create prediction input
            input_data = self.create_prediction_input(home_team_id, away_team_id)
            
            if input_data is None:
                return None
            
            # Train models if needed
            if not await self.train_models_if_needed():
                return None
            
            # Use the fastest available model if requested model not available
            available_models = self.data_loader.model_cache.get_available_models()
            model_to_use = model_type
            
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
                    return None
                print(f"⚠️ Requested model '{model_type}' not available, using '{model_to_use}'")
            
            # Make prediction (optimized)
            X_input = pd.DataFrame([input_data])[self.data_loader.features]
            
            # Ensure X_input is properly formatted for the model
            X_input = X_input.fillna(0)  # Fill any NaN values
            
            # Convert to numpy array if needed for traditional ML models
            if model_to_use in ['xgb', 'rf', 'logreg']:
                X_input = X_input.values
            
            y_pred, y_proba = self.data_loader.model_cache.predict(model_to_use, X_input)
            
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
            confidence = "High" if abs(home_win_prob - away_win_prob) > 0.3 else "Medium" if abs(home_win_prob - away_win_prob) > 0.15 else "Low"
            
            # Get team names
            home_team_name = self.data_loader.team_map.get(home_team_id, f"Team {home_team_id}")
            away_team_name = self.data_loader.team_map.get(away_team_id, f"Team {away_team_id}")
            
            return {
                "prediction": prediction,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "home_win_probability": home_win_prob,
                "away_win_probability": away_win_prob,
                "confidence": confidence,
                "model_used": model_to_use
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
