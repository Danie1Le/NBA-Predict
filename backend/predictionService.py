"""
Prediction service for NBA Game Predictor - Fixed version
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from featureEngineering import matchup_features


class PredictionService:
    """Handles all prediction-related logic"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def create_prediction_input(self, home_team_id: int, away_team_id: int) -> Optional[Dict]:
        """
        Build one feature row for a matchup that hasn't been played.

        Both teams' current form and ratings come from data_loader.team_state,
        which is each team's state after their most recent game. An earlier
        version replayed the teams' last game row instead, which meant every
        prediction was built from stats that stopped one game short.
        """
        try:
            state = self.data_loader.team_state
            features = self.data_loader.features
            if state is None or features is None:
                return None

            missing = [t for t in (home_team_id, away_team_id) if t not in state.index]
            if missing:
                print(f"Warning: no games on record for team(s) {missing}")
                return None

            row = matchup_features(state.loc[home_team_id], state.loc[away_team_id])
            return {name: float(row.get(name, 0.0)) for name in features}

        except Exception as e:
            print(f"Error creating prediction input: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            
            # Prepare training data. Gaps stay as NaN: each model's pipeline
            # imputes them from the training fold, which fillna(0) would not do.
            X = games_df[features]
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
