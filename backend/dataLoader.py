"""
Data loading and management for NBA Game Predictor
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from modelCache import ModelCache
from preprocessing import load_and_clean_data
from featureEngineering import create_features


class DataLoader:
    """Handles all data loading and model management"""
    
    def __init__(self):
        self.games_df: Optional[pd.DataFrame] = None
        self.teams_df: Optional[pd.DataFrame] = None
        self.team_map: Optional[Dict] = None
        self.features: Optional[List[str]] = None
        self.model_cache: Optional[ModelCache] = None
    
    def load_teams_data(self, data_dir: str = 'Data') -> Tuple[pd.DataFrame, Dict]:
        """Load teams data and create team mapping"""
        teams_df = pd.read_csv(f'{data_dir}/NBA_TEAMS.csv')
        team_map = dict(zip(teams_df['id'], teams_df['abbreviation']))
        return teams_df, team_map
    
    def load_games_data(self, data_dir: str = 'Data') -> pd.DataFrame:
        """Load and process games data"""
        games_df = load_and_clean_data(f'{data_dir}/NBA_GAMES.csv')
        games_df = create_features(games_df)
        return games_df
    
    def get_features_list(self) -> List[str]:
        """Get the list of features used for predictions"""
        return [
            # Core team stats
            'HOME_PTS_rolling5', 'HOME_FG_PCT_rolling5', 'HOME_FG3_PCT_rolling5', 'HOME_FT_PCT_rolling5',
            'HOME_REB_rolling5', 'HOME_AST_rolling5', 'HOME_TOV_rolling5',
            'AWAY_PTS_rolling5', 'AWAY_FG_PCT_rolling5', 'AWAY_FG3_PCT_rolling5', 'AWAY_FT_PCT_rolling5',
            'AWAY_REB_rolling5', 'AWAY_AST_rolling5', 'AWAY_TOV_rolling5',
            'HOME_TEAM_ID_WIN_PCT', 'AWAY_TEAM_ID_WIN_PCT',
            
            # Key difference features
            'WIN_PCT_DIFF', 'WIN_PCT_RATIO', 'STRENGTH_ADVANTAGE',
            'PTS_DIFF', 'FG_PCT_DIFF', 'FG3_PCT_DIFF', 'FT_PCT_DIFF',
            'REB_DIFF', 'AST_DIFF', 'TOV_DIFF',
            
            # Performance metrics
            'HOME_EFFICIENCY', 'AWAY_EFFICIENCY', 'EFFICIENCY_DIFF',
            'HOME_MOMENTUM', 'AWAY_MOMENTUM', 'MOMENTUM_DIFF',
            'HOME_COURT_ADVANTAGE', 'STATS_DOMINANCE', 'TIER_MATCHUP',
            'HOME_RECENT_FORM', 'AWAY_RECENT_FORM', 'FORM_DIFF', 'CLUTCH_FACTOR',
            
            # Advanced features
            'H2H_ADVANTAGE', 'HOME_DEF_EFFICIENCY', 'AWAY_DEF_EFFICIENCY', 'DEF_EFFICIENCY_DIFF',
            'HOME_PACE', 'AWAY_PACE', 'PACE_DIFF', 'THREE_POINT_ADVANTAGE',
            'TURNOVER_MARGIN', 'REBOUNDING_DOMINANCE', 'FT_ADVANTAGE',
            'HOME_CONSISTENCY', 'AWAY_CONSISTENCY', 'CONSISTENCY_DIFF',
            
            # Composite features
            'HOME_STRENGTH_SCORE', 'AWAY_STRENGTH_SCORE', 'STRENGTH_SCORE_DIFF',
            'HOME_MOMENTUM_COMPOSITE', 'AWAY_MOMENTUM_COMPOSITE', 'MOMENTUM_COMPOSITE_DIFF',
            'HOME_CLUTCH_COMPOSITE', 'AWAY_CLUTCH_COMPOSITE', 'CLUTCH_COMPOSITE_DIFF',
            'HOME_PRESSURE', 'AWAY_PRESSURE', 'PRESSURE_DIFF',
            'FINAL_COMPOSITE_SCORE',
            'HOME_VARIANCE', 'AWAY_VARIANCE', 'VARIANCE_DIFF',
            'WIN_PCT_EFFICIENCY_INTERACTION', 'MOMENTUM_CLUTCH_INTERACTION', 'STRENGTH_PRESSURE_INTERACTION',
            'HOME_ADVANTAGE_RATIO', 'AWAY_DISADVANTAGE_RATIO', 'ADVANTAGE_DISADVANTAGE_DIFF',
            'HOME_TREND_ACCELERATION', 'AWAY_TREND_ACCELERATION', 'TREND_ACCELERATION_DIFF',
            'HOME_OVERALL_STRENGTH', 'AWAY_OVERALL_STRENGTH', 'OVERALL_STRENGTH_DIFF',
            'HOME_WIN_LIKELIHOOD', 'AWAY_WIN_LIKELIHOOD', 'LIKELIHOOD_DIFF',
            'HOME_CONSISTENCY_SCORE', 'AWAY_CONSISTENCY_SCORE',
            'AUC_OPTIMIZED_SCORE'
        ]
    
    async def load_all_data(self, data_dir: str = 'Data', cache_dir: str = 'model_cache') -> bool:
        """Load all data and models"""
        try:
            print("🚀 Loading NBA data and models...")
            
            # Load teams data
            self.teams_df, self.team_map = self.load_teams_data(data_dir)
            
            # Load and process games data
            self.games_df = self.load_games_data(data_dir)
            self.features = self.get_features_list()
            
            # Load models
            await self.load_models(cache_dir)
            
            print("✅ Data loaded successfully!")
            print(f"📊 Games: {len(self.games_df) if self.games_df is not None else 0}")
            print(f"🏀 Teams: {len(self.teams_df) if self.teams_df is not None else 0}")
            print(f"🧠 Models: {len(self.model_cache.models) if self.model_cache else 0}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def load_models(self, cache_dir: str = 'model_cache') -> bool:
        """Load or initialize model cache"""
        try:
            self.model_cache = ModelCache(cache_dir=cache_dir)
            
            # Try to load cached models
            if self.model_cache.load_models(filename="cached_models.pkl"):
                print("✅ All pre-trained models loaded successfully!")
                print(f"📈 Available models: {self.model_cache.get_available_models()}")
                return True
            elif self.model_cache.load_models(filename="traditional_models.pkl"):
                print("✅ Traditional ML models loaded successfully!")
                print(f"📈 Available models: {self.model_cache.get_available_models()}")
                return True
            else:
                print("❌ No cached models found - will train on first request")
                # Keep model_cache initialized for training
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model_cache = None
            return False
    
    def get_team_stats(self, team_id: int) -> Optional[dict]:
        """Get detailed stats for a specific team"""
        if self.games_df is None or self.team_map is None:
            return None
        
        try:
            # Get team games - team can be either home or away
            home_games = self.games_df[self.games_df['HOME_TEAM_ID'] == team_id].sort_values('GAME_DATE_REAL', ascending=False)
            away_games = self.games_df[self.games_df['AWAY_TEAM_ID'] == team_id].sort_values('GAME_DATE_REAL', ascending=False)
            
            if len(home_games) == 0 and len(away_games) == 0:
                return None
            
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
            
            return {
                'team_id': team_id,
                'team_name': self.team_map.get(team_id, f"Team {team_id}"),
                'abbreviation': self.team_map.get(team_id, f"T{team_id}"),
                'season_wins': season_stats['wins'],
                'season_games': season_stats['games'],
                'season_win_pct': season_stats['win_pct'],
                'season_pts': season_stats['PTS'],
                'season_fg_pct': season_stats['FG_PCT'],
                'season_fg3_pct': season_stats['FG3_PCT'],
                'season_ft_pct': season_stats['FT_PCT'],
                'last_5_wins': last_5_stats['wins'],
                'last_5_games': last_5_stats['games'],
                'last_5_pts': last_5_stats['PTS'],
                'last_5_fg_pct': last_5_stats['FG_PCT'],
                'last_5_fg3_pct': last_5_stats['FG3_PCT'],
                'last_5_ft_pct': last_5_stats['FT_PCT'],
                'last_5_reb': last_5_stats['REB'],
                'last_5_ast': last_5_stats['AST'],
                'last_5_tov': last_5_stats['TOV'],
                'last_10_wins': last_10_stats['wins'],
                'last_10_games': last_10_stats['games']
            }
            
        except Exception as e:
            print(f"Error getting team stats: {e}")
            return None
