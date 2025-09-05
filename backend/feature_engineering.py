import pandas as pd
import numpy as np

def create_features(df):
    """
    Create advanced features for modeling from the game-level dataframe.
    Focuses on team strength differences and advanced statistical features.
    """
    # Rename columns to match expected feature names
    feature_mapping = {
        'HOME_PTS_rolling5': 'HOME_PTS_rolling5',
        'HOME_FG_PCT_rolling5': 'HOME_FG_PCT_rolling5', 
        'HOME_FG3_PCT_rolling5': 'HOME_FG3_PCT_rolling5',
        'HOME_FT_PCT_rolling5': 'HOME_FT_PCT_rolling5',
        'HOME_REB_rolling5': 'HOME_REB_rolling5',
        'HOME_AST_rolling5': 'HOME_AST_rolling5',
        'HOME_TOV_rolling5': 'HOME_TOV_rolling5',
        'AWAY_PTS_rolling5': 'AWAY_PTS_rolling5',
        'AWAY_FG_PCT_rolling5': 'AWAY_FG_PCT_rolling5',
        'AWAY_FG3_PCT_rolling5': 'AWAY_FG3_PCT_rolling5', 
        'AWAY_FT_PCT_rolling5': 'AWAY_FT_PCT_rolling5',
        'AWAY_REB_rolling5': 'AWAY_REB_rolling5',
        'AWAY_AST_rolling5': 'AWAY_AST_rolling5',
        'AWAY_TOV_rolling5': 'AWAY_TOV_rolling5',
        'HOME_TEAM_ID_WIN_PCT': 'HOME_SEASON_WIN_PCT',
        'AWAY_TEAM_ID_WIN_PCT': 'AWAY_SEASON_WIN_PCT'
    }
    
    # Rename columns
    df = df.rename(columns=feature_mapping)
    
    # === ADVANCED FEATURE ENGINEERING ===
    
    # 1. Team Strength Difference Features (most important)
    df['WIN_PCT_DIFF'] = df['HOME_SEASON_WIN_PCT'] - df['AWAY_SEASON_WIN_PCT']
    df['WIN_PCT_RATIO'] = df['HOME_SEASON_WIN_PCT'] / (df['AWAY_SEASON_WIN_PCT'] + 0.01)  # Avoid division by zero
    df['STRENGTH_ADVANTAGE'] = np.where(df['WIN_PCT_DIFF'] > 0.1, 1, 
                                       np.where(df['WIN_PCT_DIFF'] < -0.1, -1, 0))
    
    # 2. Offensive Efficiency Differences
    df['PTS_DIFF'] = df['HOME_PTS_rolling5'] - df['AWAY_PTS_rolling5']
    df['FG_PCT_DIFF'] = df['HOME_FG_PCT_rolling5'] - df['AWAY_FG_PCT_rolling5']
    df['FG3_PCT_DIFF'] = df['HOME_FG3_PCT_rolling5'] - df['AWAY_FG3_PCT_rolling5']
    df['FT_PCT_DIFF'] = df['HOME_FT_PCT_rolling5'] - df['AWAY_FT_PCT_rolling5']
    
    # 3. Defensive/Team Play Differences
    df['REB_DIFF'] = df['HOME_REB_rolling5'] - df['AWAY_REB_rolling5']
    df['AST_DIFF'] = df['HOME_AST_rolling5'] - df['AWAY_AST_rolling5']
    df['TOV_DIFF'] = df['AWAY_TOV_rolling5'] - df['HOME_TOV_rolling5']  # More turnovers = worse
    
    # 4. Combined Efficiency Metrics
    df['HOME_EFFICIENCY'] = (df['HOME_PTS_rolling5'] * df['HOME_FG_PCT_rolling5'] * 
                            df['HOME_AST_rolling5'] / (df['HOME_TOV_rolling5'] + 1))
    df['AWAY_EFFICIENCY'] = (df['AWAY_PTS_rolling5'] * df['AWAY_FG_PCT_rolling5'] * 
                            df['AWAY_AST_rolling5'] / (df['AWAY_TOV_rolling5'] + 1))
    df['EFFICIENCY_DIFF'] = df['HOME_EFFICIENCY'] - df['AWAY_EFFICIENCY']
    
    # 5. Momentum Features (recent form)
    df['HOME_MOMENTUM'] = df['HOME_SEASON_WIN_PCT'] * df['HOME_PTS_rolling5'] / 100
    df['AWAY_MOMENTUM'] = df['AWAY_SEASON_WIN_PCT'] * df['AWAY_PTS_rolling5'] / 100
    df['MOMENTUM_DIFF'] = df['HOME_MOMENTUM'] - df['AWAY_MOMENTUM']
    
    # 6. Home Court Advantage Multiplier
    df['HOME_COURT_ADVANTAGE'] = df['WIN_PCT_DIFF'] * 0.1  # Home teams get slight boost
    
    # 7. Statistical Dominance Features
    df['STATS_DOMINANCE'] = (
        (df['PTS_DIFF'] > 5).astype(int) +
        (df['FG_PCT_DIFF'] > 0.05).astype(int) +
        (df['REB_DIFF'] > 2).astype(int) +
        (df['AST_DIFF'] > 2).astype(int) +
        (df['TOV_DIFF'] > 1).astype(int)
    )
    
    # 8. Team Quality Tiers
    df['HOME_TIER'] = pd.cut(df['HOME_SEASON_WIN_PCT'], bins=3, labels=[1, 2, 3])
    df['AWAY_TIER'] = pd.cut(df['AWAY_SEASON_WIN_PCT'], bins=3, labels=[1, 2, 3])
    df['TIER_MATCHUP'] = df['HOME_TIER'].astype(int) - df['AWAY_TIER'].astype(int)
    
    # 9. Recent Performance Trends (last 3 games vs season average)
    # This would require more historical data, but we can simulate with current data
    df['HOME_RECENT_FORM'] = df['HOME_PTS_rolling5'] / df['HOME_SEASON_WIN_PCT'].replace(0, 0.01)
    df['AWAY_RECENT_FORM'] = df['AWAY_PTS_rolling5'] / df['AWAY_SEASON_WIN_PCT'].replace(0, 0.01)
    df['FORM_DIFF'] = df['HOME_RECENT_FORM'] - df['AWAY_RECENT_FORM']
    
    # 10. Clutch Performance (close games)
    df['CLUTCH_FACTOR'] = df['HOME_FT_PCT_rolling5'] - df['AWAY_FT_PCT_rolling5']
    
    return df