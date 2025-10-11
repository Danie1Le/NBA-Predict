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
    # Clip extreme values to prevent logistic regression issues
    df['HOME_RECENT_FORM'] = np.clip(df['HOME_PTS_rolling5'] / df['HOME_SEASON_WIN_PCT'].replace(0, 0.01), 0, 1000)
    df['AWAY_RECENT_FORM'] = np.clip(df['AWAY_PTS_rolling5'] / df['AWAY_SEASON_WIN_PCT'].replace(0, 0.01), 0, 1000)
    df['FORM_DIFF'] = df['HOME_RECENT_FORM'] - df['AWAY_RECENT_FORM']
    
    # 10. Clutch Performance (close games)
    df['CLUTCH_FACTOR'] = df['HOME_FT_PCT_rolling5'] - df['AWAY_FT_PCT_rolling5']
    
    # === ADVANCED FEATURES FOR IMPROVED ACCURACY ===
    
    # 11. Head-to-Head Performance (based on team strength)
    # Stronger teams tend to win more head-to-head matchups
    df['H2H_ADVANTAGE'] = df['WIN_PCT_DIFF'] * 0.3
    
    # 13. Defensive Efficiency
    df['HOME_DEF_EFFICIENCY'] = (df['HOME_REB_rolling5'] + df['HOME_TOV_rolling5']) / (df['HOME_PTS_rolling5'] + 1)
    df['AWAY_DEF_EFFICIENCY'] = (df['AWAY_REB_rolling5'] + df['AWAY_TOV_rolling5']) / (df['AWAY_PTS_rolling5'] + 1)
    df['DEF_EFFICIENCY_DIFF'] = df['HOME_DEF_EFFICIENCY'] - df['AWAY_DEF_EFFICIENCY']
    
    # 14. Pace and Style Features
    df['HOME_PACE'] = df['HOME_PTS_rolling5'] + df['HOME_AST_rolling5'] + df['HOME_REB_rolling5']
    df['AWAY_PACE'] = df['AWAY_PTS_rolling5'] + df['AWAY_AST_rolling5'] + df['AWAY_REB_rolling5']
    df['PACE_DIFF'] = df['HOME_PACE'] - df['AWAY_PACE']
    
    # 15. Three-Point Shooting Advantage
    df['THREE_POINT_ADVANTAGE'] = (df['HOME_FG3_PCT_rolling5'] - df['AWAY_FG3_PCT_rolling5']) * 2  # Weight 3PT more
    
    # 16. Turnover Margin (negative turnovers are good)
    df['TURNOVER_MARGIN'] = df['AWAY_TOV_rolling5'] - df['HOME_TOV_rolling5']
    
    # 17. Rebounding Dominance
    df['REBOUNDING_DOMINANCE'] = (df['HOME_REB_rolling5'] - df['AWAY_REB_rolling5']) / (df['HOME_REB_rolling5'] + df['AWAY_REB_rolling5'] + 1)
    
    # 18. Free Throw Advantage (clutch factor)
    df['FT_ADVANTAGE'] = (df['HOME_FT_PCT_rolling5'] - df['AWAY_FT_PCT_rolling5']) * 1.5
    
    # 19. Consistency Metrics (based on rolling averages)
    df['HOME_CONSISTENCY'] = 1 / (df['HOME_PTS_rolling5'].std() + 1) if len(df) > 1 else 1
    df['AWAY_CONSISTENCY'] = 1 / (df['AWAY_PTS_rolling5'].std() + 1) if len(df) > 1 else 1
    df['CONSISTENCY_DIFF'] = df['HOME_CONSISTENCY'] - df['AWAY_CONSISTENCY']
    
    # 20. Momentum Indicators (recent form vs season average)
    df['HOME_MOMENTUM_INDICATOR'] = df['HOME_PTS_rolling5'] / (df['HOME_SEASON_WIN_PCT'] * 100 + 1)
    df['AWAY_MOMENTUM_INDICATOR'] = df['AWAY_PTS_rolling5'] / (df['AWAY_SEASON_WIN_PCT'] * 100 + 1)
    df['MOMENTUM_INDICATOR_DIFF'] = df['HOME_MOMENTUM_INDICATOR'] - df['AWAY_MOMENTUM_INDICATOR']
    
    # 21. Composite Strength Score
    df['HOME_STRENGTH_SCORE'] = (
        df['HOME_SEASON_WIN_PCT'] * 0.4 +
        df['HOME_PTS_rolling5'] / 120 * 0.3 +
        df['HOME_FG_PCT_rolling5'] * 0.2 +
        df['HOME_REB_rolling5'] / 50 * 0.1
    )
    df['AWAY_STRENGTH_SCORE'] = (
        df['AWAY_SEASON_WIN_PCT'] * 0.4 +
        df['AWAY_PTS_rolling5'] / 120 * 0.3 +
        df['AWAY_FG_PCT_rolling5'] * 0.2 +
        df['AWAY_REB_rolling5'] / 50 * 0.1
    )
    df['STRENGTH_SCORE_DIFF'] = df['HOME_STRENGTH_SCORE'] - df['AWAY_STRENGTH_SCORE']
    
    # === ULTRA-ADVANCED FEATURES FOR 78%+ ACCURACY ===
    
    # 22. Key Feature Interactions
    df['WIN_PCT_INTERACTION'] = df['HOME_SEASON_WIN_PCT'] * df['AWAY_SEASON_WIN_PCT']
    df['PTS_PRODUCT'] = df['HOME_PTS_rolling5'] * df['AWAY_PTS_rolling5']
    df['PTS_RATIO'] = df['HOME_PTS_rolling5'] / (df['AWAY_PTS_rolling5'] + 1)
    
    # 23. Polynomial Features
    df['WIN_PCT_DIFF_SQUARED'] = df['WIN_PCT_DIFF'] ** 2
    df['WIN_PCT_DIFF_CUBED'] = df['WIN_PCT_DIFF'] ** 3
    df['PTS_DIFF_SQUARED'] = df['PTS_DIFF'] ** 2
    df['PTS_DIFF_ABS'] = np.abs(df['PTS_DIFF'])
    
    # 24. Key Statistical Features
    df['STRENGTH_PRODUCT'] = df['HOME_STRENGTH_SCORE'] * df['AWAY_STRENGTH_SCORE']
    df['MOMENTUM_PRODUCT'] = df['HOME_MOMENTUM'] * df['AWAY_MOMENTUM']
    df['MOMENTUM_RATIO'] = df['HOME_MOMENTUM'] / (df['AWAY_MOMENTUM'] + 0.01)
    
    
    # 28. Advanced Efficiency Metrics
    df['HOME_OFFENSIVE_EFFICIENCY'] = df['HOME_PTS_rolling5'] / (df['HOME_FGA_rolling5'] + 1) if 'HOME_FGA_rolling5' in df.columns else df['HOME_PTS_rolling5'] / 80
    df['AWAY_OFFENSIVE_EFFICIENCY'] = df['AWAY_PTS_rolling5'] / (df['AWAY_FGA_rolling5'] + 1) if 'AWAY_FGA_rolling5' in df.columns else df['AWAY_PTS_rolling5'] / 80
    df['OFFENSIVE_EFFICIENCY_DIFF'] = df['HOME_OFFENSIVE_EFFICIENCY'] - df['AWAY_OFFENSIVE_EFFICIENCY']
    
    # 29. Clutch Performance Indicators
    df['HOME_CLUTCH_FACTOR'] = df['HOME_FT_PCT_rolling5'] * df['HOME_FG3_PCT_rolling5']
    df['AWAY_CLUTCH_FACTOR'] = df['AWAY_FT_PCT_rolling5'] * df['AWAY_FG3_PCT_rolling5']
    df['CLUTCH_DIFF'] = df['HOME_CLUTCH_FACTOR'] - df['AWAY_CLUTCH_FACTOR']
    
    # 30. Momentum Shift Indicators
    df['HOME_MOMENTUM_SHIFT'] = df['HOME_PTS_rolling5'] * df['HOME_SEASON_WIN_PCT'] * df['HOME_FG_PCT_rolling5']
    df['AWAY_MOMENTUM_SHIFT'] = df['AWAY_PTS_rolling5'] * df['AWAY_SEASON_WIN_PCT'] * df['AWAY_FG_PCT_rolling5']
    df['MOMENTUM_SHIFT_DIFF'] = df['HOME_MOMENTUM_SHIFT'] - df['AWAY_MOMENTUM_SHIFT']
    
    # === FINAL PUSH FEATURES FOR 78%+ ACCURACY ===
    
    # 31. Advanced Synthetic Features
    df['SYNTHETIC_STRENGTH'] = (
        df['HOME_STRENGTH_SCORE'] * df['HOME_MOMENTUM'] * df['HOME_COURT_ADVANTAGE'] -
        df['AWAY_STRENGTH_SCORE'] * df['AWAY_MOMENTUM'] * 0.95  # Away penalty
    )
    
    # 32. Composite Performance Index
    df['HOME_PERFORMANCE_INDEX'] = (
        df['HOME_PTS_rolling5'] * 0.3 +
        df['HOME_FG_PCT_rolling5'] * 100 * 0.25 +
        df['HOME_REB_rolling5'] * 0.2 +
        df['HOME_AST_rolling5'] * 0.15 +
        df['HOME_SEASON_WIN_PCT'] * 100 * 0.1
    )
    df['AWAY_PERFORMANCE_INDEX'] = (
        df['AWAY_PTS_rolling5'] * 0.3 +
        df['AWAY_FG_PCT_rolling5'] * 100 * 0.25 +
        df['AWAY_REB_rolling5'] * 0.2 +
        df['AWAY_AST_rolling5'] * 0.15 +
        df['AWAY_SEASON_WIN_PCT'] * 100 * 0.1
    )
    df['PERFORMANCE_INDEX_DIFF'] = df['HOME_PERFORMANCE_INDEX'] - df['AWAY_PERFORMANCE_INDEX']
    
    # 33. Advanced Momentum Metrics
    df['HOME_MOMENTUM_COMPOSITE'] = (
        df['HOME_MOMENTUM'] * 0.4 +
        df['HOME_RECENT_FORM'] * 0.3 +
        df['HOME_MOMENTUM_INDICATOR'] * 0.3
    )
    df['AWAY_MOMENTUM_COMPOSITE'] = (
        df['AWAY_MOMENTUM'] * 0.4 +
        df['AWAY_RECENT_FORM'] * 0.3 +
        df['AWAY_MOMENTUM_INDICATOR'] * 0.3
    )
    df['MOMENTUM_COMPOSITE_DIFF'] = df['HOME_MOMENTUM_COMPOSITE'] - df['AWAY_MOMENTUM_COMPOSITE']
    
    # 34. Clutch Performance Composite
    df['HOME_CLUTCH_COMPOSITE'] = (
        df['HOME_FT_PCT_rolling5'] * 0.4 +
        df['HOME_FG3_PCT_rolling5'] * 0.3 +
        df['HOME_CLUTCH_FACTOR'] * 0.3
    )
    df['AWAY_CLUTCH_COMPOSITE'] = (
        df['AWAY_FT_PCT_rolling5'] * 0.4 +
        df['AWAY_FG3_PCT_rolling5'] * 0.3 +
        df['AWAY_CLUTCH_FACTOR'] * 0.3
    )
    df['CLUTCH_COMPOSITE_DIFF'] = df['HOME_CLUTCH_COMPOSITE'] - df['AWAY_CLUTCH_COMPOSITE']
    
    # 35. Advanced Efficiency Metrics
    df['HOME_EFFICIENCY_COMPOSITE'] = (
        df['HOME_EFFICIENCY'] * 0.5 +
        df['HOME_OFFENSIVE_EFFICIENCY'] * 0.3 +
        df['HOME_DEF_EFFICIENCY'] * 0.2
    )
    df['AWAY_EFFICIENCY_COMPOSITE'] = (
        df['AWAY_EFFICIENCY'] * 0.5 +
        df['AWAY_OFFENSIVE_EFFICIENCY'] * 0.3 +
        df['AWAY_DEF_EFFICIENCY'] * 0.2
    )
    df['EFFICIENCY_COMPOSITE_DIFF'] = df['HOME_EFFICIENCY_COMPOSITE'] - df['AWAY_EFFICIENCY_COMPOSITE']
    
    # 36. Game Context Features (based on team strength)
    df['GAME_IMPORTANCE'] = (df['HOME_SEASON_WIN_PCT'] + df['AWAY_SEASON_WIN_PCT']) / 2
    df['HOME_PRESSURE'] = df['GAME_IMPORTANCE'] * df['HOME_COURT_ADVANTAGE']
    df['AWAY_PRESSURE'] = df['GAME_IMPORTANCE'] * 0.8  # Away teams face less pressure
    df['PRESSURE_DIFF'] = df['HOME_PRESSURE'] - df['AWAY_PRESSURE']
    
    # 37. Advanced Statistical Interactions
    df['WIN_PCT_MOMENTUM_INTERACTION'] = df['WIN_PCT_DIFF'] * df['MOMENTUM_COMPOSITE_DIFF']
    df['STRENGTH_CLUTCH_INTERACTION'] = df['STRENGTH_SCORE_DIFF'] * df['CLUTCH_COMPOSITE_DIFF']
    df['EFFICIENCY_PRESSURE_INTERACTION'] = df['EFFICIENCY_COMPOSITE_DIFF'] * df['PRESSURE_DIFF']
    
    # 38. Advanced Ratio Features
    df['HOME_DOMINANCE_RATIO'] = df['HOME_PTS_rolling5'] / (df['AWAY_PTS_rolling5'] + 1) * df['HOME_FG_PCT_rolling5'] / (df['AWAY_FG_PCT_rolling5'] + 0.01)
    df['AWAY_DOMINANCE_RATIO'] = df['AWAY_PTS_rolling5'] / (df['HOME_PTS_rolling5'] + 1) * df['AWAY_FG_PCT_rolling5'] / (df['HOME_FG_PCT_rolling5'] + 0.01)
    df['DOMINANCE_RATIO_DIFF'] = df['HOME_DOMINANCE_RATIO'] - df['AWAY_DOMINANCE_RATIO']
    
    # 39. Advanced Trend Features
    df['HOME_TREND_STRENGTH'] = df['HOME_MOMENTUM_COMPOSITE'] * df['HOME_RECENT_FORM'] * df['HOME_SEASON_WIN_PCT']
    df['AWAY_TREND_STRENGTH'] = df['AWAY_MOMENTUM_COMPOSITE'] * df['AWAY_RECENT_FORM'] * df['AWAY_SEASON_WIN_PCT']
    df['TREND_STRENGTH_DIFF'] = df['HOME_TREND_STRENGTH'] - df['AWAY_TREND_STRENGTH']
    
    # 40. Final Composite Score
    df['FINAL_COMPOSITE_SCORE'] = (
        df['PERFORMANCE_INDEX_DIFF'] * 0.25 +
        df['MOMENTUM_COMPOSITE_DIFF'] * 0.2 +
        df['CLUTCH_COMPOSITE_DIFF'] * 0.15 +
        df['EFFICIENCY_COMPOSITE_DIFF'] * 0.15 +
        df['PRESSURE_DIFF'] * 0.1 +
        df['DOMINANCE_RATIO_DIFF'] * 0.1 +
        df['TREND_STRENGTH_DIFF'] * 0.05
    )
    
    # === AUC-OPTIMIZED FEATURES FOR 0.88+ AUC ===
    
    # 41. Advanced Probability Calibration Features
    df['HOME_WIN_PROBABILITY'] = 1 / (1 + np.exp(-df['FINAL_COMPOSITE_SCORE']))
    df['AWAY_WIN_PROBABILITY'] = 1 - df['HOME_WIN_PROBABILITY']
    df['PROBABILITY_DIFF'] = df['HOME_WIN_PROBABILITY'] - df['AWAY_WIN_PROBABILITY']
    
    # 42. Advanced Ranking Features
    df['HOME_RANKING_SCORE'] = (
        df['HOME_SEASON_WIN_PCT'] * 0.4 +
        df['HOME_PTS_rolling5'] / 120 * 0.3 +
        df['HOME_FG_PCT_rolling5'] * 0.2 +
        df['HOME_REB_rolling5'] / 50 * 0.1
    )
    df['AWAY_RANKING_SCORE'] = (
        df['AWAY_SEASON_WIN_PCT'] * 0.4 +
        df['AWAY_PTS_rolling5'] / 120 * 0.3 +
        df['AWAY_FG_PCT_rolling5'] * 0.2 +
        df['AWAY_REB_rolling5'] / 50 * 0.1
    )
    df['RANKING_DIFF'] = df['HOME_RANKING_SCORE'] - df['AWAY_RANKING_SCORE']
    
    # 43. Advanced Statistical Moments (based on actual data variance)
    df['HOME_VARIANCE'] = df['HOME_PTS_rolling5'].rolling(window=5).std().fillna(0.1)
    df['AWAY_VARIANCE'] = df['AWAY_PTS_rolling5'].rolling(window=5).std().fillna(0.1)
    df['VARIANCE_DIFF'] = df['HOME_VARIANCE'] - df['AWAY_VARIANCE']
    
    # 44. Advanced Interaction Terms for AUC
    df['WIN_PCT_EFFICIENCY_INTERACTION'] = df['WIN_PCT_DIFF'] * df['EFFICIENCY_COMPOSITE_DIFF']
    df['MOMENTUM_CLUTCH_INTERACTION'] = df['MOMENTUM_COMPOSITE_DIFF'] * df['CLUTCH_COMPOSITE_DIFF']
    df['STRENGTH_PRESSURE_INTERACTION'] = df['STRENGTH_SCORE_DIFF'] * df['PRESSURE_DIFF']
    
    # 45. Advanced Ratio Features for AUC
    df['HOME_ADVANTAGE_RATIO'] = df['HOME_COURT_ADVANTAGE'] * df['HOME_MOMENTUM_COMPOSITE'] * df['HOME_CLUTCH_COMPOSITE']
    df['AWAY_DISADVANTAGE_RATIO'] = 0.8 * df['AWAY_MOMENTUM_COMPOSITE'] * df['AWAY_CLUTCH_COMPOSITE']
    df['ADVANTAGE_DISADVANTAGE_DIFF'] = df['HOME_ADVANTAGE_RATIO'] - df['AWAY_DISADVANTAGE_RATIO']
    
    # 46. Advanced Trend Analysis (with clipping to prevent extreme values)
    df['HOME_TREND_ACCELERATION'] = np.clip(df['HOME_MOMENTUM_COMPOSITE'] * df['HOME_RECENT_FORM'] * df['HOME_SEASON_WIN_PCT'], -1000, 1000)
    df['AWAY_TREND_ACCELERATION'] = np.clip(df['AWAY_MOMENTUM_COMPOSITE'] * df['AWAY_RECENT_FORM'] * df['AWAY_SEASON_WIN_PCT'], -1000, 1000)
    df['TREND_ACCELERATION_DIFF'] = df['HOME_TREND_ACCELERATION'] - df['AWAY_TREND_ACCELERATION']
    
    # 47. Advanced Composite Metrics for AUC
    df['HOME_OVERALL_STRENGTH'] = (
        df['HOME_PERFORMANCE_INDEX'] * 0.3 +
        df['HOME_MOMENTUM_COMPOSITE'] * 0.25 +
        df['HOME_CLUTCH_COMPOSITE'] * 0.2 +
        df['HOME_EFFICIENCY_COMPOSITE'] * 0.15 +
        df['HOME_COURT_ADVANTAGE'] * 0.1
    )
    df['AWAY_OVERALL_STRENGTH'] = (
        df['AWAY_PERFORMANCE_INDEX'] * 0.3 +
        df['AWAY_MOMENTUM_COMPOSITE'] * 0.25 +
        df['AWAY_CLUTCH_COMPOSITE'] * 0.2 +
        df['AWAY_EFFICIENCY_COMPOSITE'] * 0.15 +
        0.8 * 0.1  # Away penalty
    )
    df['OVERALL_STRENGTH_DIFF'] = df['HOME_OVERALL_STRENGTH'] - df['AWAY_OVERALL_STRENGTH']
    
    # 48. Advanced Probability Features
    df['HOME_WIN_LIKELIHOOD'] = 1 / (1 + np.exp(-df['OVERALL_STRENGTH_DIFF']))
    df['AWAY_WIN_LIKELIHOOD'] = 1 - df['HOME_WIN_LIKELIHOOD']
    df['LIKELIHOOD_DIFF'] = df['HOME_WIN_LIKELIHOOD'] - df['AWAY_WIN_LIKELIHOOD']
    
    # 49. Advanced Statistical Features
    df['HOME_CONSISTENCY_SCORE'] = 1 / (df['HOME_VARIANCE'] + 0.01)
    df['AWAY_CONSISTENCY_SCORE'] = 1 / (df['AWAY_VARIANCE'] + 0.01)
    df['CONSISTENCY_DIFF'] = df['HOME_CONSISTENCY_SCORE'] - df['AWAY_CONSISTENCY_SCORE']
    
    # 50. Final AUC-Optimized Score
    df['AUC_OPTIMIZED_SCORE'] = (
        df['OVERALL_STRENGTH_DIFF'] * 0.3 +
        df['PROBABILITY_DIFF'] * 0.25 +
        df['ADVANTAGE_DISADVANTAGE_DIFF'] * 0.2 +
        df['TREND_ACCELERATION_DIFF'] * 0.15 +
        df['CONSISTENCY_DIFF'] * 0.1
    )
    
    return df