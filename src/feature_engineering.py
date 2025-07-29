import pandas as pd

def create_features(df):
    """
    Create features for modeling from the cleaned dataframe.
    Adds a 'HOME' indicator, rolling averages for key stats, and recent win streak for each team.
    """
    # HOME: 1 if 'vs.' in MATCHUP, 0 if '@'
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    # Sort for rolling features
    df = df.sort_values(['Team_ID', 'GAME_DATE_REAL'])

    # --- Rest days for each team ---
    df['GAME_DATE_REAL'] = pd.to_datetime(df['GAME_DATE_REAL'])
    df['REST_DAYS'] = df.groupby('Team_ID')['GAME_DATE_REAL'].diff().dt.days.fillna(0)

    # --- Overall team strength (season win percentage) ---
    # Calculate cumulative wins and losses for each team
    df['WIN'] = (df['WL'] == 'W').astype(int)
    df['LOSS'] = (df['WL'] == 'L').astype(int)
    
    # Calculate cumulative wins and losses
    df['CUM_WINS'] = df.groupby('Team_ID')['WIN'].cumsum()
    df['CUM_LOSSES'] = df.groupby('Team_ID')['LOSS'].cumsum()
    
    # Calculate season win percentage (excluding current game)
    df['SEASON_WIN_PCT'] = df['CUM_WINS'] / (df['CUM_WINS'] + df['CUM_LOSSES']).fillna(0.5)
    
    # --- Opponent Team_ID ---
    # Create a mapping of game IDs to team pairs
    game_teams = df.groupby('Game_ID')['Team_ID'].apply(list).reset_index()
    game_teams['OPP_TEAM_ID'] = game_teams['Team_ID'].apply(lambda x: x[1] if len(x) > 1 else None)
    game_teams['TEAM_ID'] = game_teams['Team_ID'].apply(lambda x: x[0] if len(x) > 0 else None)
    
    # Create a lookup dictionary for opponent team IDs
    opp_lookup = {}
    for _, row in game_teams.iterrows():
        if row['TEAM_ID'] is not None and row['OPP_TEAM_ID'] is not None:
            opp_lookup[(row['Game_ID'], row['TEAM_ID'])] = row['OPP_TEAM_ID']
    
    # Apply opponent team ID mapping
    df['OPP_TEAM_ID'] = df.apply(lambda row: opp_lookup.get((row['Game_ID'], row['Team_ID'])), axis=1)

    # --- Rolling averages (last 5 and 10 games) ---
    rolling_stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']
    rolling_windows = [5, 10]
    for window in rolling_windows:
        for stat in rolling_stats:
            df[f'{stat}_rolling{window}'] = df.groupby('Team_ID')[stat].transform(lambda x: x.rolling(window, 1).mean())
    
    # Rolling win streak (last 5 games)
    df['WIN_STREAK5'] = df.groupby('Team_ID')['WIN'].transform(lambda x: x.rolling(5, 1).sum())

    # --- Opponent features via lookup (no merge, unique index) ---
    for window in rolling_windows:
        for stat in rolling_stats:
            stat_col = f'{stat}_rolling{window}'
            opp_col = f'OPP_{stat}_rolling{window}'
            stat_lookup = (
                df.groupby(['Team_ID', 'GAME_DATE_REAL'])[stat_col]
                .mean()
            )
            df[opp_col] = df.set_index(['OPP_TEAM_ID', 'GAME_DATE_REAL']).index.map(stat_lookup)
    
    # Opponent rest days
    rest_lookup = (
        df.groupby(['Team_ID', 'GAME_DATE_REAL'])['REST_DAYS']
        .mean()
    )
    df['OPP_REST_DAYS'] = df.set_index(['OPP_TEAM_ID', 'GAME_DATE_REAL']).index.map(rest_lookup)
    
    # Opponent season win percentage
    season_win_lookup = (
        df.groupby(['Team_ID', 'GAME_DATE_REAL'])['SEASON_WIN_PCT']
        .mean()
    )
    df['OPP_SEASON_WIN_PCT'] = df.set_index(['OPP_TEAM_ID', 'GAME_DATE_REAL']).index.map(season_win_lookup)

    return df 