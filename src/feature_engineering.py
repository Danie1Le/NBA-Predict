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

    # --- Opponent Team_ID ---
    team_abbr_map = df.groupby('Team_ID').first().reset_index()
    team_abbr_map['ABBR'] = team_abbr_map['MATCHUP'].str.split(' ').str[0]
    abbr_to_id = dict(zip(team_abbr_map['ABBR'], team_abbr_map['Team_ID']))
    def get_opponent_team_id(row):
        parts = row['MATCHUP'].split(' ')
        opp_abbr = parts[-1]
        return abbr_to_id.get(opp_abbr, None)
    df['OPP_TEAM_ID'] = df.apply(get_opponent_team_id, axis=1)

    # --- Rolling averages (last 5 and 10 games) ---
    rolling_stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']
    rolling_windows = [5, 10]
    for window in rolling_windows:
        for stat in rolling_stats:
            df[f'{stat}_rolling{window}'] = df.groupby('Team_ID')[stat].transform(lambda x: x.rolling(window, 1).mean())
    # Convert WL to numeric: 1 for 'W', 0 for 'L'
    df['WIN'] = (df['WL'] == 'W').astype(int)
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

    return df 