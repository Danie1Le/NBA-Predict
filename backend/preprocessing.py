import pandas as pd
import numpy as np

def create_game_level_dataset(csv_path, start_date='2024-10-15'):
    """
    Create a proper game-level dataset where each row represents a game between two teams.
    This fixes the data leakage issue by having one row per game instead of two.
    """
    df = pd.read_csv(csv_path)
    
    # Convert date and filter
    df['GAME_DATE_REAL'] = pd.to_datetime(df['GAME_DATE_REAL'])
    start_date = pd.to_datetime(start_date)
    df = df[df['GAME_DATE_REAL'] >= start_date]
    
    print(f'Using games from {start_date.strftime("%B %d, %Y")} onwards')
    print(f'Total rows before game-level processing: {len(df)}')
    
    # Create game-level dataset
    game_data = []
    
    # Group by Game_ID to get both teams for each game
    for game_id, game_teams in df.groupby('Game_ID'):
        if len(game_teams) != 2:
            continue  # Skip games with missing data
            
        # Sort teams to ensure consistent ordering (home team first)
        game_teams = game_teams.sort_values('Team_ID')
        
        # Get home and away teams
        home_team = game_teams.iloc[0] if 'vs.' in game_teams.iloc[0]['MATCHUP'] else game_teams.iloc[1]
        away_team = game_teams.iloc[1] if home_team is game_teams.iloc[0] else game_teams.iloc[0]
        
        # Determine winner
        home_won = home_team['WL'] == 'W'
        
        # Create game-level features
        game_row = {
            'Game_ID': game_id,
            'GAME_DATE_REAL': home_team['GAME_DATE_REAL'],
            'HOME_TEAM_ID': home_team['Team_ID'],
            'AWAY_TEAM_ID': away_team['Team_ID'],
            'HOME_WON': int(home_won),
            
            # Home team features
            'HOME_PTS': home_team['PTS'],
            'HOME_FG_PCT': home_team['FG_PCT'],
            'HOME_FG3_PCT': home_team['FG3_PCT'],
            'HOME_FT_PCT': home_team['FT_PCT'],
            'HOME_REB': home_team['REB'],
            'HOME_AST': home_team['AST'],
            'HOME_TOV': home_team['TOV'],
            
            # Away team features
            'AWAY_PTS': away_team['PTS'],
            'AWAY_FG_PCT': away_team['FG_PCT'],
            'AWAY_FG3_PCT': away_team['FG3_PCT'],
            'AWAY_FT_PCT': away_team['FT_PCT'],
            'AWAY_REB': away_team['REB'],
            'AWAY_AST': away_team['AST'],
            'AWAY_TOV': away_team['TOV'],
        }
        
        game_data.append(game_row)
    
    game_df = pd.DataFrame(game_data)
    print(f'Created {len(game_df)} game-level records')
    
    return game_df

def add_game_level_features(game_df):
    """
    Add rolling averages and other features to the game-level dataset.
    """
    # Sort by date for rolling calculations
    game_df = game_df.sort_values('GAME_DATE_REAL')
    
    # Add rolling averages for each team
    for team_col in ['HOME_TEAM_ID', 'AWAY_TEAM_ID']:
        for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']:
            home_stat = f'HOME_{stat}'
            away_stat = f'AWAY_{stat}'
            
            # Calculate rolling averages for home team
            home_rolling = game_df.groupby(team_col)[home_stat].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
            game_df[f'HOME_{stat}_rolling5'] = home_rolling
            
            # Calculate rolling averages for away team
            away_rolling = game_df.groupby(team_col)[away_stat].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
            game_df[f'AWAY_{stat}_rolling5'] = away_rolling
    
    # Add win percentage features
    # For home teams: use HOME_WON directly
    game_df['HOME_TEAM_ID_WIN_PCT'] = game_df.groupby('HOME_TEAM_ID')['HOME_WON'].expanding().mean().reset_index(0, drop=True)
    
    # For away teams: use 1 - HOME_WON (since HOME_WON = 1 means home team won, so away team lost)
    game_df['AWAY_TEAM_ID_WIN_PCT'] = game_df.groupby('AWAY_TEAM_ID')['HOME_WON'].apply(lambda x: (1 - x).expanding().mean()).reset_index(0, drop=True)
    
    return game_df

def load_and_clean_data(csv_path, start_date='2024-10-15'):
    """
    Load NBA games data and create proper game-level dataset.
    This fixes the fundamental data structure issue by creating one row per game.
    """
    # Create game-level dataset
    game_df = create_game_level_dataset(csv_path, start_date)
    
    # Add rolling features
    game_df = add_game_level_features(game_df)
    
    # Print missing value counts
    print('Missing values per column:')
    print(game_df.isnull().sum())
    
    # Drop rows with missing values
    game_df = game_df.dropna()
    
    print(f'Final game-level dataset: {len(game_df)} games')
    return game_df 