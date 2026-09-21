"""
Build the game-level modelling dataset from raw team box scores.

Every column produced here describes what was known BEFORE tip-off. Team history
is shifted by one game, and Elo ratings are the ratings carried into the game.
The only columns that describe the game itself are HOME_WON and the raw box-score
columns, which exist for the team-stats screen and are never used as features.
"""

import numpy as np
import pandas as pd

# Box-score columns carried over from the raw export, for a team and its opponent.
BOX_STATS = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
             'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

# Per-game quantities that get averaged over a team's recent games.
ROLLING_STATS = ['PTS', 'OPP_PTS', 'MARGIN', 'NET_RTG', 'OFF_RTG', 'DEF_RTG', 'EFG',
                 'OPP_EFG', 'TS', 'REB', 'AST', 'TOV', 'TOV_RATE', 'OREB_PCT',
                 'FT_RATE', 'FG3_PCT', 'FT_PCT', 'STL', 'BLK']

ROLLING_WINDOWS = (5, 10)

# Two Elo ratings, deliberately different. The fast one reacts to margin of
# victory and moves quickly, so it tracks current form; the slow one only counts
# wins and losses, so it holds a longer view of team strength. Together they beat
# either alone. The fast settings were tuned on the first 40% of the season only.
ELO_START = 1500.0
ELO_FAST = dict(k=40, home_advantage=40, margin_of_victory=True, column='ELO_PRE')
ELO_SLOW = dict(k=20, home_advantage=100, margin_of_victory=False, column='ELO_SLOW_PRE')

ELO_K = ELO_FAST['k']
ELO_HOME_ADVANTAGE = ELO_FAST['home_advantage']
ELO_SLOW_HOME_ADVANTAGE = ELO_SLOW['home_advantage']


def load_team_games(csv_path, start_date='2024-10-15'):
    """
    One row per (game, team), with the opponent's box score attached.

    The raw export re-appends playoff games on each scrape, so the same
    (Game_ID, Team_ID) can appear up to 39 times with different DATE_ADDED.
    Dropping those duplicates is what keeps the playoff games in the dataset.
    """
    df = pd.read_csv(csv_path)
    df['GAME_DATE_REAL'] = pd.to_datetime(df['GAME_DATE_REAL'])
    df = df[df['GAME_DATE_REAL'] >= pd.to_datetime(start_date)]

    before = len(df)
    df = df.drop_duplicates(subset=['Game_ID', 'Team_ID'], keep='first')
    if before != len(df):
        print(f'Dropped {before - len(df)} duplicate team-game rows from the export')

    # A game needs both teams to be usable.
    two_sided = df.groupby('Game_ID')['Team_ID'].transform('size') == 2
    if (~two_sided).any():
        print(f'Skipping {df.loc[~two_sided, "Game_ID"].nunique()} games without exactly two teams')
    df = df[two_sided].copy()

    opponent = df[['Game_ID', 'Team_ID'] + BOX_STATS].copy()
    opponent.columns = ['Game_ID', 'OPP_TEAM_ID'] + ['OPP_' + c for c in BOX_STATS]
    df = df.merge(opponent, on='Game_ID')
    df = df[df['Team_ID'] != df['OPP_TEAM_ID']].copy()

    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', regex=False).astype(int)
    df['WON'] = (df['WL'] == 'W').astype(int)
    df = df.sort_values(['GAME_DATE_REAL', 'Game_ID', 'Team_ID']).reset_index(drop=True)

    # Efficiency measures the raw export doesn't carry.
    fga = df['FGA'].replace(0, np.nan)
    df['EFG'] = (df['FGM'] + 0.5 * df['FG3M']) / fga
    df['TS'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
    df['OPP_EFG'] = (df['OPP_FGM'] + 0.5 * df['OPP_FG3M']) / df['OPP_FGA'].replace(0, np.nan)
    df['POSS'] = df['FGA'] - df['OREB'] + df['TOV'] + 0.44 * df['FTA']
    poss = df['POSS'].replace(0, np.nan)
    df['OFF_RTG'] = 100 * df['PTS'] / poss
    df['DEF_RTG'] = 100 * df['OPP_PTS'] / poss
    df['NET_RTG'] = df['OFF_RTG'] - df['DEF_RTG']
    df['MARGIN'] = df['PTS'] - df['OPP_PTS']
    df['TOV_RATE'] = df['TOV'] / poss
    df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['OPP_DREB']).replace(0, np.nan)
    df['FT_RATE'] = df['FTA'] / fga

    print(f'Loaded {df["Game_ID"].nunique()} games '
          f'({df["GAME_DATE_REAL"].min():%b %d, %Y} to {df["GAME_DATE_REAL"].max():%b %d, %Y})')
    return df


def add_pre_game_history(panel, shift=True):
    """
    Rolling and season-to-date form for each team.

    With shift=True every window ends at the previous game, so a row never sees
    its own result. shift=False is only for building the state of a team going
    into its NEXT game, which is what live predictions need.
    """
    panel = panel.sort_values(['Team_ID', 'GAME_DATE_REAL', 'Game_ID']).copy()
    lag = (lambda s: s.shift(1)) if shift else (lambda s: s)
    by_team = panel.groupby('Team_ID', sort=False)

    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            panel[f'{stat}_r{window}'] = by_team[stat].transform(
                lambda s: lag(s).rolling(window, min_periods=2).mean())

    panel['WIN_PCT_std'] = by_team['WON'].transform(lambda s: lag(s).expanding().mean())
    panel['MARGIN_std'] = by_team['MARGIN'].transform(lambda s: lag(s).expanding().mean())
    panel['WIN_r10'] = by_team['WON'].transform(lambda s: lag(s).rolling(10, min_periods=2).mean())
    panel['MARGIN_SD10'] = by_team['MARGIN'].transform(lambda s: lag(s).rolling(10, min_periods=3).std())
    panel['GAMES_PLAYED'] = by_team.cumcount() + (0 if shift else 1)

    # Days since the team last played. Capped: 8 days off and 20 days off are
    # the same thing as far as fatigue goes.
    panel['REST'] = by_team['GAME_DATE_REAL'].transform(
        lambda s: (s - s.shift(1)).dt.days).clip(upper=7)
    panel['B2B'] = np.where(panel['REST'].isna(), np.nan, (panel['REST'] == 1).astype(float))

    # Form split by venue -- some teams are much better at home than on the road.
    for name, is_home in (('HOMEREC', 1), ('AWAYREC', 0)):
        venue_result = panel['WON'].where(panel['IS_HOME'] == is_home)
        panel[name] = (panel.assign(_r=venue_result)
                       .groupby('Team_ID', sort=False)['_r']
                       .transform(lambda s: lag(s).expanding().mean()))

    return panel.sort_values(['GAME_DATE_REAL', 'Game_ID', 'Team_ID']).reset_index(drop=True)


def add_elo(panel, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE, start=ELO_START,
            margin_of_victory=True, column='ELO_PRE'):
    """
    Sequential Elo. The rating stored on each row is the one carried INTO the
    game, so it is known before tip-off; the update happens once the result is in.

    Returns (panel, final_ratings) where final_ratings is the state after the
    last game -- what a prediction for a future game should use.
    """
    ratings = {}
    pre = pd.Series(np.nan, index=panel.index)

    for _, game in panel.groupby('Game_ID', sort=False):
        home = game[game['IS_HOME'] == 1]
        away = game[game['IS_HOME'] == 0]
        if len(home) != 1 or len(away) != 1:
            for idx in game.index:
                pre[idx] = ratings.get(panel.at[idx, 'Team_ID'], start)
            continue

        h_idx, a_idx = home.index[0], away.index[0]
        h_team, a_team = panel.at[h_idx, 'Team_ID'], panel.at[a_idx, 'Team_ID']
        h_rating = ratings.get(h_team, start)
        a_rating = ratings.get(a_team, start)
        pre[h_idx], pre[a_idx] = h_rating, a_rating

        expected_home = 1 / (1 + 10 ** ((a_rating - h_rating - home_advantage) / 400))
        home_won = panel.at[h_idx, 'WON']

        multiplier = 1.0
        if margin_of_victory:
            # A blowout moves the ratings further than a one-possession game, damped
            # by how lopsided the matchup already looked (FiveThirtyEight's form).
            margin = abs(panel.at[h_idx, 'MARGIN'])
            edge = (h_rating + home_advantage - a_rating) if home_won else \
                   (a_rating - h_rating - home_advantage)
            multiplier = ((margin + 3) ** 0.8) / (7.5 + 0.006 * edge)

        delta = k * multiplier * (home_won - expected_home)
        ratings[h_team] = h_rating + delta
        ratings[a_team] = a_rating - delta

    panel = panel.copy()
    panel[column] = pre
    return panel, ratings


# Columns carried into the game-level table for both teams.
HISTORY_COLS = (
    [f'{s}_r{w}' for w in ROLLING_WINDOWS for s in ROLLING_STATS]
    + ['WIN_PCT_std', 'MARGIN_std', 'WIN_r10', 'MARGIN_SD10', 'REST', 'B2B',
       'HOMEREC', 'AWAYREC', 'ELO_PRE', 'ELO_SLOW_PRE', 'GAMES_PLAYED']
)

# Raw box-score columns kept for the team-stats screen. Not model features.
RESULT_COLS = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']


def to_game_level(panel):
    """Fold the two rows of each game into one, from the home team's point of view."""
    home = panel[panel['IS_HOME'] == 1].set_index('Game_ID')
    away = panel[panel['IS_HOME'] == 0].set_index('Game_ID')

    columns = {
        'GAME_DATE_REAL': home['GAME_DATE_REAL'],
        'HOME_TEAM_ID': home['Team_ID'],
        'AWAY_TEAM_ID': away['Team_ID'],
        'HOME_WON': home['WON'],
        'SEASON_TYPE': pd.Series([str(g).zfill(10)[:3] for g in home.index], index=home.index),
    }
    for col in HISTORY_COLS + RESULT_COLS:
        columns['HOME_' + col] = home[col]
        columns['AWAY_' + col] = away[col]

    games = pd.DataFrame(columns)
    return games.sort_values('GAME_DATE_REAL').reset_index()


def current_team_state(csv_path, start_date='2024-10-15'):
    """
    Each team's form and Elo going into their NEXT game.

    Same columns as the model's training rows, but the windows include the
    team's most recent game rather than stopping one short of it.
    """
    panel = load_team_games(csv_path, start_date)
    _, fast = add_elo(panel, **ELO_FAST)
    _, slow = add_elo(panel, **ELO_SLOW)
    unshifted = add_pre_game_history(panel, shift=False)
    latest = (unshifted.sort_values(['GAME_DATE_REAL', 'Game_ID'])
              .groupby('Team_ID').last())
    elo_cols = {'ELO_PRE', 'ELO_SLOW_PRE'}
    state = latest[[c for c in HISTORY_COLS if c not in elo_cols]].copy()
    state['ELO_PRE'] = pd.Series(fast).reindex(state.index).fillna(ELO_START)
    state['ELO_SLOW_PRE'] = pd.Series(slow).reindex(state.index).fillna(ELO_START)
    return state


def load_and_clean_data(csv_path, start_date='2024-10-15'):
    """Game-level dataset with pre-game features only."""
    panel = load_team_games(csv_path, start_date)
    panel = add_pre_game_history(panel)
    panel, _ = add_elo(panel, **ELO_FAST)
    panel, _ = add_elo(panel, **ELO_SLOW)
    games = to_game_level(panel)

    # A team's first game has no history to average; those rows can't be modelled.
    usable = games['HOME_MARGIN_r10'].notna() & games['AWAY_MARGIN_r10'].notna() \
        & games['HOME_WIN_PCT_std'].notna() & games['AWAY_WIN_PCT_std'].notna()
    print(f'{len(games)} games, {(~usable).sum()} dropped for having no prior form')
    return games[usable].reset_index(drop=True)
