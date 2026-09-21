"""
Turn the two teams' pre-game form into the features the models actually see.

Almost everything here is a difference: what matters for who wins is not how
good the home team is, but how much better it is than tonight's opponent.
"""

import numpy as np
import pandas as pd

from preprocessing import (ELO_HOME_ADVANTAGE, ELO_SLOW_HOME_ADVANTAGE,
                           HISTORY_COLS)

# Differences worth computing. The model uses a subset (MODEL_FEATURES); the
# rest are here so the evaluation script can test whether they earn their place.
DIFF_COLS = [c for c in HISTORY_COLS if c != 'GAMES_PLAYED']

# The features the shipped models train on, chosen by walk-forward validation
# in evaluate.py. Adding the other 100-odd differences did not beat this set.
MODEL_FEATURES = [
    'ELO_EXP_HOME',      # fast Elo's estimate of the home team's win probability
    'ELO_EXP_SLOW',      # the slow rating's estimate -- a longer view of strength
    'D_ELO_PRE',         # rating gap, which the model may scale differently
    'D_REST',            # days off, home minus away
    'D_B2B',             # playing a second night in a row
    'D_MARGIN_r10',      # points won by over the last 10 games
    'D_NET_RTG_r10',     # the same thing per 100 possessions
    'D_WIN_PCT_std',     # season-to-date win rate
    'D_EFG_r10',         # shooting efficiency
    'D_OPP_EFG_r10',     # shooting efficiency allowed -- the defensive half
    'D_TOV_RATE_r10',
    'D_OREB_PCT_r10',
]

# Rest and back-to-back can't be known for a hypothetical matchup with no date.
UNKNOWN_BEFORE_SCHEDULING = ['D_REST', 'D_B2B']


def elo_expectation(home_rating, away_rating, home_advantage=ELO_HOME_ADVANTAGE):
    """Elo's win probability for the home team."""
    return 1 / (1 + 10 ** ((away_rating - home_rating - home_advantage) / 400))


def create_features(df):
    """Add the difference features and Elo's win probability to a game-level frame."""
    df = df.copy()

    diffs = {}
    for col in DIFF_COLS:
        home, away = 'HOME_' + col, 'AWAY_' + col
        if home in df.columns and away in df.columns:
            diffs['D_' + col] = df[home] - df[away]
    df = pd.concat([df, pd.DataFrame(diffs, index=df.index)], axis=1)

    df['ELO_EXP_HOME'] = elo_expectation(df['HOME_ELO_PRE'], df['AWAY_ELO_PRE'])
    df['ELO_EXP_SLOW'] = elo_expectation(df['HOME_ELO_SLOW_PRE'], df['AWAY_ELO_SLOW_PRE'],
                                         ELO_SLOW_HOME_ADVANTAGE)
    return df.replace([np.inf, -np.inf], np.nan)


def matchup_features(home_state, away_state):
    """
    Build one prediction row from two teams' current state.

    home_state and away_state are rows of preprocessing.current_team_state().
    Rest days are left at zero: without a scheduled date there is nothing to
    compute, and zero is the neutral value for a difference.
    """
    row = {}
    for col in DIFF_COLS:
        if col in home_state.index and col in away_state.index:
            row['D_' + col] = float(home_state[col]) - float(away_state[col])
    for col in UNKNOWN_BEFORE_SCHEDULING:
        row[col] = 0.0
    row['ELO_EXP_HOME'] = elo_expectation(float(home_state['ELO_PRE']),
                                          float(away_state['ELO_PRE']))
    row['ELO_EXP_SLOW'] = elo_expectation(float(home_state['ELO_SLOW_PRE']),
                                          float(away_state['ELO_SLOW_PRE']),
                                          ELO_SLOW_HOME_ADVANTAGE)
    return row
