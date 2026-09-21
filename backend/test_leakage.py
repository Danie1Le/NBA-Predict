"""
Guards against the model seeing the future. Run: python test_leakage.py
(also works under pytest, if you install it)

These exist because the feature code used to average each team's stats over a
window that included the game being predicted, which made the reported accuracy
meaningless. Every check here re-derives values from the raw rows rather than
trusting the feature code.
"""

import sys
import warnings

import numpy as np
import pandas as pd

from featureEngineering import (MODEL_FEATURES, create_features, elo_expectation,
                                matchup_features)
from preprocessing import (ELO_FAST, ELO_SLOW, add_elo, add_pre_game_history,
                           current_team_state, load_and_clean_data, load_team_games)
from trainModel import build_model, date_ordered_split

warnings.filterwarnings('ignore')
CSV = 'Data/NBA_GAMES.csv'

_panel = None
_games = None


def panel():
    global _panel
    if _panel is None:
        p = add_pre_game_history(load_team_games(CSV))
        p, _ = add_elo(p, **ELO_FAST)
        p, _ = add_elo(p, **ELO_SLOW)
        _panel = p
    return _panel


def games():
    global _games
    if _games is None:
        _games = create_features(load_and_clean_data(CSV))
    return _games


def test_rolling_windows_use_only_earlier_games():
    """Recompute sampled rolling values from the raw rows that precede them."""
    p = panel()
    ordered = p.sort_values(['Team_ID', 'GAME_DATE_REAL', 'Game_ID'])
    rng = np.random.default_rng(0)
    checked = 0
    for i in rng.choice(len(p), 300, replace=False):
        row = p.iloc[i]
        earlier = ordered[(ordered['Team_ID'] == row['Team_ID'])
                          & ((ordered['GAME_DATE_REAL'] < row['GAME_DATE_REAL'])
                             | ((ordered['GAME_DATE_REAL'] == row['GAME_DATE_REAL'])
                                & (ordered['Game_ID'] < row['Game_ID'])))]
        for stat, window in (('PTS', 5), ('MARGIN', 10), ('NET_RTG', 5)):
            prior = earlier[stat].dropna().tail(window)
            expected = prior.mean() if len(prior) >= 2 else np.nan
            actual = row[f'{stat}_r{window}']
            assert (np.isnan(actual) and np.isnan(expected)) or abs(actual - expected) < 1e-8, \
                f'{stat}_r{window} for team {row["Team_ID"]}: {actual} != {expected}'
            checked += 1
    print(f'  {checked} rolling values match a from-scratch recomputation')


def test_shipped_game_frame_is_pre_game():
    """
    The checks above exercise the helpers. This one checks the frame dataLoader
    actually trains on, so a leak reintroduced in load_and_clean_data is caught.
    """
    g = games()
    raw = load_team_games(CSV).sort_values(['Team_ID', 'GAME_DATE_REAL', 'Game_ID'])
    rng = np.random.default_rng(7)
    for i in rng.choice(len(g), 120, replace=False):
        row = g.iloc[i]
        for side in ('HOME', 'AWAY'):
            team = row[f'{side}_TEAM_ID']
            earlier = raw[(raw['Team_ID'] == team)
                          & ((raw['GAME_DATE_REAL'] < row['GAME_DATE_REAL'])
                             | ((raw['GAME_DATE_REAL'] == row['GAME_DATE_REAL'])
                                & (raw['Game_ID'] < row['Game_ID'])))]
            expected = earlier['PTS'].tail(5).mean()
            actual = row[f'{side}_PTS_r5']
            assert abs(actual - expected) < 1e-8, (
                f'{side}_PTS_r5 for team {team} on {row["GAME_DATE_REAL"]:%Y-%m-%d}: '
                f'{actual:.4f} != {expected:.4f} from prior games')
    print('  240 team-rows in the shipped frame average prior games only')


def test_shipped_features_do_not_see_the_outcome():
    """A tripwire on the trained frame: form predicts, it does not reveal."""
    g = games()
    corr = g[MODEL_FEATURES].corrwith(g['HOME_WON']).abs().sort_values(ascending=False)
    assert corr.max() < 0.45, f'{corr.idxmax()} correlates {corr.max():.3f} with the result'
    print(f'  strongest model-feature correlation: {corr.idxmax()} at {corr.max():.3f}')


def test_season_to_date_excludes_the_current_game():
    p = panel().sort_values(['Team_ID', 'GAME_DATE_REAL', 'Game_ID'])
    expected = p.groupby('Team_ID')['WON'].transform(lambda s: s.shift(1).expanding().mean())
    assert np.allclose(expected, p['WIN_PCT_std'], equal_nan=True)


def test_a_teams_first_game_has_no_history():
    first = panel().query('GAMES_PLAYED == 0')
    assert len(first) == 30
    for col in ('WIN_PCT_std', 'PTS_r5', 'MARGIN_r10', 'REST'):
        assert first[col].isna().all(), f'{col} is populated before any game was played'


def test_no_feature_is_a_disguised_copy_of_the_result():
    """A feature that saw the outcome would correlate far harder than form does."""
    p = panel()
    cols = [c for c in p.select_dtypes('number').columns
            if c.endswith(('_r5', '_r10', '_std')) or c in
            ('ELO_PRE', 'ELO_SLOW_PRE', 'WIN_r10', 'MARGIN_SD10', 'REST', 'B2B',
             'HOMEREC', 'AWAYREC')]
    corr = p[cols].corrwith(p['WON']).abs().sort_values(ascending=False)
    assert corr.max() < 0.5, f'{corr.idxmax()} correlates {corr.max():.3f} with the result'
    print(f'  strongest pre-game correlation: {corr.idxmax()} at {corr.max():.3f}')


def test_elo_is_the_rating_carried_into_the_game():
    p = panel()
    assert (p.query('GAMES_PLAYED == 0')['ELO_PRE'] == 1500).all()
    # Elo is zero-sum across teams. It is NOT zero-sum across team-games: playoff
    # teams play more games and rate higher, so a game-weighted mean sits high.
    final = p.sort_values(['GAME_DATE_REAL', 'Game_ID']).groupby('Team_ID')['ELO_PRE'].last()
    assert abs(final.mean() - 1500) < 10, f'ratings drifted to {final.mean():.1f}'


def test_duplicate_scrapes_are_dropped_and_playoffs_survive():
    """The raw export repeats playoff games; skipping them loses the postseason."""
    p = panel()
    assert not p.duplicated(subset=['Game_ID', 'Team_ID']).any()
    assert (p.groupby('Game_ID').size() == 2).all()
    assert (p.groupby('Game_ID')['IS_HOME'].sum() == 1).all()
    playoffs = sum(str(g).zfill(10).startswith('004') for g in p['Game_ID'].unique())
    assert playoffs == 84, f'expected 84 playoff games, kept {playoffs}'


def test_model_features_are_present_and_complete():
    g = games()
    missing = [f for f in MODEL_FEATURES if f not in g.columns]
    assert not missing, f'missing features: {missing}'
    nulls = g[MODEL_FEATURES].isna().sum()
    assert nulls.sum() == 0, f'null features: {nulls[nulls > 0].to_dict()}'


def test_the_split_never_trains_on_a_later_game():
    g = games().sort_values('GAME_DATE_REAL')
    dates = g['GAME_DATE_REAL'].values
    train, test, _, _ = date_ordered_split(dates, g['HOME_WON'].values)
    assert train.max() <= test.min(), 'a training game was played after a test game'


def test_shuffled_labels_fall_to_chance():
    """If the harness itself leaked, this would score above the base rate."""
    g = games()
    X = g[MODEL_FEATURES].values.astype(float)
    y = g['HOME_WON'].values
    rng = np.random.default_rng(1)
    scores = []
    for _ in range(5):
        shuffled = rng.permutation(y)
        X_tr, X_te, y_tr, y_te = date_ordered_split(X, shuffled)
        model = build_model('logreg').fit(X_tr, y_tr)
        scores.append((model.predict(X_te) == y_te).mean())
    assert max(scores) < 0.60, f'shuffled labels scored {max(scores):.3f}'
    print(f'  shuffled-label accuracy {np.mean(scores):.3f} (chance)')


def test_swapping_the_teams_flips_the_matchup():
    state = current_team_state(CSV)
    a, b = state.index[0], state.index[1]
    forward = matchup_features(state.loc[a], state.loc[b])
    reverse = matchup_features(state.loc[b], state.loc[a])
    for key, value in forward.items():
        if key.startswith('D_'):
            assert abs(value + reverse[key]) < 1e-9, f'{key} did not negate'
    # Home advantage is the one thing that does not flip: whoever hosts gains it.
    assert forward['ELO_EXP_HOME'] > 1 - reverse['ELO_EXP_HOME']


def test_live_state_includes_the_most_recent_game():
    """Training rows stop one game short by design; a live prediction must not."""
    state = current_team_state(CSV)
    p = panel().sort_values(['GAME_DATE_REAL', 'Game_ID'])
    team = state.index[0]
    played = p[p['Team_ID'] == team]
    assert abs(state.loc[team, 'PTS_r5'] - played['PTS'].tail(5).mean()) < 1e-8
    last_row_value = played['PTS_r5'].iloc[-1]
    assert abs(state.loc[team, 'PTS_r5'] - last_row_value) > 1e-9, \
        'live state matches the last training row, so it is a game behind'


def test_elo_expectation_is_symmetric_without_home_advantage():
    assert abs(elo_expectation(1600, 1400, 0) + elo_expectation(1400, 1600, 0) - 1) < 1e-12
    assert abs(elo_expectation(1500, 1500, 0) - 0.5) < 1e-12


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = []
    for test in tests:
        try:
            test()
            print(f'ok   {test.__name__}')
        except AssertionError as exc:
            print(f'FAIL {test.__name__}: {exc}')
            failed.append(test.__name__)
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
