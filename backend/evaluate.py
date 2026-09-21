"""
Walk-forward evaluation. Run: python evaluate.py

Every fold trains only on games played before the ones it predicts, which is the
only way to get a number that means anything for a model meant to predict the
future. The single date-ordered split inside train_model() is the same idea with
one fold; this script reports the average over several, plus calibration and the
accuracy-by-confidence figures the UI quotes.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)

from featureEngineering import MODEL_FEATURES, create_features
from preprocessing import load_and_clean_data
from trainModel import MODEL_TYPES, build_model

warnings.filterwarnings('ignore')

N_FOLDS = 6
BURN_IN = 0.4          # the first 40% of games only ever train, never get scored


def walk_forward(X, y, model_type, n_folds=N_FOLDS, burn_in=BURN_IN):
    """Expanding window: fold k trains on every game before its block."""
    bounds = np.linspace(int(len(y) * burn_in), len(y), n_folds + 1).astype(int)
    probs, index, fold_acc = [], [], []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        model = build_model(model_type)
        model.fit(X[:start], y[:start])
        p = model.predict_proba(X[start:stop])[:, 1]
        probs.append(p)
        index.append(np.arange(start, stop))
        fold_acc.append(accuracy_score(y[start:stop], (p >= 0.5).astype(int)))
    return np.concatenate(probs), np.concatenate(index), np.array(fold_acc)


def score(y_true, p):
    return dict(accuracy=accuracy_score(y_true, (p >= 0.5).astype(int)),
                auc=roc_auc_score(y_true, p),
                log_loss=log_loss(y_true, p),
                brier=brier_score_loss(y_true, p))


def main(csv_path='Data/NBA_GAMES.csv'):
    games = create_features(load_and_clean_data(csv_path))
    y = games['HOME_WON'].values
    print(f'\n{len(games)} games  {games.GAME_DATE_REAL.min():%b %d %Y} to '
          f'{games.GAME_DATE_REAL.max():%b %d %Y}   home win rate {y.mean():.3f}\n')

    all_diffs = [c for c in games.columns if c.startswith('D_')] + ['ELO_EXP_HOME']
    feature_sets = {
        'elo only (1 feature)': ['ELO_EXP_HOME'],
        'shipped features': MODEL_FEATURES,
        'every difference': all_diffs,
    }

    print('Walk-forward, %d folds:' % N_FOLDS)
    rows = []
    best = None
    for set_name, cols in feature_sets.items():
        X = games[cols].values.astype(float)
        for model_type in MODEL_TYPES:
            try:
                p, idx, folds = walk_forward(X, y, model_type)
            except ValueError as exc:
                print(f'  skipped {model_type}: {exc}')
                continue
            row = dict(features=set_name, n=len(cols), model=model_type,
                       **score(y[idx], p), fold_sd=folds.std())
            rows.append(row)
            if set_name == 'shipped features' and (best is None or row['auc'] > best[0]['auc']):
                best = (row, p, idx)

    table = pd.DataFrame(rows).sort_values('auc', ascending=False)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    row, p, idx = best
    y_held = y[idx]
    print(f'\nBaselines on the same {len(y_held)} scored games:')
    print(f'  always pick the home team   accuracy={y_held.mean():.4f}')
    print(f'  shipped {row["model"]:<19} accuracy={row["accuracy"]:.4f}  auc={row["auc"]:.4f}  '
          f'log loss={row["log_loss"]:.4f}')

    print('\nCalibration -- when it says 60%, does it happen 60% of the time?')
    cal = (pd.DataFrame({'p': p, 'y': y_held, 'bin': pd.cut(p, [0, .3, .4, .5, .6, .7, 1.0])})
           .groupby('bin', observed=True)
           .agg(games=('y', 'size'), predicted=('p', 'mean'), actual=('y', 'mean')))
    print(cal.to_string(float_format=lambda v: f'{v:.3f}'))

    # Same thresholds predictionService uses to label a prediction.
    print("\nAccuracy by the app's confidence label:")
    edge = np.abs(p - (1 - p))
    label = np.where(edge > 0.30, 'High', np.where(edge > 0.15, 'Medium', 'Low'))
    buckets = (pd.DataFrame({'label': label, 'correct': (p >= .5).astype(int) == y_held})
               .groupby('label').agg(games=('correct', 'size'), accuracy=('correct', 'mean'))
               .reindex(['High', 'Medium', 'Low']))
    buckets['share'] = buckets['games'] / buckets['games'].sum()
    print(buckets.to_string(float_format=lambda v: f'{v:.3f}'))

    season_type = games.loc[idx, 'SEASON_TYPE'].values
    print('\nRegular season vs playoffs:')
    for code, name in (('002', 'regular'), ('004', 'playoff')):
        mask = season_type == code
        if mask.sum() > 20:
            s = score(y_held[mask], p[mask])
            print(f'  {name:8s} games={mask.sum():4d} accuracy={s["accuracy"]:.4f} auc={s["auc"]:.4f}')

    print('\nCeiling check -- if these probabilities were exactly right, the best')
    print(f'possible accuracy would be {np.mean(np.maximum(p, 1 - p)):.4f}. '
          f'The model gets {row["accuracy"]:.4f}.')
    print('Beating that means predicting games more confidently, which needs')
    print('information box scores do not contain: injuries, rest, lineups.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='Data/NBA_GAMES.csv')
    main(ap.parse_args().csv)
