import { AlertCircle } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { inkFor } from '../teams';
import { MODEL_NAMES } from './ModelSelector';

// Top-down NBA court in feet (94 x 50), with a 2.5 ft apron around it.
// Corner threes run straight until they meet the 23.75 ft arc at x = 14.2.
const LEFT_END = 'M0 17H19V33H0 M19 19A6 6 0 0 1 19 31 M0 3H14.2A23.75 23.75 0 0 1 14.2 47H0 M5.25 21A4 4 0 0 1 5.25 29 M4 22V28';
const RIGHT_END = 'M94 17H75V33H94 M75 19A6 6 0 0 0 75 31 M94 3H79.8A23.75 23.75 0 0 0 79.8 47H94 M88.75 21A4 4 0 0 0 88.75 29 M90 22V28';

const CourtLines = ({ ink }) => (
  <svg viewBox="-2.5 -2.5 99 55" aria-hidden="true" focusable="false">
    <g fill={ink} fillOpacity="0.08">
      <rect x="0" y="17" width="19" height="16" />
      <rect x="75" y="17" width="19" height="16" />
    </g>
    <g fill="none" stroke={ink} strokeOpacity="0.5" strokeWidth="1.5">
      <rect x="0" y="0" width="94" height="50" />
      <path d="M47 0V50" />
      <circle cx="47" cy="25" r="6" />
      <path d={LEFT_END} />
      <path d={RIGHT_END} />
      <circle cx="5.25" cy="25" r="0.75" />
      <circle cx="88.75" cy="25" r="0.75" />
    </g>
  </svg>
);

// Counts from the previous value to the new one in step with the court split.
const useCountUp = (target, duration = 700) => {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target ?? 50);

  useEffect(() => {
    if (target == null) {
      fromRef.current = 50;
      setValue(null);
      return undefined;
    }
    const from = fromRef.current;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion || from === target) {
      fromRef.current = target;
      setValue(target);
      return undefined;
    }

    let frame;
    const start = performance.now();
    const tick = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const next = from + (target - from) * (1 - (1 - progress) ** 3);
      fromRef.current = next;
      setValue(next);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return value;
};

// "an 82%", "an 11%", "a 64%"
const withArticle = (pct) => `${/^(8|11|18)/.test(String(pct)) ? 'an' : 'a'} ${pct}%`;

const summarize = (result, favorite, favoriteIsHome, favoritePct) => {
  const model = MODEL_NAMES[result.model_used] ?? result.model_used;
  const venue = favoriteIsHome ? 'at home' : 'on the road';
  const chance = `with ${withArticle(favoritePct)} chance to win`;
  switch (String(result.confidence).toLowerCase()) {
    case 'high':
      return `${model} makes the ${favorite.nickname} clear favorites ${venue}, ${chance}.`;
    case 'medium':
    case 'moderate':
      return `${model} favors the ${favorite.nickname} ${venue}, ${chance}.`;
    default:
      return `${model} sees a close game. The ${favorite.nickname} get a slight edge ${venue}, ${chance}.`;
  }
};

const TeamFigure = ({ team, pct, favored, align }) => (
  <div className={`min-w-0 ${align === 'right' ? 'text-right' : ''}`}>
    <p className="truncate text-sm font-medium text-label-2">{team.nickname}</p>
    <p
      className={`text-[2.75rem] md:text-5xl font-bold leading-none tracking-tight transition-colors
                  ${pct != null && !favored ? 'text-label-2' : 'text-label'}`}
    >
      {pct == null ? team.abbreviation : `${pct}%`}
    </p>
  </div>
);

const PredictionResult = ({ awayTeam, homeTeam, colors, result, requestedModel, stale, error, onRetry }) => {
  const homePct = result ? Math.round(result.home_win_probability * 100) : null;
  const animatedHome = useCountUp(homePct);
  const shownHome = animatedHome == null ? null : Math.round(animatedHome);
  const shownAway = shownHome == null ? null : 100 - shownHome;

  const homeFavored = result ? result.prediction === 1 : null;
  const split = result ? `${(result.away_win_probability * 100).toFixed(2)}%` : '50%';

  const usedFallback = result && result.model_used !== requestedModel;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="court-fit">
        <div className="court-frame">
          <div className="mb-3 flex items-end justify-between gap-4">
            <TeamFigure team={awayTeam} pct={shownAway} favored={homeFavored === false} align="left" />
            <TeamFigure team={homeTeam} pct={shownHome} favored={homeFavored === true} align="right" />
          </div>

          <div className="court" style={{ '--split': split, opacity: stale ? 0.55 : 1 }}>
            <div className="court-layer" style={{ backgroundColor: colors.home }}>
              <CourtLines ink={inkFor(colors.home)} />
            </div>
            <div className="court-layer court-away" style={{ backgroundColor: colors.away }}>
              <CourtLines ink={inkFor(colors.away)} />
            </div>
            <div className="court-seam" />
          </div>
        </div>

        <div aria-live="polite" className="mt-3 min-h-[3.25rem] text-pretty md:text-balance md:text-center">
          {error ? (
            <div role="alert" className="flex items-start gap-2 md:justify-center">
              <AlertCircle aria-hidden="true" className="mt-1 size-4 shrink-0 text-danger" />
              <p>
                {error}{' '}
                <button
                  type="button"
                  onClick={onRetry}
                  className="font-semibold text-tint underline-offset-2 hover:underline"
                >
                  Try again
                </button>
              </p>
            </div>
          ) : result ? (
            <>
              <p className="text-label">
                {homeFavored
                  ? summarize(result, homeTeam, true, homePct)
                  : summarize(result, awayTeam, false, 100 - homePct)}
              </p>
              {usedFallback && (
                <p className="mt-1 text-sm text-label-2">
                  {MODEL_NAMES[requestedModel] ?? requestedModel} isn't available on the server yet,
                  so {MODEL_NAMES[result.model_used] ?? result.model_used} made this prediction.
                </p>
              )}
            </>
          ) : (
            !stale && (
              <p className="text-label-2">
                The half-court line marks an even game. After you predict, each team's color covers its share of
                the court.
              </p>
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
