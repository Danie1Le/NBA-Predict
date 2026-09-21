import { Info } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

// A popover rather than a page section, so the dashboard fits on one screen.
const HowItWorks = () => {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const buttonRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false);
        buttonRef.current?.focus();
      }
    };
    const onPointerDown = (e) => {
      if (!rootRef.current?.contains(e.target)) setOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('pointerdown', onPointerDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('pointerdown', onPointerDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={buttonRef}
        type="button"
        aria-expanded={open}
        aria-controls="how-it-works"
        onClick={() => setOpen((isOpen) => !isOpen)}
        className="inline-flex min-h-[44px] items-center gap-1.5 rounded-lg px-3 text-sm font-medium text-label-2 transition-colors hover:bg-fill hover:text-label"
      >
        <Info aria-hidden="true" className="size-4" />
        How it works
      </button>

      {open && (
        <div
          id="how-it-works"
          role="region"
          aria-label="How the prediction works"
          className="popover-in absolute right-0 top-full z-20 mt-2 w-[min(26rem,calc(100vw-2rem))] space-y-3 rounded-2xl
                     bg-surface/90 p-5 text-sm text-label-2 shadow-xl ring-1 ring-separator backdrop-blur-xl backdrop-saturate-150"
        >
          <p>
            Each model learned from the 1,281 games of the 2024–25 NBA season, playoffs included. For a matchup, it
            compares the two teams' Elo ratings, their form over their last five and ten games (points won by,
            shooting efficiency, turnovers, rebounding), their season records, and home-court advantage, then
            estimates each team's chance to win.
          </p>
          <p>
            It doesn't know about injuries, trades or who's resting, so read the result as a take on team strength
            and recent form rather than a forecast for a specific night.
          </p>
          <p>
            Tested by walk-forward validation, where every prediction comes from a model trained only on earlier
            games, it picked the winner 66.6% of the time (AUC 0.72). Always backing the home team gets 54%. When
            it calls a game with high confidence, which is about half of them, it is right 74% of the time.
          </p>
          <p className="border-t border-separator pt-3">
            Not affiliated with the NBA.{' '}
            <a
              href="https://github.com/Danie1Le/NBA-Predict"
              className="font-medium text-tint underline-offset-2 hover:underline"
            >
              Source on GitHub
            </a>
          </p>
        </div>
      )}
    </div>
  );
};

export default HowItWorks;
