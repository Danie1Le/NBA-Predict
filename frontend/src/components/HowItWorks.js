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
            Each model learned from games of the 2024–25 NBA season, playoffs included. For a matchup, it compares
            the two teams' averages over their last five games (points, shooting, rebounds, assists and turnovers),
            their win rates, and home-court advantage, then estimates each team's chance to win.
          </p>
          <p>
            It doesn't know about injuries, trades or rest days, so read the result as a take on recent form rather
            than a forecast for a specific night.
          </p>
          <p>
            In the project's testing on held-out games, the models picked the winner 78.6% of the time (AUC 0.837),
            and 89.8% of the time when their confidence was high.
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
