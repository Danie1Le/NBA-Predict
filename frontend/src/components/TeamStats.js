import React from 'react';

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const avg = (value) => value.toFixed(1);
const record = (wins, games) => `${wins}–${games - wins}`;
const winRate = (wins, games) => (games ? wins / games : 0);

// `value` decides who leads the row; `display` is what we show.
const GROUPS = [
  {
    title: 'Record',
    rows: [
      { label: 'All games', value: (s) => s.season_win_pct, display: (s) => record(s.season_wins, s.season_games) },
      { label: 'Last 10 games', value: (s) => winRate(s.last_10_wins, s.last_10_games), display: (s) => record(s.last_10_wins, s.last_10_games) },
      { label: 'Last 5 games', value: (s) => winRate(s.last_5_wins, s.last_5_games), display: (s) => record(s.last_5_wins, s.last_5_games) },
    ],
  },
  {
    title: 'Averages, all games',
    rows: [
      { label: 'Points', value: (s) => s.season_pts, display: (s) => avg(s.season_pts) },
      { label: 'Field goal %', value: (s) => s.season_fg_pct, display: (s) => pct(s.season_fg_pct) },
      { label: 'Three-point %', value: (s) => s.season_fg3_pct, display: (s) => pct(s.season_fg3_pct) },
      { label: 'Free throw %', value: (s) => s.season_ft_pct, display: (s) => pct(s.season_ft_pct) },
    ],
  },
  {
    title: 'Averages, last 5 games',
    rows: [
      { label: 'Points', value: (s) => s.last_5_pts, display: (s) => avg(s.last_5_pts) },
      { label: 'Field goal %', value: (s) => s.last_5_fg_pct, display: (s) => pct(s.last_5_fg_pct) },
      { label: 'Three-point %', value: (s) => s.last_5_fg3_pct, display: (s) => pct(s.last_5_fg3_pct) },
      { label: 'Free throw %', value: (s) => s.last_5_ft_pct, display: (s) => pct(s.last_5_ft_pct) },
      { label: 'Rebounds', value: (s) => s.last_5_reb, display: (s) => avg(s.last_5_reb) },
      { label: 'Assists', value: (s) => s.last_5_ast, display: (s) => avg(s.last_5_ast) },
      { label: 'Turnovers', value: (s) => s.last_5_tov, display: (s) => avg(s.last_5_tov), lowerIsBetter: true },
    ],
  },
];

const leaderOf = (row, away, home) => {
  if (!away || !home) return null;
  const a = row.value(away);
  const h = row.value(home);
  if (Math.abs(a - h) < 1e-9) return null;
  return (a > h) !== Boolean(row.lowerIsBetter) ? 'away' : 'home';
};

const Dot = ({ color }) => (
  <span aria-hidden="true" className="inline-block size-2 shrink-0 rounded-full ring-1 ring-black/10" style={{ backgroundColor: color }} />
);

const Value = ({ stats, row, leads, color, align }) => {
  if (!stats) {
    return <span className="text-label-3">–</span>;
  }
  return (
    <span className={`inline-flex items-center gap-2 whitespace-nowrap ${align === 'right' ? 'flex-row-reverse' : ''}`}>
      <span className={`tabular-nums ${leads ? 'font-semibold text-label' : 'text-label-2'}`}>{row.display(stats)}</span>
      {leads && <Dot color={color} />}
      {leads && <span className="sr-only">(leads)</span>}
    </span>
  );
};

const StatusNote = ({ team, state, serverReady, onRetry }) => {
  if (!serverReady) return null;
  if (state?.status === 'error') {
    return (
      <p className="mt-2 text-sm text-label-2">
        Stats for the {team.nickname} didn't load.{' '}
        <button type="button" onClick={onRetry} className="font-medium text-tint underline-offset-2 hover:underline">
          Try again
        </button>
      </p>
    );
  }
  return null;
};

const TeamStats = ({ awayTeam, homeTeam, colors, awayState, homeState, serverReady, onRetry }) => {
  const away = awayState?.status === 'ready' ? awayState.data : null;
  const home = homeState?.status === 'ready' ? homeState.data : null;
  const loading = serverReady && (!away || !home) && awayState?.status !== 'error' && homeState?.status !== 'error';

  return (
    <section
      aria-labelledby="head-to-head-heading"
      aria-busy={loading}
      className="flex min-h-0 flex-col rounded-2xl bg-surface p-4 shadow-sm ring-1 ring-separator sm:p-5"
    >
      <h2 id="head-to-head-heading" className="text-lg font-semibold leading-tight tracking-tight">
        Head to head
      </h2>
      <p className="mt-0.5 text-[0.8125rem] text-label-2">
        {serverReady
          ? "From the model's 2024–25 dataset, not every game."
          : 'Stats load once the prediction server is ready.'}
      </p>

      <div className="space-y-1 empty:hidden">
        <StatusNote team={awayTeam} state={awayState} serverReady={serverReady} onRetry={() => onRetry(awayTeam.id)} />
        <StatusNote team={homeTeam} state={homeState} serverReady={serverReady} onRetry={() => onRetry(homeTeam.id)} />
      </div>

      {/* The table stretches to fill the card; only the stat rows grow. */}
      <div className="mt-2 min-h-0 flex-1 overflow-y-auto">
        <table className="h-full w-full table-fixed text-[0.875rem]">
          <caption className="sr-only">
            {awayTeam.name} (away) compared with {homeTeam.name} (home)
          </caption>
          <colgroup>
            <col />
            <col className="w-[40%]" />
            <col />
          </colgroup>
          <thead>
            <tr className="h-8 border-b border-separator">
              <th scope="col" className="px-3 text-right font-semibold">
                <span className="inline-flex items-center gap-2">
                  <Dot color={colors.away} />
                  {awayTeam.abbreviation}
                </span>
              </th>
              <th scope="col">
                <span className="sr-only">Stat</span>
              </th>
              <th scope="col" className="px-3 text-left font-semibold">
                <span className="inline-flex items-center gap-2">
                  {homeTeam.abbreviation}
                  <Dot color={colors.home} />
                </span>
              </th>
            </tr>
          </thead>
          {GROUPS.map((group) => (
            <tbody key={group.title}>
              <tr className="h-6">
                <th colSpan={3} scope="colgroup" className="rounded-md bg-fill/70 px-3 text-center text-xs font-semibold text-label-2">
                  {group.title}
                </th>
              </tr>
              {group.rows.map((row) => {
                const leader = leaderOf(row, away, home);
                return (
                  <tr key={row.label} className="border-t border-separator first:border-t-0">
                    <td className="px-3 py-0.5 text-right">
                      <Value stats={away} row={row} leads={leader === 'away'} color={colors.away} align="right" />
                    </td>
                    <th scope="row" className="px-1 py-0.5 text-center font-normal text-label-2">
                      {row.label}
                    </th>
                    <td className="px-3 py-0.5 text-left">
                      <Value stats={home} row={row} leads={leader === 'home'} color={colors.home} align="left" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      </div>
    </section>
  );
};

export default TeamStats;
