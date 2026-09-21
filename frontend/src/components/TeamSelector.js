import { ChevronDown } from 'lucide-react';
import React from 'react';

const CONFERENCES = [
  { key: 'East', label: 'Eastern Conference' },
  { key: 'West', label: 'Western Conference' },
];

const byName = (a, b) => a.name.localeCompare(b.name);

// A native select keeps the platform picker on phones and full keyboard
// support; the color chip mirrors the team's side of the court.
const TeamSelector = ({ id, label, teams, selectedTeam, otherTeam, otherLabel, color, onTeamSelect }) => {
  const grouped = CONFERENCES.map((conference) => ({
    ...conference,
    teams: teams.filter((team) => team.conference === conference.key).sort(byName),
  }));
  const ungrouped = teams.filter((team) => !CONFERENCES.some((c) => c.key === team.conference)).sort(byName);

  const renderOption = (team) => {
    const takenByOther = team.id === otherTeam?.id;
    return (
      <option key={team.id} value={team.id} disabled={takenByOther}>
        {team.name}{takenByOther ? ` (${otherLabel.toLowerCase()})` : ''}
      </option>
    );
  };

  return (
    <div className="min-w-0">
      <label htmlFor={id} className="block text-sm font-medium text-label-2 mb-1.5">
        {label}
      </label>
      <div className="relative">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 size-4 rounded-full ring-1 ring-black/10 transition-colors"
          style={{ backgroundColor: color }}
        />
        <select
          id={id}
          value={selectedTeam?.id ?? ''}
          onChange={(e) => onTeamSelect(Number(e.target.value))}
          className="w-full h-11 appearance-none truncate rounded-xl bg-fill pl-10 pr-10 text-[1.0625rem] font-medium text-label
                     cursor-pointer hover:bg-fill/70 transition-colors"
        >
          {grouped.map((conference) =>
            conference.teams.length > 0 && (
              <optgroup key={conference.key} label={conference.label}>
                {conference.teams.map(renderOption)}
              </optgroup>
            )
          )}
          {ungrouped.map(renderOption)}
        </select>
        <ChevronDown
          aria-hidden="true"
          className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 size-5 text-label-2"
        />
      </div>
    </div>
  );
};

export default TeamSelector;
