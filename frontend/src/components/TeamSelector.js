import { ChevronDown } from 'lucide-react';
import React from 'react';

const TeamSelector = ({ label, teams, selectedTeam, onTeamSelect, disabled }) => {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-300">
        {label}
      </label>
      
      <div className="relative">
        <select
          value={selectedTeam?.id || ''}
          onChange={(e) => {
            const teamId = parseInt(e.target.value);
            const team = teams.find(t => t.id === teamId);
            onTeamSelect(team || null);
          }}
          disabled={disabled}
          className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white
                   focus:outline-none focus:ring-2 focus:ring-nba-orange focus:border-transparent
                   disabled:opacity-50 disabled:cursor-not-allowed
                   appearance-none cursor-pointer"
        >
          <option value="" className="bg-gray-800 text-white">
            Select {label.toLowerCase()}
          </option>
          {teams.map((team) => (
            <option
              key={team.id}
              value={team.id}
              className="bg-gray-800 text-white"
            >
              {team.abbreviation} - {team.name}
            </option>
          ))}
        </select>
        
        <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
      </div>
      
      {selectedTeam && (
        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-nba-orange to-nba-red rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-sm">
                {selectedTeam.abbreviation.charAt(0)}
              </span>
            </div>
            <div>
              <p className="text-white font-medium">{selectedTeam.abbreviation}</p>
              <p className="text-gray-300 text-sm">{selectedTeam.name}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamSelector;
