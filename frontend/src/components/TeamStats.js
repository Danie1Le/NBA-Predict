import { BarChart3, Target, TrendingUp, Zap } from 'lucide-react';
import React from 'react';

const TeamStats = ({ teamStats, teamName, isHome }) => {
  if (!teamStats) {
    return (
      <div className={`p-4 rounded-lg border-2 ${
        isHome 
          ? 'border-blue-500/50 bg-blue-500/10' 
          : 'border-red-500/50 bg-red-500/10'
      }`}>
        <div className="text-center text-gray-400">
          <BarChart3 className="h-8 w-8 mx-auto mb-2" />
          <p>Loading stats...</p>
        </div>
      </div>
    );
  }

  const formatPercentage = (value) => `${(value * 100).toFixed(1)}%`;
  const formatNumber = (value) => value ? value.toFixed(1) : '0.0';

  return (
    <div className={`p-4 rounded-lg border-2 ${
      isHome 
        ? 'border-blue-500/50 bg-blue-500/10' 
        : 'border-red-500/50 bg-red-500/10'
    }`}>
      <div className="flex items-center space-x-3 mb-4">
        <div className={`p-2 rounded-lg ${
          isHome ? 'bg-blue-500/30' : 'bg-red-500/30'
        }`}>
          <BarChart3 className={`h-5 w-5 ${
            isHome ? 'text-blue-400' : 'text-red-400'
          }`} />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">
            {teamName}
          </h3>
          <p className="text-sm text-gray-400">
            {isHome ? 'Home Team' : 'Away Team'}
          </p>
        </div>
      </div>

      {/* Records */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className="text-xs text-gray-400 mb-1">Last 5</div>
          <div className="text-sm font-bold text-white">
            {teamStats.last_5_wins}-{teamStats.last_5_games - teamStats.last_5_wins}
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className="text-xs text-gray-400 mb-1">Last 10</div>
          <div className="text-sm font-bold text-white">
            {teamStats.last_10_wins}-{teamStats.last_10_games - teamStats.last_10_wins}
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className="text-xs text-gray-400 mb-1">Season</div>
          <div className="text-sm font-bold text-white">
            {teamStats.season_wins}-{teamStats.season_games - teamStats.season_wins}
          </div>
        </div>
      </div>

      {/* Playoff Record - Not available in current API */}

      {/* Recent Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Zap className="h-4 w-4 text-yellow-400" />
            <span className="text-xs text-gray-400">Points (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatNumber(teamStats.last_5_pts)}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Target className="h-4 w-4 text-green-400" />
            <span className="text-xs text-gray-400">FG% (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatPercentage(teamStats.last_5_fg_pct)}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <TrendingUp className="h-4 w-4 text-blue-400" />
            <span className="text-xs text-gray-400">3P% (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatPercentage(teamStats.last_5_fg3_pct)}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <BarChart3 className="h-4 w-4 text-purple-400" />
            <span className="text-xs text-gray-400">Rebounds (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatNumber(teamStats.last_5_reb)}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Zap className="h-4 w-4 text-cyan-400" />
            <span className="text-xs text-gray-400">Assists (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatNumber(teamStats.last_5_ast)}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Target className="h-4 w-4 text-orange-400" />
            <span className="text-xs text-gray-400">Turnovers (5g)</span>
          </div>
          <div className="text-lg font-bold text-white">
            {formatNumber(teamStats.last_5_tov)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamStats;
