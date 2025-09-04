import { Home, Plane, RefreshCw, TrendingUp, Trophy } from 'lucide-react';
import React from 'react';

const PredictionResult = ({ prediction, onReset }) => {
  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'HIGH': return 'text-green-400';
      case 'MODERATE': return 'text-yellow-400';
      case 'LOW': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getConfidenceIcon = (confidence) => {
    switch (confidence) {
      case 'HIGH': return '🔥';
      case 'MODERATE': return '✅';
      case 'LOW': return '⚠️';
      default: return '❓';
    }
  };

  const winner = prediction.prediction === 1 
    ? { abbreviation: prediction.homeTeamName.split(' ').pop(), full_name: prediction.homeTeamName }
    : { abbreviation: prediction.awayTeamName.split(' ').pop(), full_name: prediction.awayTeamName };
  const winnerProb = prediction.prediction === 1 
    ? prediction.home_win_probability 
    : prediction.away_win_probability;

  return (
    <div className="glass-effect rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-white flex items-center">
          <Trophy className="h-6 w-6 mr-3 text-nba-orange" />
          Prediction Result
        </h2>
        <button
          onClick={onReset}
          className="p-2 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className="h-5 w-5" />
        </button>
      </div>

      {/* Winner Display */}
      <div className="text-center mb-6">
        <div className="bg-gradient-to-r from-nba-orange to-nba-red rounded-full p-1 mb-4">
          <div className="bg-gray-900 rounded-full p-4">
            <div className="flex items-center justify-center space-x-3">
              <div className="text-4xl">
                {prediction.prediction === 1 ? '🏠' : '✈️'}
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white">
                  {winner.abbreviation} WINS!
                </h3>
                <p className="text-gray-300">
                  {winner.full_name}
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="text-3xl font-bold text-nba-orange">
          {(winnerProb * 100).toFixed(1)}%
        </div>
        <p className="text-gray-300">Confidence</p>
      </div>

      {/* Probability Breakdown */}
      <div className="space-y-4 mb-6">
        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center space-x-3">
            <Home className="h-5 w-5 text-blue-400" />
            <span className="text-white font-medium">{prediction.homeTeamName.split(' ').pop()}</span>
          </div>
          <div className="text-right">
            <div className="text-lg font-semibold text-white">
              {(prediction.home_win_probability * 100).toFixed(1)}%
            </div>
            <div className="w-20 bg-gray-600 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${prediction.home_win_probability * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center space-x-3">
            <Plane className="h-5 w-5 text-red-400" />
            <span className="text-white font-medium">{prediction.awayTeamName.split(' ').pop()}</span>
          </div>
          <div className="text-right">
            <div className="text-lg font-semibold text-white">
              {(prediction.away_win_probability * 100).toFixed(1)}%
            </div>
            <div className="w-20 bg-gray-600 rounded-full h-2">
              <div 
                className="bg-red-500 h-2 rounded-full"
                style={{ width: `${prediction.away_win_probability * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="border-t border-white/20 pt-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">Model Used:</span>
          <span className="text-white font-medium capitalize">
            {prediction.model_used}
          </span>
        </div>
        
        <div className="flex items-center justify-between text-sm mt-2">
          <span className="text-gray-400">Confidence Level:</span>
          <span className={`font-medium flex items-center space-x-1 ${getConfidenceColor(prediction.confidence)}`}>
            <span>{getConfidenceIcon(prediction.confidence)}</span>
            <span>{prediction.confidence}</span>
          </span>
        </div>
      </div>

      {/* Performance Stats */}
      <div className="mt-4 p-3 bg-green-500/20 border border-green-500/50 rounded-lg">
        <div className="flex items-center space-x-2 text-green-200">
          <TrendingUp className="h-4 w-4" />
          <span className="text-sm font-medium">Model Performance</span>
        </div>
        <p className="text-xs text-green-300 mt-1">
          78.5% Accuracy • 0.879 AUC • 87.3% High Confidence Accuracy
        </p>
      </div>
    </div>
  );
};

export default PredictionResult;
