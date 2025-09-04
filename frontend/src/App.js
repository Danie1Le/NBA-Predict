import axios from 'axios';
import { BarChart3, Circle, Target, TrendingUp, Users, Zap } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import LoadingSpinner from './components/LoadingSpinner';
import ModelSelector from './components/ModelSelector';
import PredictionResult from './components/PredictionResult';
import TeamSelector from './components/TeamSelector';
import TeamStats from './components/TeamStats';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://nba-predict-7hz6.onrender.com';

function App() {
  const [teams, setTeams] = useState([]);
  const [selectedHomeTeam, setSelectedHomeTeam] = useState(null);
  const [selectedAwayTeam, setSelectedAwayTeam] = useState(null);
  const [selectedModel, setSelectedModel] = useState('xgb');
  const [availableModels, setAvailableModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [homeTeamStats, setHomeTeamStats] = useState(null);
  const [awayTeamStats, setAwayTeamStats] = useState(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (selectedHomeTeam) {
      loadTeamStats(selectedHomeTeam.id, 'home');
    }
  }, [selectedHomeTeam]);

  useEffect(() => {
    if (selectedAwayTeam) {
      loadTeamStats(selectedAwayTeam.id, 'away');
    }
  }, [selectedAwayTeam]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [teamsResponse, modelsResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/teams`),
        axios.get(`${API_BASE_URL}/models`)
      ]);
      
      setTeams(teamsResponse.data);
      // Show all models by default, even if backend doesn't have them yet
      const allModels = ['xgb', 'rf', 'logreg', 'pytorch', 'tensorflow', 'ensemble'];
      setAvailableModels(allModels);
    } catch (err) {
      setError('Failed to load initial data. Make sure the backend is running.');
      console.error('Error loading initial data:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadTeamStats = async (teamId, type) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/team-stats/${teamId}`);
      if (type === 'home') {
        setHomeTeamStats(response.data);
      } else {
        setAwayTeamStats(response.data);
      }
    } catch (err) {
      console.error(`Error loading ${type} team stats:`, err);
    }
  };

  const handlePredict = async () => {
    if (!selectedHomeTeam || !selectedAwayTeam) {
      setError('Please select both home and away teams');
      return;
    }

    if (selectedHomeTeam.id === selectedAwayTeam.id) {
      setError('Home and away teams must be different');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        home_team_id: selectedHomeTeam.id,
        away_team_id: selectedAwayTeam.id,
        model_type: selectedModel
      });
      
      setPrediction({
        ...response.data,
        homeTeamName: selectedHomeTeam.full_name,
        awayTeamName: selectedAwayTeam.full_name
      });
    } catch (err) {
      setError('Prediction failed. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      {/* Header */}
      <header className="glass-effect border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-center space-x-4">
            <Circle className="h-12 w-12 text-nba-orange animate-float" />
            <div className="text-center">
              <h1 className="text-4xl font-bold text-white gradient-text">
                NBA Game Predictor
              </h1>
              <p className="text-lg text-gray-300 mt-2">
                AI-Powered Basketball Game Predictions
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && !prediction && (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner />
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Team Selection */}
          <div className="lg:col-span-2 space-y-6">
            {/* Team Selection */}
            <div className="glass-effect rounded-xl p-6">
              <h2 className="text-2xl font-semibold text-white mb-6 flex items-center">
                <Users className="h-6 w-6 mr-3 text-nba-orange" />
                Team Selection
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <TeamSelector
                  label="Home Team"
                  teams={teams}
                  selectedTeam={selectedHomeTeam}
                  onTeamSelect={setSelectedHomeTeam}
                  disabled={loading}
                />
                
                <TeamSelector
                  label="Away Team"
                  teams={teams}
                  selectedTeam={selectedAwayTeam}
                  onTeamSelect={setSelectedAwayTeam}
                  disabled={loading}
                />
              </div>
            </div>

            {/* Model Selection */}
            <div className="glass-effect rounded-xl p-6">
              <h2 className="text-2xl font-semibold text-white mb-6 flex items-center">
                <Target className="h-6 w-6 mr-3 text-nba-orange" />
                Model Selection
              </h2>
              
              <ModelSelector
                models={availableModels}
                selectedModel={selectedModel}
                onModelSelect={setSelectedModel}
                disabled={loading}
              />
            </div>

            {/* Team Statistics */}
            {(homeTeamStats || awayTeamStats) && (
              <div className="glass-effect rounded-xl p-6">
                <h2 className="text-2xl font-semibold text-white mb-6 flex items-center">
                  <BarChart3 className="h-6 w-6 mr-3 text-nba-orange" />
                  Team Statistics
                </h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {homeTeamStats && selectedHomeTeam && (
                    <TeamStats
                      teamStats={homeTeamStats}
                      teamName={selectedHomeTeam.full_name}
                      isHome={true}
                    />
                  )}
                  
                  {awayTeamStats && selectedAwayTeam && (
                    <TeamStats
                      teamStats={awayTeamStats}
                      teamName={selectedAwayTeam.full_name}
                      isHome={false}
                    />
                  )}
                </div>
              </div>
            )}

          </div>

          {/* Right Column - Prediction Results */}
          <div className="lg:col-span-1 space-y-6">
            {prediction && (
              <PredictionResult
                prediction={prediction}
                onReset={resetPrediction}
              />
            )}
            
            {!prediction && (
              <div className="glass-effect rounded-xl p-6 text-center">
                <TrendingUp className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  Ready to Predict
                </h3>
                <p className="text-gray-300">
                  Select teams and click predict to see the AI's prediction
                </p>
              </div>
            )}
            
            {/* Predict Button */}
            <div className="flex justify-center">
              <button
                onClick={handlePredict}
                disabled={loading || !selectedHomeTeam || !selectedAwayTeam}
                className="bg-gradient-to-r from-nba-orange to-nba-red hover:from-orange-600 hover:to-red-600 
                         disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed
                         text-white font-bold py-4 px-8 rounded-full text-lg
                         transform hover:scale-105 transition-all duration-200
                         flex items-center space-x-3 shadow-lg"
              >
                <Zap className="h-6 w-6" />
                <span>Predict Game Outcome</span>
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="glass-effect border-t border-white/20 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-300">
            <p>Powered by Machine Learning • XGBoost • PyTorch • TensorFlow</p>
            <p className="text-sm mt-2">78.5% Accuracy • 0.879 AUC</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
