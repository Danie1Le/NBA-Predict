import axios from 'axios';
import { BarChart3, Circle, Target, TrendingUp, Users, Zap } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import LoadingSpinner from './components/LoadingSpinner';
import ModelSelector from './components/ModelSelector';
import PredictionResult from './components/PredictionResult';
import TeamSelector from './components/TeamSelector';
import TeamStats from './components/TeamStats';

// Auto-detect environment and set API URL
const getApiUrl = () => {
  // If running locally (development)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  // Production URL
  return process.env.REACT_APP_API_URL || 'https://nba-predict-7hz6.onrender.com';
};

const API_BASE_URL = getApiUrl();

// Debug logging

function App() {
  const [teams, setTeams] = useState([]);
  const [selectedHomeTeam, setSelectedHomeTeam] = useState(null);
  const [selectedAwayTeam, setSelectedAwayTeam] = useState(null);
  const [selectedModel, setSelectedModel] = useState('xgb');
  const [availableModels, setAvailableModels] = useState(['xgb', 'rf', 'logreg']); // Always include logistic regression
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [homeTeamStats, setHomeTeamStats] = useState(null);
  const [awayTeamStats, setAwayTeamStats] = useState(null);


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

  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000 // 5 second timeout for health check
      });
      return response.data.status === 'healthy';
    } catch (err) {
      console.warn('Backend health check failed:', err.message);
      return false;
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const modelsResponse = await axios.get(`${API_BASE_URL}/models`, {
        timeout: 10000 // 10 second timeout
      });
      const backendModels = modelsResponse.data.available_models || [];
      
      // Always ensure logistic regression is available
      const modelsWithLogReg = [...new Set([...backendModels, 'logreg'])];
      
      setAvailableModels(prevModels => {
        // If we now have more models, show a notification
        if (modelsWithLogReg.length > prevModels.length) {
          // Could show notification here if needed
        }
        return modelsWithLogReg;
      });
    } catch (err) {
      console.error('Error loading models:', err);
      // Fallback to default models including logistic regression
      setAvailableModels(['xgb', 'rf', 'logreg']);
    }
  }, []);

  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Check backend health first
      const isHealthy = await checkBackendHealth();
      if (!isHealthy) {
        throw new Error('Backend is not responding to health checks');
      }
      // Reduced timeout for faster failure detection
      const axiosConfig = {
        timeout: 8000, // 8 second timeout
        headers: {
          'Content-Type': 'application/json'
        }
      };
      
      // Load teams first (fast), then models (may be slower)
      const teamsResponse = await axios.get(`${API_BASE_URL}/teams`, axiosConfig);
      setTeams(teamsResponse.data);
      
      // Load models with separate timeout handling
      try {
        const modelsResponse = await axios.get(`${API_BASE_URL}/models`, axiosConfig);
        const backendModels = modelsResponse.data.available_models || [];
        setAvailableModels(backendModels);
        
        // If deep learning models are not available yet, check again in 30 seconds
        const hasDeepLearning = backendModels.some(model => 
          ['pytorch', 'tensorflow', 'ensemble'].includes(model)
        );
        
        if (!hasDeepLearning && backendModels.length > 0) {
          setTimeout(() => {
            loadModels(); // Check for updated models
          }, 30000);
        }
      } catch (modelsErr) {
        // Fallback to basic models if models endpoint fails
        setAvailableModels(['xgb']);
      }
      
    } catch (err) {
      console.error('Error loading initial data:', err);
      
      // Show more specific error messages
      if (err.code === 'ECONNABORTED') {
        setError('Backend is taking longer than expected to respond. The service might be starting up. Please wait a moment and try again.');
      } else if (err.response?.status === 404) {
        setError('Backend endpoint not found. Please check if the backend is deployed correctly.');
      } else if (err.response?.status >= 500) {
        setError('Backend server error. The service might be starting up. Please wait a moment and try again.');
      } else {
        setError('Failed to load initial data. Please check your connection and try again.');
      }
    } finally {
      setLoading(false);
    }
  }, [loadModels, checkBackendHealth]);

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
      // Set empty stats to show error state
      if (type === 'home') {
        setHomeTeamStats(null);
      } else {
        setAwayTeamStats(null);
      }
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
      
      // Add timeout for better UX
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        home_team_id: selectedHomeTeam.id,
        away_team_id: selectedAwayTeam.id,
        model_type: selectedModel
      }, {
        timeout: 30000 // 30 second timeout
      });
      
      setPrediction({
        ...response.data,
        homeTeamName: selectedHomeTeam.full_name,
        awayTeamName: selectedAwayTeam.full_name
      });
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Prediction is taking longer than expected. The AI is working hard! Please try again.');
      } else {
        setError('Prediction failed. Please try again.');
      }
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  // Load initial data after functions are defined
  useEffect(() => {
    loadInitialData();
    
    // Retry loading data with exponential backoff if it failed
    let retryCount = 0;
    const maxRetries = 5;
    
    const retryInterval = setInterval(() => {
      if (teams.length === 0) {
        retryCount++;
        if (retryCount <= maxRetries) {
          loadInitialData();
        } else {
          clearInterval(retryInterval);
        }
      } else {
        // Success - clear the interval
        clearInterval(retryInterval);
      }
    }, Math.min(10000 * Math.pow(2, retryCount), 60000)); // Exponential backoff, max 60s
    
    return () => clearInterval(retryInterval);
  }, [teams.length, loadInitialData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 flex flex-col">
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
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1">
        {loading && !prediction && (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner 
              message="Loading NBA Predictor..." 
              subMessage="Traditional models ready, deep learning models training in background"
            />
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
            <button
              onClick={() => {
                setError(null);
                loadInitialData();
              }}
              className="mt-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
            >
              Retry Connection
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Column - Model Selection (Smaller) */}
          <div className="lg:col-span-1">
            <div className="glass-effect rounded-xl p-4">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Target className="h-5 w-5 mr-2 text-nba-orange" />
                Model
              </h2>
              
              <ModelSelector
                models={availableModels}
                selectedModel={selectedModel}
                onModelSelect={setSelectedModel}
                disabled={loading}
              />
            </div>
          </div>

          {/* Middle Column - Team Selection */}
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

              {/* Team Statistics - Right under team selection */}
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2 text-nba-orange" />
                  Team Statistics
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {homeTeamStats && selectedHomeTeam ? (
                    <TeamStats
                      teamStats={homeTeamStats}
                      teamName={selectedHomeTeam.name}
                      isHome={true}
                    />
                  ) : selectedHomeTeam ? (
                    <div className="p-4 rounded-lg border-2 border-blue-500/50 bg-blue-500/10">
                      <div className="text-center text-gray-400">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                        <p>Loading {selectedHomeTeam.name} stats...</p>
                      </div>
                    </div>
                  ) : (
                    <div className="p-4 rounded-lg border-2 border-gray-500/50 bg-gray-500/10">
                      <div className="text-center text-gray-400">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                        <p>Select Home Team</p>
                      </div>
                    </div>
                  )}
                  
                  {awayTeamStats && selectedAwayTeam ? (
                    <TeamStats
                      teamStats={awayTeamStats}
                      teamName={selectedAwayTeam.name}
                      isHome={false}
                    />
                  ) : selectedAwayTeam ? (
                    <div className="p-4 rounded-lg border-2 border-red-500/50 bg-red-500/10">
                      <div className="text-center text-gray-400">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                        <p>Loading {selectedAwayTeam.name} stats...</p>
                      </div>
                    </div>
                  ) : (
                    <div className="p-4 rounded-lg border-2 border-gray-500/50 bg-gray-500/10">
                      <div className="text-center text-gray-400">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                        <p>Select Away Team</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

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
      <footer className="glass-effect border-t border-white/20 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-300">
            <p>Powered by Machine Learning • XGBoost • Random Forest • Logistic Regression</p>
            <p className="text-sm mt-2">78.6% Accuracy • 0.837 AUC</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
