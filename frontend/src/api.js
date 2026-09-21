import axios from 'axios';

// Auto-detect environment and set API URL
const getApiUrl = () => {
  // If running locally (development)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  // Production URL
  return process.env.REACT_APP_API_URL || 'https://nba-predict-7hz6.onrender.com';
};

const client = axios.create({ baseURL: getApiUrl() });

// The free Render instance sleeps when idle, so the first health check can
// take most of a minute while it boots.
export const checkHealth = () =>
  client.get('/health', { timeout: 20000 }).then((res) => res.data.status === 'healthy');

export const getTeams = () => client.get('/teams', { timeout: 10000 }).then((res) => res.data);

export const getModels = () => client.get('/models', { timeout: 10000 }).then((res) => res.data);

export const getTeamStats = (teamId) =>
  client.get(`/team-stats/${teamId}`, { timeout: 15000 }).then((res) => res.data);

// The first prediction after a restart also trains the models, so allow extra time.
export const predictGame = ({ homeTeamId, awayTeamId, model }) =>
  client
    .post('/predict', { home_team_id: homeTeamId, away_team_id: awayTeamId, model_type: model }, { timeout: 60000 })
    .then((res) => res.data);

export const describeError = (err) => {
  if (err.code === 'ECONNABORTED') {
    return 'The server took too long to answer. It may still be starting up.';
  }
  if (!err.response) {
    return "Couldn't reach the prediction server. Check your connection.";
  }
  return `The server couldn't make this prediction (error ${err.response.status}).`;
};
