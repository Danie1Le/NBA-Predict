import { ArrowLeftRight } from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { checkHealth, describeError, getModels, getTeams, getTeamStats, predictGame } from './api';
import HowItWorks from './components/HowItWorks';
import LoadingSpinner from './components/LoadingSpinner';
import ModelSelector from './components/ModelSelector';
import PredictionResult from './components/PredictionResult';
import ServerStatus from './components/ServerStatus';
import TeamSelector from './components/TeamSelector';
import TeamStats from './components/TeamStats';
import { matchupColors, TEAMS, withTeamMeta } from './teams';

// The backend trains these on the first prediction, so they're always offered.
const DEFAULT_MODELS = ['xgb', 'rf', 'logreg'];
const MODEL_ORDER = ['ensemble', 'xgb', 'rf', 'logreg', 'pytorch', 'tensorflow'];

// Open on the 2025 Finals so there's something to predict right away.
const DEFAULT_AWAY = 'IND';
const DEFAULT_HOME = 'OKC';

const HEALTH_ATTEMPTS = 6;
const RETRY_DELAY_MS = 3000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const findByAbbreviation = (abbreviation) =>
  TEAMS.find((team) => team.abbreviation === abbreviation?.toUpperCase());

// Matchup and model live in the URL so a prediction can be shared or bookmarked.
const readInitialState = () => {
  const params = new URLSearchParams(window.location.search);
  let away = findByAbbreviation(params.get('away')) ?? findByAbbreviation(DEFAULT_AWAY);
  let home = findByAbbreviation(params.get('home')) ?? findByAbbreviation(DEFAULT_HOME);
  if (away.id === home.id) {
    away = findByAbbreviation(DEFAULT_AWAY);
    home = findByAbbreviation(DEFAULT_HOME);
  }
  return { awayId: away.id, homeId: home.id, model: params.get('model') };
};

function App() {
  const [initial] = useState(readInitialState);
  const [server, setServer] = useState('connecting'); // connecting | waking | ready | offline
  const [connectAttempt, setConnectAttempt] = useState(0);
  const [teams, setTeams] = useState(TEAMS);
  const [models, setModels] = useState(DEFAULT_MODELS);
  const [recommendedModel, setRecommendedModel] = useState('xgb');

  const [awayId, setAwayId] = useState(initial.awayId);
  const [homeId, setHomeId] = useState(initial.homeId);
  const [chosenModel, setChosenModel] = useState(initial.model);

  // Results are cached per away/home/model, so switching back is instant
  // and a result never shows next to inputs it wasn't made for.
  const [results, setResults] = useState({});
  const [pendingKey, setPendingKey] = useState(null);
  const [predictError, setPredictError] = useState(null);

  const [teamStats, setTeamStats] = useState({});
  const requestedStats = useRef(new Set());
  const lastResult = useRef(null);

  const findTeam = (id) => teams.find((team) => team.id === id) ?? TEAMS.find((team) => team.id === id);
  const awayTeam = findTeam(awayId);
  const homeTeam = findTeam(homeId);
  const colors = useMemo(() => matchupColors(awayTeam, homeTeam), [awayTeam, homeTeam]);

  const model = models.includes(chosenModel) ? chosenModel : recommendedModel;
  const matchupKey = `${awayId}:${homeId}`;
  const resultKey = `${matchupKey}:${model}`;
  const result = results[resultKey];
  const pending = pendingKey === resultKey;
  const serverReady = server === 'ready';

  // While another model runs for the same matchup, keep the last result on
  // screen (dimmed) instead of snapping the court back to even.
  const heldResult =
    !result && pending && lastResult.current?.matchupKey === matchupKey ? lastResult.current.result : null;

  useEffect(() => {
    if (result) lastResult.current = { matchupKey, result };
  }, [matchupKey, result]);

  // Connect: wait for /health (Render cold starts), then load teams and models.
  useEffect(() => {
    let cancelled = false;
    setServer('connecting');
    const wakingTimer = setTimeout(() => {
      if (!cancelled) setServer((state) => (state === 'connecting' ? 'waking' : state));
    }, RETRY_DELAY_MS);

    const connect = async () => {
      for (let attempt = 0; attempt < HEALTH_ATTEMPTS; attempt += 1) {
        try {
          if (await checkHealth()) {
            const [teamList, modelInfo] = await Promise.all([
              getTeams().catch(() => null),
              getModels().catch(() => null),
            ]);
            if (cancelled) return;
            if (teamList?.length) setTeams(withTeamMeta(teamList));
            if (modelInfo) {
              const available = new Set([...DEFAULT_MODELS, ...(modelInfo.available_models ?? [])]);
              setModels(MODEL_ORDER.filter((name) => available.has(name)));
              if (available.has(modelInfo.recommended_model)) setRecommendedModel(modelInfo.recommended_model);
            }
            setServer('ready');
            return;
          }
        } catch {
          // Still starting up; try again below.
        }
        if (cancelled) return;
        await sleep(RETRY_DELAY_MS);
        if (cancelled) return;
      }
      setServer('offline');
    };
    connect();

    return () => {
      cancelled = true;
      clearTimeout(wakingTimer);
    };
  }, [connectAttempt]);

  const loadStats = useCallback((teamId) => {
    requestedStats.current.add(teamId);
    setTeamStats((all) => ({ ...all, [teamId]: { status: 'loading' } }));
    getTeamStats(teamId)
      .then((data) => setTeamStats((all) => ({ ...all, [teamId]: { status: 'ready', data } })))
      .catch(() => {
        requestedStats.current.delete(teamId);
        setTeamStats((all) => ({ ...all, [teamId]: { status: 'error' } }));
      });
  }, []);

  useEffect(() => {
    if (!serverReady) return;
    [awayId, homeId].forEach((teamId) => {
      if (!requestedStats.current.has(teamId)) loadStats(teamId);
    });
  }, [serverReady, awayId, homeId, loadStats]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('away', awayTeam.abbreviation);
    params.set('home', homeTeam.abbreviation);
    if (chosenModel) params.set('model', chosenModel);
    else params.delete('model');
    window.history.replaceState(null, '', `${window.location.pathname}?${params}`);
  }, [awayTeam.abbreviation, homeTeam.abbreviation, chosenModel]);

  const predict = useCallback(async (modelType) => {
    const key = `${awayId}:${homeId}:${modelType}`;
    setPendingKey(key);
    setPredictError(null);
    try {
      const data = await predictGame({ homeTeamId: homeId, awayTeamId: awayId, model: modelType });
      setResults((all) => ({ ...all, [key]: data }));
    } catch (err) {
      console.error('Prediction error:', err);
      setPredictError({ key, message: describeError(err) });
    } finally {
      setPendingKey((current) => (current === key ? null : current));
    }
  }, [awayId, homeId]);

  // Once a matchup has been predicted, switching models answers right away.
  const handleModelSelect = (nextModel) => {
    setChosenModel(nextModel);
    const nextKey = `${matchupKey}:${nextModel}`;
    const matchupPredicted = Object.keys(results).some((key) => key.startsWith(`${matchupKey}:`));
    if (serverReady && matchupPredicted && !results[nextKey] && pendingKey !== nextKey) {
      predict(nextModel);
    }
  };

  const swapTeams = () => {
    setAwayId(homeId);
    setHomeId(awayId);
  };

  let predictLabel = 'Predict winner';
  if (pending) predictLabel = 'Predicting…';
  else if (server === 'waking') predictLabel = 'Waiting for the server…';
  else if (server === 'offline') predictLabel = 'Server unavailable';

  return (
    // On tablets and up everything fits one screen: the layout is capped to the
    // viewport height and the court and stats table flex to fill it. Phones
    // stack the two panels and scroll.
    <div className="flex min-h-dvh flex-col justify-center">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 md:h-dvh md:max-h-[60rem] md:min-h-[38rem]">
        <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 md:flex-nowrap">
          <div className="min-w-0 md:flex md:flex-1 md:items-baseline md:gap-3">
            <h1 className="text-2xl font-bold leading-tight tracking-tight md:shrink-0 md:whitespace-nowrap">
              NBA Game Predictor
            </h1>
            <p className="text-sm text-label-2 md:min-w-0 md:truncate">
              Each team's chance to win any 2024–25 matchup, from machine learning models.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <ServerStatus state={server} onRetry={() => setConnectAttempt((n) => n + 1)} />
            <HowItWorks />
          </div>
        </header>

        <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] md:grid-rows-[minmax(0,1fr)]">
          <section
            aria-labelledby="matchup-heading"
            className="flex min-h-0 flex-col gap-4 rounded-2xl bg-surface p-4 shadow-sm ring-1 ring-separator sm:p-5"
          >
            <h2 id="matchup-heading" className="sr-only">Matchup</h2>

            <div className="grid grid-cols-1 items-end gap-2 sm:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] md:grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
              <TeamSelector
                id="away-team"
                label="Away"
                teams={teams}
                selectedTeam={awayTeam}
                otherTeam={homeTeam}
                otherLabel="Home"
                color={colors.away}
                onTeamSelect={setAwayId}
              />
              <button
                type="button"
                onClick={swapTeams}
                aria-label="Swap home and away teams"
                title="Swap home and away"
                className="mx-auto flex size-11 items-center justify-center rounded-full text-label-2 transition-colors hover:bg-fill hover:text-label"
              >
                <ArrowLeftRight
                  aria-hidden="true"
                  className="size-5 rotate-90 sm:rotate-0 md:rotate-90 lg:rotate-0"
                />
              </button>
              <TeamSelector
                id="home-team"
                label="Home"
                teams={teams}
                selectedTeam={homeTeam}
                otherTeam={awayTeam}
                otherLabel="Away"
                color={colors.home}
                onTeamSelect={setHomeId}
              />
            </div>

            <div className="grid grid-cols-1 items-end gap-3 xl:grid-cols-[minmax(0,1fr)_11rem]">
              <ModelSelector
                models={models}
                selectedModel={model}
                recommendedModel={recommendedModel}
                onModelSelect={handleModelSelect}
              />
              <button
                type="button"
                onClick={() => predict(model)}
                disabled={!serverReady || pending}
                aria-busy={pending}
                className={`flex min-h-[52px] w-full items-center justify-center gap-2 rounded-xl px-4 font-semibold
                            transition-[background-color,color,transform] active:scale-[0.99] disabled:active:scale-100
                            ${serverReady
                              ? 'bg-tint text-on-tint hover:bg-tint/90 disabled:cursor-progress'
                              : 'bg-fill text-label-2 cursor-not-allowed'}`}
              >
                {(pending || server === 'waking') && <LoadingSpinner />}
                {predictLabel}
              </button>
            </div>

            <PredictionResult
              awayTeam={awayTeam}
              homeTeam={homeTeam}
              colors={colors}
              result={result ?? heldResult}
              requestedModel={model}
              stale={pending}
              error={predictError?.key === resultKey ? predictError.message : null}
              onRetry={() => predict(model)}
            />
          </section>

          <TeamStats
            awayTeam={awayTeam}
            homeTeam={homeTeam}
            colors={colors}
            awayState={teamStats[awayId]}
            homeState={teamStats[homeId]}
            serverReady={serverReady}
            onRetry={loadStats}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
