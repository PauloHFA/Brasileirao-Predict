import React, { useState, useEffect } from 'react';
import { MatchItem, PredictionResult, ExplanationResult, ExperimentResults } from './types';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState<'simulator' | 'experiments'>('simulator');
  const [season, setSeason] = useState<number>(2024);
  const [round, setRound] = useState<number>(1);
  const [matches, setMatches] = useState<MatchItem[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<MatchItem | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('numeric_lgbm');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [evidenceData, setEvidenceData] = useState<any>(null);
  const [experimentsData, setExperimentsData] = useState<ExperimentResults | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingExp, setLoadingExp] = useState<boolean>(false);

  // Fetch matches when season or round changes
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/matches?season=${season}&round=${round}`)
      .then((res) => res.json())
      .then((data: MatchItem[]) => {
        setMatches(data);
        if (data.length > 0) {
          setSelectedMatch(data[0]);
        }
      })
      .catch((err) => console.error('Error fetching matches:', err));
  }, [season, round]);

  // Fetch prediction and evidence when selectedMatch or selectedModel changes
  useEffect(() => {
    if (!selectedMatch) return;
    setLoading(true);
    setExplanation(null);

    // Fetch prediction simulation
    fetch(`${API_BASE}/api/v1/predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: selectedMatch.match_id,
        model_type: selectedModel,
        as_of_hours: 24,
      }),
    })
      .then((res) => res.json())
      .then((data: PredictionResult) => {
        setPrediction(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error simulating prediction:', err);
        setLoading(false);
      });

    // Fetch evidence
    fetch(`${API_BASE}/api/v1/matches/${selectedMatch.match_id}/evidence`)
      .then((res) => res.json())
      .then((data) => setEvidenceData(data))
      .catch((err) => console.error('Error fetching evidence:', err));
  }, [selectedMatch, selectedModel]);

  // Fetch experiments report when tab opened
  useEffect(() => {
    if (activeTab === 'experiments') {
      fetch(`${API_BASE}/api/v1/experiments`)
        .then((res) => res.json())
        .then((data: ExperimentResults) => setExperimentsData(data))
        .catch((err) => console.error('Error fetching experiments:', err));
    }
  }, [activeTab]);

  const handleGenerateExplanation = () => {
    if (!selectedMatch) return;
    setLoadingExp(true);
    fetch(`${API_BASE}/api/v1/explanations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ match_id: selectedMatch.match_id }),
    })
      .then((res) => res.json())
      .then((data: ExplanationResult) => {
        setExplanation(data);
        setLoadingExp(false);
      })
      .catch((err) => {
        console.error('Error generating explanation:', err);
        setLoadingExp(false);
      });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚽</span>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Brasileirão Predictor AI
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                v1.0 Produção
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Fusão de estatísticas tabulares e extração estruturada de contexto esportivo por LLM
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => setActiveTab('simulator')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition ${
              activeTab === 'simulator'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Simulador de Rodada
          </button>
          <button
            onClick={() => setActiveTab('experiments')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition ${
              activeTab === 'experiments'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Experimentos & Governança
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto">
        {activeTab === 'simulator' ? (
          <div className="space-y-6">
            {/* Filter Bar */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Temporada</label>
                  <select
                    value={season}
                    onChange={(e) => setSeason(Number(e.target.value))}
                    className="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-1.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={2024}>2024 (Teste Fechado)</option>
                    <option value={2023}>2023 (Validação/Calibração)</option>
                    <option value={2022}>2022 (Treino)</option>
                    <option value={2021}>2021 (Treino)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Rodada</label>
                  <select
                    value={round}
                    onChange={(e) => setRound(Number(e.target.value))}
                    className="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-1.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {Array.from({ length: 38 }, (_, i) => i + 1).map((r) => (
                      <option key={r} value={r}>
                        Rodada {r}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Model Selector */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Modelo Preditivo Ativo</label>
                <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700 gap-1 text-xs">
                  <button
                    onClick={() => setSelectedModel('numeric_lgbm')}
                    className={`px-3 py-1 rounded font-medium transition ${
                      selectedModel === 'numeric_lgbm' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    B1: Numérico (Padrão)
                  </button>
                  <button
                    onClick={() => setSelectedModel('contextual_lgbm')}
                    className={`px-3 py-1 rounded font-medium transition ${
                      selectedModel === 'contextual_lgbm' ? 'bg-purple-600 text-white' : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    M1: Contextual LLM
                  </button>
                  <button
                    onClick={() => setSelectedModel('poisson')}
                    className={`px-3 py-1 rounded font-medium transition ${
                      selectedModel === 'poisson' ? 'bg-emerald-600 text-white' : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    G1: Poisson Placares
                  </button>
                  <button
                    onClick={() => setSelectedModel('prior')}
                    className={`px-3 py-1 rounded font-medium transition ${
                      selectedModel === 'prior' ? 'bg-slate-700 text-white' : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    B0: Prior Ingênuo
                  </button>
                </div>
              </div>
            </div>

            {/* Split View: Matches & Match Prediction Details */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Match List */}
              <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col h-[700px]">
                <h3 className="font-semibold text-sm text-slate-300 mb-3 flex items-center justify-between">
                  <span>Jogos da Rodada {round}</span>
                  <span className="text-xs text-slate-500">{matches.length} partidas</span>
                </h3>
                <div className="overflow-y-auto space-y-2 flex-1 pr-1">
                  {matches.map((m) => {
                    const isSelected = selectedMatch?.match_id === m.match_id;
                    return (
                      <div
                        key={m.match_id}
                        onClick={() => setSelectedMatch(m)}
                        className={`p-3 rounded-lg border cursor-pointer transition ${
                          isSelected
                            ? 'bg-blue-950/60 border-blue-500 shadow-sm'
                            : 'bg-slate-800/60 border-slate-700/60 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-xs text-slate-400 mb-1 flex justify-between">
                          <span>{new Date(m.kickoff_at).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                          <span className="font-mono text-emerald-400">{m.status}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm font-semibold">
                          <span className="truncate flex-1">{m.home_team_name}</span>
                          <span className="px-2 py-0.5 mx-2 bg-slate-950 rounded text-slate-200 font-mono text-xs">
                            {m.goals_home !== null ? `${m.goals_home} x ${m.goals_away}` : 'vs'}
                          </span>
                          <span className="truncate flex-1 text-right">{m.away_team_name}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Right Column: Prediction & Context Visualizer */}
              <div className="lg:col-span-8 space-y-6">
                {selectedMatch && prediction && (
                  <>
                    {/* Match Card & Probability Bar */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-4 mb-4 pb-4 border-b border-slate-800">
                        <div>
                          <div className="text-xs text-slate-400 mb-1">
                            Corte Temporal Pré-Jogo: <span className="font-mono text-amber-400">{new Date(prediction.as_of_at).toLocaleString('pt-BR')} (T-24h)</span>
                          </div>
                          <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            {selectedMatch.home_team_name} <span className="text-slate-500 font-normal">vs</span> {selectedMatch.away_team_name}
                          </h2>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-slate-400">Modelo em execução</div>
                          <div className="text-xs font-mono font-bold text-blue-400 uppercase">{prediction.model_id}</div>
                        </div>
                      </div>

                      {/* Probabilities 1X2 Card */}
                      <div className="mb-6">
                        <div className="text-xs font-medium text-slate-400 mb-2 flex justify-between">
                          <span>Probabilidade 1X2 (Vitória Mandante / Empate / Vitória Visitante)</span>
                          <span className="text-slate-500">Soma: 100%</span>
                        </div>
                        <div className="grid grid-cols-3 gap-3 mb-2 text-center">
                          <div className="bg-blue-950/40 border border-blue-800/50 p-3 rounded-lg">
                            <div className="text-xs text-slate-400">Vitória {selectedMatch.home_team_name}</div>
                            <div className="text-2xl font-bold text-blue-400 font-mono">
                              {(prediction.probabilities.home * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-amber-950/40 border border-amber-800/50 p-3 rounded-lg">
                            <div className="text-xs text-slate-400">Empate</div>
                            <div className="text-2xl font-bold text-amber-400 font-mono">
                              {(prediction.probabilities.draw * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-emerald-950/40 border border-emerald-800/50 p-3 rounded-lg">
                            <div className="text-xs text-slate-400">Vitória {selectedMatch.away_team_name}</div>
                            <div className="text-2xl font-bold text-emerald-400 font-mono">
                              {(prediction.probabilities.away * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Stacked bar visualization */}
                        <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex shadow-inner">
                          <div
                            style={{ width: `${prediction.probabilities.home * 100}%` }}
                            className="bg-blue-500 h-full transition-all duration-500"
                            title={`Mandante: ${(prediction.probabilities.home * 100).toFixed(1)}%`}
                          />
                          <div
                            style={{ width: `${prediction.probabilities.draw * 100}%` }}
                            className="bg-amber-500 h-full transition-all duration-500"
                            title={`Empate: ${(prediction.probabilities.draw * 100).toFixed(1)}%`}
                          />
                          <div
                            style={{ width: `${prediction.probabilities.away * 100}%` }}
                            className="bg-emerald-500 h-full transition-all duration-500"
                            title={`Visitante: ${(prediction.probabilities.away * 100).toFixed(1)}%`}
                          />
                        </div>
                      </div>

                      {/* Top Scores Poisson */}
                      {prediction.score_model && (
                        <div>
                          <div className="text-xs font-medium text-slate-400 mb-2">
                            Placares Mais Prováveis (Modelo de Poisson G1):
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {prediction.score_model.top_scores.map((sc, idx) => (
                              <div
                                key={idx}
                                className="bg-slate-800 border border-slate-700/80 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs"
                              >
                                <span className="font-mono font-bold text-slate-100">
                                  {sc.home} - {sc.away}
                                </span>
                                <span className="text-blue-400 font-mono">
                                  {(sc.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Context & Verified Evidence Panel */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-md font-semibold text-slate-200 flex items-center gap-2">
                          <span>Contexto Jornalístico Auditado & Evidências</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                            {evidenceData?.articles_count || 0} fontes válidas antes do corte
                          </span>
                        </h3>
                        <button
                          onClick={handleGenerateExplanation}
                          disabled={loadingExp}
                          className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition flex items-center gap-1.5 shadow"
                        >
                          {loadingExp ? 'Gerando Análise...' : '💡 Explicar Previsão (RAG)'}
                        </button>
                      </div>

                      {/* Articles list */}
                      {evidenceData?.articles && evidenceData.articles.length > 0 ? (
                        <div className="space-y-3">
                          {evidenceData.articles.map((art: any, i: number) => (
                            <div key={i} className="bg-slate-800/70 border border-slate-700 rounded-lg p-3 text-xs">
                              <div className="flex items-center justify-between font-semibold text-slate-200 mb-1">
                                <a href={art.canonical_url} target="_blank" rel="noreferrer" className="hover:underline text-blue-400">
                                  {art.title}
                                </a>
                                <span className="text-slate-400 font-mono">{art.source_name}</span>
                              </div>
                              {art.evidence_spans && art.evidence_spans.length > 0 && (
                                <div className="mt-2 pl-3 border-l-2 border-purple-500 space-y-1">
                                  {art.evidence_spans.map((ev: any, j: number) => (
                                    <div key={j} className="text-slate-300">
                                      <span className="font-mono text-purple-400 font-bold">[{ev.id}]</span>{' '}
                                      <span className="text-slate-400 italic">"{ev.quote}"</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-slate-400 bg-slate-800/40 p-3 rounded-lg border border-slate-800">
                          Nenhuma notícia esportiva com evidências cadastrada para este confronto antes do corte temporal.
                          O modelo opera puramente sobre os fatores estatísticos e histórico de forma/Elo.
                        </div>
                      )}

                      {/* Explanation report box */}
                      {explanation && (
                        <div className="mt-4 p-4 rounded-xl bg-blue-950/40 border border-blue-800/60 text-xs space-y-3">
                          <h4 className="font-bold text-blue-300 text-sm">Relatório Explicativo RAG:</h4>
                          <div className="whitespace-pre-line text-slate-200 leading-relaxed font-sans">
                            {explanation.explanation}
                          </div>
                          <div className="text-[11px] text-amber-400/90 bg-amber-950/30 p-2 rounded border border-amber-900/40">
                            {explanation.disclaimer}
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Tab 2: Experiments & Scientific Governance */
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h2 className="text-lg font-bold text-white mb-2">Relatório Experimental & Governança da Hipótese Nula (H₀)</h2>
              <p className="text-xs text-slate-400 mb-6">
                Avaliação fora da amostra no conjunto de teste fechado (Temporada 2024, 380 partidas) utilizando validação walk-forward cronológica.
              </p>

              {/* Promotion Gate Status Card */}
              {experimentsData?.latest_experiment?.promotion_gate && (
                <div className="p-4 rounded-xl bg-slate-800/80 border border-slate-700 mb-6 flex items-start gap-4">
                  <div className="text-3xl">⚖️</div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Decisão do Gate de Promoção:</span>
                      <span className="px-2 py-0.5 rounded text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                        {experimentsData.latest_experiment.promotion_gate.decision}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 mt-1">
                      {experimentsData.latest_experiment.promotion_gate.explanation}
                    </p>
                    <div className="text-xs text-slate-400 mt-2">
                      Modelo ativo em produção: <span className="font-mono text-emerald-400 font-bold">{experimentsData.latest_experiment.promotion_gate.active_model}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Metrics Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-3 px-4 font-semibold">Modelo</th>
                      <th className="py-3 px-4 font-semibold">Descrição</th>
                      <th className="py-3 px-4 font-semibold font-mono">Log Loss (Primária ↓)</th>
                      <th className="py-3 px-4 font-semibold font-mono">Brier Score (↓)</th>
                      <th className="py-3 px-4 font-semibold font-mono">ECE (Calibração ↓)</th>
                      <th className="py-3 px-4 font-semibold font-mono">Acurácia (↑)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {experimentsData?.latest_experiment?.metrics && (
                      <>
                        <tr className="hover:bg-slate-800/30">
                          <td className="py-3 px-4 font-bold text-slate-300">B0: Prior Simples</td>
                          <td className="py-3 px-4 text-slate-400">Frequência histórica pura de classes</td>
                          <td className="py-3 px-4 font-mono font-bold text-blue-400">{experimentsData.latest_experiment.metrics.B0_prior?.log_loss}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B0_prior?.brier_score}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B0_prior?.ece}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{(experimentsData.latest_experiment.metrics.B0_prior?.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                        <tr className="hover:bg-slate-800/30 bg-blue-950/20">
                          <td className="py-3 px-4 font-bold text-blue-300">B1: Numérico (LightGBM)</td>
                          <td className="py-3 px-4 text-slate-400">Elo, forma 5 jogos, saldo, descanso, casa</td>
                          <td className="py-3 px-4 font-mono font-bold text-emerald-400">{experimentsData.latest_experiment.metrics.B1_numeric?.log_loss}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B1_numeric?.brier_score}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B1_numeric?.ece}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{(experimentsData.latest_experiment.metrics.B1_numeric?.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                        <tr className="hover:bg-slate-800/30">
                          <td className="py-3 px-4 font-bold text-slate-300">B2: Volume de Notícias</td>
                          <td className="py-3 px-4 text-slate-400">B1 + contagem bruta de matérias</td>
                          <td className="py-3 px-4 font-mono font-bold text-blue-400">{experimentsData.latest_experiment.metrics.B2_news_count?.log_loss}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B2_news_count?.brier_score}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.B2_news_count?.ece}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{(experimentsData.latest_experiment.metrics.B2_news_count?.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                        <tr className="hover:bg-slate-800/30 bg-purple-950/20">
                          <td className="py-3 px-4 font-bold text-purple-300">M1: Contextual LLM</td>
                          <td className="py-3 px-4 text-slate-400">B1 + rubricas de crise, desfalques, rodízio e missing</td>
                          <td className="py-3 px-4 font-mono font-bold text-purple-400">{experimentsData.latest_experiment.metrics.M1_contextual?.log_loss}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.M1_contextual?.brier_score}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.M1_contextual?.ece}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{(experimentsData.latest_experiment.metrics.M1_contextual?.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                        <tr className="hover:bg-slate-800/30">
                          <td className="py-3 px-4 font-bold text-slate-300">G1: Poisson de Gols</td>
                          <td className="py-3 px-4 text-slate-400">Ataque/defesa e distribuição de placares</td>
                          <td className="py-3 px-4 font-mono font-bold text-blue-400">{experimentsData.latest_experiment.metrics.G1_poisson?.log_loss}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.G1_poisson?.brier_score}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{experimentsData.latest_experiment.metrics.G1_poisson?.ece}</td>
                          <td className="py-3 px-4 font-mono text-slate-300">{(experimentsData.latest_experiment.metrics.G1_poisson?.accuracy * 100).toFixed(1)}%</td>
                        </tr>
                      </>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Bootstrap Confidence Interval Box */}
              {experimentsData?.latest_experiment?.paired_bootstrap && (
                <div className="mt-6 p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-xs">
                  <h4 className="font-bold text-slate-200 mb-1">Teste de Hipótese Pareado (Bootstrap 2.000 Amostras):</h4>
                  <p className="text-slate-400">
                    Δ Log Loss médio (B1 - M1): <span className="font-mono text-white font-bold">{experimentsData.latest_experiment.paired_bootstrap.mean_delta_log_loss}</span>{' '}
                    | Intervalo de Confiança 95%: <span className="font-mono text-white">[{experimentsData.latest_experiment.paired_bootstrap.ci_lower_95}, {experimentsData.latest_experiment.paired_bootstrap.ci_upper_95}]</span>.
                  </p>
                  <p className="text-slate-400 mt-1">
                    Conclusão Estatística:{' '}
                    <span className="font-semibold text-amber-400">{experimentsData.latest_experiment.paired_bootstrap.favors}</span> (Hipótese Nula H₀ mantida).
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-4 px-6 text-center text-xs text-slate-500">
        Brasileirão Predictor AI • Projeto de Pesquisa Aplicada • Dados OpenFootball CC0 1.0 • Não constitui recomendação financeira ou de apostas.
      </footer>
    </div>
  );
}
