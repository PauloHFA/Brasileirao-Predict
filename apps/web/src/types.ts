export interface MatchItem {
  match_id: number;
  season: number;
  round: number;
  kickoff_at: string;
  home_team_id: number;
  away_team_id: number;
  home_team_name: string;
  away_team_name: string;
  goals_home: number | null;
  goals_away: number | null;
  status: string;
}

export interface ScoreProbItem {
  home: number;
  away: number;
  probability: number;
}

export interface PredictionResult {
  match_id: number;
  as_of_at: string;
  model_id: string;
  snapshot_id: number;
  probabilities: {
    home: number;
    draw: number;
    away: number;
  };
  score_model?: {
    name: string;
    top_scores: ScoreProbItem[];
  };
  news_coverage: string;
  contextual_model_available: boolean;
  warnings: string[];
}

export interface ExplanationResult {
  match_id: number;
  snapshot_id: number;
  explanation: string;
  evidence_ids: string[];
  doc_urls: string[];
  disclaimer: string;
}

export interface ExperimentResults {
  latest_experiment: {
    config?: any;
    sample_counts?: {
      train: number;
      val: number;
      test: number;
    };
    metrics?: {
      [key: string]: {
        log_loss: number;
        brier_score: number;
        ece: number;
        accuracy: number;
        macro_f1: number;
      };
    };
    paired_bootstrap?: {
      mean_delta_log_loss: number;
      ci_lower_95: number;
      ci_upper_95: number;
      statistically_significant: boolean;
      favors: string;
    };
    promotion_gate?: {
      decision: string;
      active_model: string;
      relative_improvement: number;
      meets_loss_target: boolean;
      meets_calibration: boolean;
      explanation: string;
    };
  };
  historical_runs?: any[];
}
