import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class PoissonGoalModel:
    """
    Bivariate / Independent Poisson goal model for football matches.
    Estimates attack, defense and home advantage parameters to predict score distributions and 1X2.
    """

    def __init__(self, max_goals: int = 7):
        self.max_goals = max_goals
        self.fitted = False
        self.attack_strengths: Dict[int, float] = defaultdict(lambda: 0.0)
        self.defense_strengths: Dict[int, float] = defaultdict(lambda: 0.0)
        self.home_advantage: float = 0.25
        self.base_log_rate: float = 0.20 # ~1.22 goals per match base

    def fit(self, matches_data: List[Dict[str, Any]]) -> "PoissonGoalModel":
        """
        Fits team attack and defense relative rates from historical match data.
        """
        if not matches_data:
            self.fitted = True
            return self

        team_goals_scored: Dict[int, List[int]] = defaultdict(list)
        team_goals_conceded: Dict[int, List[int]] = defaultdict(list)
        all_home_goals = []
        all_away_goals = []

        for m in matches_data:
            gh = m["goals_home"]
            ga = m["goals_away"]
            h_id = m["home_team_id"]
            a_id = m["away_team_id"]

            team_goals_scored[h_id].append(gh)
            team_goals_conceded[h_id].append(ga)
            team_goals_scored[a_id].append(ga)
            team_goals_conceded[a_id].append(gh)

            all_home_goals.append(gh)
            all_away_goals.append(ga)

        avg_home = np.mean(all_home_goals) if all_home_goals else 1.35
        avg_away = np.mean(all_away_goals) if all_away_goals else 1.05
        overall_avg = (avg_home + avg_away) / 2.0

        self.home_advantage = float(np.log(max(avg_home, 0.1) / max(avg_away, 0.1)))
        self.base_log_rate = float(np.log(max(overall_avg, 0.5)))

        for team_id in team_goals_scored:
            scored = np.mean(team_goals_scored[team_id]) if team_goals_scored[team_id] else overall_avg
            conceded = np.mean(team_goals_conceded[team_id]) if team_goals_conceded[team_id] else overall_avg

            # Log relative strength centered at 0
            self.attack_strengths[team_id] = float(np.log(max(scored / overall_avg, 0.2)))
            self.defense_strengths[team_id] = float(np.log(max(conceded / overall_avg, 0.2)))

        self.fitted = True
        return self

    def predict_lambdas(self, home_team_id: int, away_team_id: int) -> Tuple[float, float]:
        """Calculates expected goals for home and away teams."""
        att_home = self.attack_strengths[home_team_id]
        def_home = self.defense_strengths[home_team_id]
        att_away = self.attack_strengths[away_team_id]
        def_away = self.defense_strengths[away_team_id]

        # lambda_home = exp(base + att_home + def_away + home_adv)
        log_lambda_home = self.base_log_rate + att_home + def_away + (self.home_advantage / 2.0)
        # lambda_away = exp(base + att_away + def_home - home_adv)
        log_lambda_away = self.base_log_rate + att_away + def_home - (self.home_advantage / 2.0)

        lambda_home = max(0.2, min(5.0, float(np.exp(log_lambda_home))))
        lambda_away = max(0.2, min(5.0, float(np.exp(log_lambda_away))))
        return lambda_home, lambda_away

    def predict_score_grid(self, home_team_id: int, away_team_id: int) -> np.ndarray:
        """Returns (max_goals+1, max_goals+1) matrix of score probabilities."""
        lambda_h, lambda_a = self.predict_lambdas(home_team_id, away_team_id)

        # Poisson PMFs
        p_home_goals = np.array([
            (lambda_h ** k * np.exp(-lambda_h)) / math.factorial(k)
            for k in range(self.max_goals + 1)
        ])
        p_away_goals = np.array([
            (lambda_a ** k * np.exp(-lambda_a)) / math.factorial(k)
            for k in range(self.max_goals + 1)
        ])

        # Independent bivariate matrix
        score_matrix = np.outer(p_home_goals, p_away_goals)
        # Normalize to sum exactly to 1.0 within grid
        score_matrix = score_matrix / np.sum(score_matrix)
        return score_matrix

    def predict_1x2_and_scores(
        self, home_team_id: int, away_team_id: int, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Derives P(H), P(D), P(A) by summing score probabilities, guaranteeing exact normalization.
        """
        grid = self.predict_score_grid(home_team_id, away_team_id)
        n = self.max_goals + 1

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        score_list: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(n):
                prob = float(grid[i, j])
                score_list.append({"home": i, "away": j, "probability": round(prob, 4)})
                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob

        # Sort top scores
        score_list.sort(key=lambda s: s["probability"], reverse=True)
        top_scores = score_list[:top_k]

        probabilities = [p_home, p_draw, p_away]

        # Invariant checks required by section 15 of spec
        assert all(0.0 <= p <= 1.0 for p in probabilities), "Probabilities must be within [0, 1]"
        total_p = sum(probabilities)
        assert abs(total_p - 1.0) < 1e-5, f"Sum of probabilities {total_p} must equal 1.0"

        # Renormalize tiny residual float error
        norm_factor = sum(probabilities)
        p_home /= norm_factor
        p_draw /= norm_factor
        p_away /= norm_factor

        lambda_h, lambda_a = self.predict_lambdas(home_team_id, away_team_id)

        return {
            "model_id": "poisson-1.0",
            "expected_goals_home": round(lambda_h, 2),
            "expected_goals_away": round(lambda_a, 2),
            "probabilities": {
                "home": round(float(p_home), 4),
                "draw": round(float(p_draw), 4),
                "away": round(float(p_away), 4)
            },
            "top_scores": top_scores
        }
