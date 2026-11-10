import pytest

from src.models.poisson_goals import PoissonGoalModel


def test_poisson_probability_axioms():
    model = PoissonGoalModel(max_goals=7)
    # Fit on small sample
    sample_matches = [
        {"home_team_id": 1, "away_team_id": 2, "goals_home": 2, "goals_away": 1},
        {"home_team_id": 2, "away_team_id": 3, "goals_home": 1, "goals_away": 1},
        {"home_team_id": 3, "away_team_id": 1, "goals_home": 0, "goals_away": 3},
    ]
    model.fit(sample_matches)

    pred = model.predict_1x2_and_scores(home_team_id=1, away_team_id=2)
    probs = pred["probabilities"]

    assert 0.0 <= probs["home"] <= 1.0
    assert 0.0 <= probs["draw"] <= 1.0
    assert 0.0 <= probs["away"] <= 1.0
    total = probs["home"] + probs["draw"] + probs["away"]
    assert pytest.approx(total, abs=1e-4) == 1.0
    assert len(pred["top_scores"]) > 0

def test_poisson_grid_normalization():
    model = PoissonGoalModel(max_goals=6)
    grid = model.predict_score_grid(1, 2)
    assert grid.shape == (7, 7)
    assert pytest.approx(float(grid.sum()), abs=1e-5) == 1.0
