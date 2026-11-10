import pytest

from src.features.numerical import EloCalculator


def test_elo_initialization_and_expected_score():
    calc = EloCalculator(base_elo=1500.0, home_advantage=65.0)
    # Neutral match without home advantage would be 0.5; with home advantage home has > 0.5
    exp_home = calc.expected_score(1500.0, 1500.0)
    assert 0.58 < exp_home < 0.61

def test_elo_update_win_loss():
    calc = EloCalculator(base_elo=1500.0, k_factor=20.0, home_advantage=0.0)
    # Both 1500, home wins
    calc.update(home_id=1, away_id=2, goals_home=2, goals_away=1)
    assert calc.get_rating(1) > 1500.0
    assert calc.get_rating(2) < 1500.0
    # Zero-sum property
    assert pytest.approx(calc.get_rating(1) + calc.get_rating(2), 0.01) == 3000.0

def test_elo_season_carryover_shrinkage():
    calc = EloCalculator(base_elo=1500.0, shrinkage=0.75)
    calc.current_ratings[1] = 1700.0
    calc.apply_season_carryover(2022)
    # Shrinkage: 0.75 * 1700 + 0.25 * 1500 = 1275 + 375 = 1650
    assert pytest.approx(calc.get_rating(1), 0.1) == 1650.0
