from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.schemas.models import Match

DEFAULT_ELO = 1500.0
DEFAULT_K = 24.0
HOME_ADVANTAGE_ELO = 65.0
SHRINKAGE_FACTOR = 0.75

class EloCalculator:
    """Chronological Elo rating calculator with home advantage and seasonal shrinkage."""

    def __init__(
        self,
        base_elo: float = DEFAULT_ELO,
        k_factor: float = DEFAULT_K,
        home_advantage: float = HOME_ADVANTAGE_ELO,
        shrinkage: float = SHRINKAGE_FACTOR
    ):
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.shrinkage = shrinkage
        self.current_ratings: Dict[int, float] = defaultdict(lambda: self.base_elo)
        self.last_season_seen: Optional[int] = None

    def apply_season_carryover(self, new_season: int) -> None:
        """Shrink ratings towards 1500 at the start of a new season."""
        if self.last_season_seen is not None and new_season != self.last_season_seen:
            for team_id in list(self.current_ratings.keys()):
                self.current_ratings[team_id] = (
                    self.shrinkage * self.current_ratings[team_id] + (1.0 - self.shrinkage) * self.base_elo
                )
        self.last_season_seen = new_season

    def get_rating(self, team_id: int) -> float:
        return self.current_ratings[team_id]

    def expected_score(self, rating_home: float, rating_away: float) -> float:
        """Expected home score incorporating home advantage."""
        diff = (rating_home + self.home_advantage) - rating_away
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, home_id: int, away_id: int, goals_home: int, goals_away: int) -> None:
        """Update ratings after a match concludes."""
        r_home = self.current_ratings[home_id]
        r_away = self.current_ratings[away_id]

        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        if goals_home > goals_away:
            s_home, s_away = 1.0, 0.0
        elif goals_home == goals_away:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        self.current_ratings[home_id] = r_home + self.k_factor * (s_home - e_home)
        self.current_ratings[away_id] = r_away + self.k_factor * (s_away - e_away)

def compute_historical_features(db: Session, as_of_at: datetime) -> Dict[str, Any]:
    """
    Computes all historical numerical features available strictly before cutoff as_of_at.
    Returns team-level histories up to as_of_at.
    """
    past_matches = (
        db.query(Match)
        .filter(Match.kickoff_at < as_of_at, Match.status == "FINISHED")
        .order_by(Match.kickoff_at.asc())
        .all()
    )

    elo_calc = EloCalculator()
    team_matches: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for m in past_matches:
        elo_calc.apply_season_carryover(m.season)

        # Record match for home team
        team_matches[m.home_team_id].append({
            "match_id": m.match_id,
            "kickoff_at": m.kickoff_at,
            "is_home": True,
            "goals_for": m.goals_home,
            "goals_against": m.goals_away,
            "points": 3 if m.goals_home > m.goals_away else (1 if m.goals_home == m.goals_away else 0)
        })

        # Record match for away team
        team_matches[m.away_team_id].append({
            "match_id": m.match_id,
            "kickoff_at": m.kickoff_at,
            "is_home": False,
            "goals_for": m.goals_away,
            "goals_against": m.goals_home,
            "points": 3 if m.goals_away > m.goals_home else (1 if m.goals_away == m.goals_home else 0)
        })

        # Advance Elo ratings
        elo_calc.update(m.home_team_id, m.away_team_id, m.goals_home, m.goals_away)

    return {
        "elo_calc": elo_calc,
        "team_matches": team_matches
    }

def get_match_numerical_features(
    home_team_id: int,
    away_team_id: int,
    kickoff_at: datetime,
    as_of_at: datetime,
    historical_state: Dict[str, Any]
) -> Dict[str, float]:
    """
    Extracts tabular features for a specific match given precomputed historical state at as_of_at.
    """
    elo_calc: EloCalculator = historical_state["elo_calc"]
    team_matches: Dict[int, List[Dict[str, Any]]] = historical_state["team_matches"]

    elo_home = elo_calc.get_rating(home_team_id)
    elo_away = elo_calc.get_rating(away_team_id)
    elo_diff_pre = elo_home - elo_away

    # Home team rolling form
    home_history = team_matches.get(home_team_id, [])
    recent_home = home_history[-5:] if home_history else []
    
    form_points_5_home = (sum(m["points"] for m in recent_home) / len(recent_home)) if recent_home else 1.33
    goals_for_5_home = (sum(m["goals_for"] for m in recent_home) / len(recent_home)) if recent_home else 1.2
    goals_against_5_home = (sum(m["goals_against"] for m in recent_home) / len(recent_home)) if recent_home else 1.2

    # Away team rolling form
    away_history = team_matches.get(away_team_id, [])
    recent_away = away_history[-5:] if away_history else []
    
    form_points_5_away = (sum(m["points"] for m in recent_away) / len(recent_away)) if recent_away else 1.15
    goals_for_5_away = (sum(m["goals_for"] for m in recent_away) / len(recent_away)) if recent_away else 1.0
    goals_against_5_away = (sum(m["goals_against"] for m in recent_away) / len(recent_away)) if recent_away else 1.4

    # Rest days
    if home_history:
        last_kickoff = home_history[-1]["kickoff_at"]
        rest_days_home = max(1.0, (kickoff_at - last_kickoff).total_seconds() / 86400.0)
    else:
        rest_days_home = 7.0

    if away_history:
        last_kickoff = away_history[-1]["kickoff_at"]
        rest_days_away = max(1.0, (kickoff_at - last_kickoff).total_seconds() / 86400.0)
    else:
        rest_days_away = 7.0

    return {
        "elo_diff_pre": round(float(elo_diff_pre), 2),
        "elo_home": round(float(elo_home), 2),
        "elo_away": round(float(elo_away), 2),
        "form_points_5_home": round(float(form_points_5_home), 3),
        "form_points_5_away": round(float(form_points_5_away), 3),
        "goals_for_5_home": round(float(goals_for_5_home), 3),
        "goals_against_5_home": round(float(goals_against_5_home), 3),
        "goals_for_5_away": round(float(goals_for_5_away), 3),
        "goals_against_5_away": round(float(goals_against_5_away), 3),
        "rest_days_home": round(float(rest_days_home), 1),
        "rest_days_away": round(float(rest_days_away), 1),
        "home_advantage": 1.0
    }
