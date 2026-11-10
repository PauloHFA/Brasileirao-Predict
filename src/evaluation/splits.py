from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.schemas.models import FeatureSnapshot, Match


def create_temporal_splits(
    db: Session,
    train_seasons: List[int],
    val_seasons: List[int],
    test_seasons: List[int],
    feature_version: str = "v1.0"
) -> Dict[str, Dict[str, Any]]:
    """
    Creates chronological train/validation/test splits from feature_snapshots.
    Returns dictionaries containing match_ids, X_numeric, X_contextual, y, kickoff_ats.
    """
    all_seasons = sorted(list(set(train_seasons + val_seasons + test_seasons)))

    snapshots = (
        db.query(FeatureSnapshot, Match)
        .join(Match, FeatureSnapshot.match_id == Match.match_id)
        .filter(
            Match.season.in_(all_seasons),
            FeatureSnapshot.feature_version == feature_version,
            Match.status == "FINISHED",
            Match.goals_home.isnot(None),
            Match.goals_away.isnot(None)
        )
        .order_by(Match.kickoff_at.asc())
        .all()
    )

    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }

    for snap, match in snapshots:
        gh = match.goals_home
        ga = match.goals_away
        if gh > ga:
            label = "H"
        elif gh == ga:
            label = "D"
        else:
            label = "A"

        item = {
            "match_id": match.match_id,
            "season": match.season,
            "round": match.round,
            "kickoff_at": match.kickoff_at,
            "as_of_at": snap.as_of_at,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "features": snap.features,
            "news_coverage": snap.news_coverage or {},
            "label": label
        }

        if match.season in train_seasons:
            splits["train"].append(item)
        elif match.season in val_seasons:
            splits["val"].append(item)
        elif match.season in test_seasons:
            splits["test"].append(item)

    return splits
