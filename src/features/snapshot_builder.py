import argparse
import hashlib
import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.database import SessionLocal, init_db
from src.features.cutoff import get_cutoff_time
from src.features.numerical import compute_historical_features, get_match_numerical_features
from src.schemas.models import FeatureSnapshot, Match

FEATURE_VERSION = "v1.0"

def build_snapshots_for_matches(
    db: Session,
    as_of_hours: int = 24,
    seasons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Builds immutable pre-match feature snapshots strictly as of (kickoff - as_of_hours).
    Ensures zero future data leakage.
    """
    query = db.query(Match).order_by(Match.kickoff_at.asc())
    if seasons:
        query = query.filter(Match.season.in_(seasons))

    matches = query.all()
    if not matches:
        return {"total_matches": 0, "snapshots_created": 0, "snapshots_updated": 0}

    created = 0
    updated = 0

    # Build chronologically to maintain historical simulation state efficiently
    for idx, match in enumerate(matches):
        as_of_at = get_cutoff_time(match.kickoff_at, as_of_hours=as_of_hours)
        
        # Historical state only sees matches finished strictly before this match's cutoff
        hist_state = compute_historical_features(db, as_of_at)
        features = get_match_numerical_features(
            match.home_team_id,
            match.away_team_id,
            match.kickoff_at,
            as_of_at,
            hist_state
        )

        payload_bytes = json.dumps(features, sort_keys=True).encode("utf-8")
        snapshot_hash = hashlib.sha256(payload_bytes).hexdigest()

        existing = (
            db.query(FeatureSnapshot)
            .filter(
                FeatureSnapshot.match_id == match.match_id,
                FeatureSnapshot.as_of_at == as_of_at,
                FeatureSnapshot.feature_version == FEATURE_VERSION
            )
            .first()
        )

        if existing:
            existing.features = features
            existing.snapshot_hash = snapshot_hash
            updated += 1
        else:
            snap = FeatureSnapshot(
                match_id=match.match_id,
                as_of_at=as_of_at,
                feature_version=FEATURE_VERSION,
                features=features,
                news_coverage={},
                snapshot_hash=snapshot_hash
            )
            db.add(snap)
            created += 1

        if (idx + 1) % 100 == 0:
            db.commit()

    db.commit()
    return {
        "total_matches": len(matches),
        "snapshots_created": created,
        "snapshots_updated": updated,
        "as_of_hours": as_of_hours,
        "feature_version": FEATURE_VERSION
    }

if __name__ == "__main__":
    init_db()
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-hours", type=int, default=24)
    parser.add_argument("--seasons", type=str, default="")
    args = parser.parse_args()

    seasons_list = [int(s.strip()) for s in args.seasons.split(",") if s.strip()] or None
    session = SessionLocal()
    try:
        res = build_snapshots_for_matches(session, as_of_hours=args.as_of_hours, seasons=seasons_list)
        print(f"Snapshots built: {res['snapshots_created']} created, {res['snapshots_updated']} updated for {res['total_matches']} matches.")
    finally:
        session.close()
