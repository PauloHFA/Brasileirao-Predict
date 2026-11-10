from datetime import datetime, timezone

from src.database import SessionLocal, init_db
from src.features.scouts import ScoutsFeatureCalculator


def test_scouts_calculator():
    init_db()
    db = SessionLocal()
    try:
        calc = ScoutsFeatureCalculator(db)
        as_of = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        history = calc.compute_team_scouts_history(as_of)
        assert len(history) > 0

        # Match between team 1 and team 2
        feats = calc.get_match_scouts_features(1, 2, history)
        assert "shots_5_home" in feats
        assert "xg_5_home" in feats
        assert "xg_diff_5" in feats
        assert feats["xg_5_home"] > 0
        assert feats["shots_5_home"] > 0
    finally:
        db.close()
