from src.database import SessionLocal, init_db
from src.evaluation.ablation import AblationStudyRunner


def test_ablation_study_structure():
    init_db()
    db = SessionLocal()
    try:
        runner = AblationStudyRunner(db)
        # Run on subset
        results = runner.run_full_ablation_study(
            train_seasons=[2021],
            val_seasons=[2022],
            test_seasons=[2023]
        )
        assert "feature_ablations" in results
        assert "M1_full_context" in results["feature_ablations"]
        assert "ablation_without_absences" in results["feature_ablations"]
        assert "ablation_without_crisis" in results["feature_ablations"]
        assert "media_bias_analysis" in results
        assert "metrics_popular_teams" in results["media_bias_analysis"]
    finally:
        db.close()
