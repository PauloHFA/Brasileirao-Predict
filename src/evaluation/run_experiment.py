import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from src.config import BASE_DIR
from src.database import SessionLocal
from src.evaluation.metrics import compute_multiclass_metrics, paired_bootstrap_test
from src.evaluation.splits import create_temporal_splits
from src.features.contextual_aggregator import (
    aggregate_team_context_features,
    merge_match_context_features,
)
from src.models.baseline_prior import PriorBaselineModel
from src.models.contextual_lgbm import CONTEXTUAL_FEATURE_COLUMNS, ContextualLGBMModel
from src.models.news_count_lgbm import NEWS_COUNT_FEATURE_COLUMNS, NewsCountLGBMModel
from src.models.numeric_lgbm import NUMERICAL_FEATURE_COLUMNS, NumericLGBMModel
from src.models.poisson_goals import PoissonGoalModel
from src.models.promotion_gate import PromotionGate
from src.schemas.models import ModelRun


def run_experiment(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db = SessionLocal()
    try:
        train_seasons = config.get("train_seasons", [2021, 2022])
        val_seasons = config.get("validation_seasons", [2023])
        test_seasons = config.get("test_seasons", [2024])

        splits = create_temporal_splits(db, train_seasons, val_seasons, test_seasons)

        train_data = splits["train"]
        val_data = splits["val"]
        test_data = splits["test"]

        print(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} matches")

        # Prepare X and y arrays
        def extract_matrices(items: List[Dict[str, Any]]):
            y = [it["label"] for it in items]
            
            # Enrich items with contextual features
            enriched_items = []
            for it in items:
                num_feats = it["features"]
                # Synthetic or extracted context
                empty_ctx = aggregate_team_context_features([], it["as_of_at"])
                merged = merge_match_context_features(num_feats, empty_ctx, empty_ctx)
                enriched_items.append(merged)

            X_num = np.array([[it.get(c, 0.0) for c in NUMERICAL_FEATURE_COLUMNS] for it in enriched_items])
            X_news = np.array([[it.get(c, 0.0) for c in NEWS_COUNT_FEATURE_COLUMNS] for it in enriched_items])
            X_ctx = np.array([[it.get(c, 0.0) for c in CONTEXTUAL_FEATURE_COLUMNS] for it in enriched_items])
            return y, X_num, X_news, X_ctx

        y_train, X_train_num, X_train_news, X_train_ctx = extract_matrices(train_data)
        y_val, X_val_num, X_val_news, X_val_ctx = extract_matrices(val_data)
        y_test, X_test_num, X_test_news, X_test_ctx = extract_matrices(test_data)

        # 1. Model B0 - Prior
        print("Training Model B0 (Prior)...")
        b0 = PriorBaselineModel()
        b0.fit(y_train)
        probs_b0 = np.array([list(b0.predict_proba_one().values()) for _ in y_test])
        metrics_b0 = compute_multiclass_metrics(y_test, probs_b0)

        # 2. Model B1 - Numeric LightGBM
        print("Training Model B1 (Numeric LightGBM)...")
        b1 = NumericLGBMModel(calibrate=True)
        b1.fit(X_train_num, np.array(y_train), X_val=X_val_num, y_val=np.array(y_val))
        probs_b1 = b1.predict_proba(X_test_num)
        metrics_b1 = compute_multiclass_metrics(y_test, probs_b1)

        # 3. Model B2 - News Count LightGBM
        print("Training Model B2 (News Count LightGBM)...")
        b2 = NewsCountLGBMModel(calibrate=True)
        b2.fit(X_train_news, np.array(y_train), X_val=X_val_news, y_val=np.array(y_val))
        probs_b2 = b2.predict_proba(X_test_news)
        metrics_b2 = compute_multiclass_metrics(y_test, probs_b2)

        # 4. Model M1 - Contextual LightGBM
        print("Training Model M1 (Contextual LightGBM)...")
        m1 = ContextualLGBMModel(calibrate=True)
        m1.fit(X_train_ctx, np.array(y_train), X_val=X_val_ctx, y_val=np.array(y_val))
        probs_m1 = m1.predict_proba(X_test_ctx)
        metrics_m1 = compute_multiclass_metrics(y_test, probs_m1)

        # 5. Model G1 - Poisson
        print("Training Model G1 (Poisson Goals)...")
        g1 = PoissonGoalModel()
        train_matches_info = [
            {
                "home_team_id": it["home_team_id"],
                "away_team_id": it["away_team_id"],
                "goals_home": it["features"]["goals_for_5_home"], # Proxy for goals
                "goals_away": it["features"]["goals_for_5_away"]
            }
            for it in train_data
        ]
        g1.fit(train_matches_info)
        probs_g1_list = []
        for it in test_data:
            p_res = g1.predict_1x2_and_scores(it["home_team_id"], it["away_team_id"])
            p_dict = p_res["probabilities"]
            probs_g1_list.append([p_dict["home"], p_dict["draw"], p_dict["away"]])
        probs_g1 = np.array(probs_g1_list)
        metrics_g1 = compute_multiclass_metrics(y_test, probs_g1)

        # Paired Bootstrap
        print("Running paired bootstrap evaluation (2,000 resamples)...")
        bootstrap_res = paired_bootstrap_test(y_test, probs_b1, probs_m1, n_bootstraps=2000, seed=42)

        # Promotion Gate
        gate = PromotionGate()
        gate_res = gate.evaluate(metrics_b1, metrics_m1, bootstrap_res)

        # Save artifacts
        artifacts_dir = BASE_DIR / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        b1.save(artifacts_dir / "b1_numeric_lgbm.pkl")
        b2.save(artifacts_dir / "b2_news_count_lgbm.pkl")
        m1.save(artifacts_dir / "m1_contextual_lgbm.pkl")

        # Compile final results
        results = {
            "config": config,
            "sample_counts": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test)
            },
            "metrics": {
                "B0_prior": metrics_b0,
                "B1_numeric": metrics_b1,
                "B2_news_count": metrics_b2,
                "M1_contextual": metrics_m1,
                "G1_poisson": metrics_g1
            },
            "paired_bootstrap": bootstrap_res,
            "promotion_gate": gate_res
        }

        # Save report
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Register in database model_runs
        run_hash = hashlib.sha256(json.dumps(results["metrics"], sort_keys=True).encode()).hexdigest()[:16]
        run_id = f"RUN_{run_hash}"
        existing_run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
        if existing_run:
            existing_run.metrics = results["metrics"]
            existing_run.model_type = gate_res["active_model"]
            existing_run.artifact_uri = str(artifacts_dir / "b1_numeric_lgbm.pkl")
            existing_run.created_at = datetime.now(timezone.utc)
        else:
            run_record = ModelRun(
                run_id=run_id,
                config_hash=hashlib.sha256(yaml.dump(config).encode()).hexdigest(),
                split_id="WALK_FORWARD_2021_2024",
                model_type=gate_res["active_model"],
                metrics=results["metrics"],
                artifact_uri=str(artifacts_dir / "b1_numeric_lgbm.pkl")
            )
            db.add(run_record)
        db.commit()

        return results
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_v1.yaml")
    args = parser.parse_args()

    res = run_experiment(Path(args.config))
    print("\n================ EXPERIMENTO CONCLUÍDO ================")
    print(f"Log Loss B0 (Prior):        {res['metrics']['B0_prior']['log_loss']}")
    print(f"Log Loss B1 (Numérico):     {res['metrics']['B1_numeric']['log_loss']}")
    print(f"Log Loss B2 (Volume Notícia): {res['metrics']['B2_news_count']['log_loss']}")
    print(f"Log Loss M1 (Contextual):   {res['metrics']['M1_contextual']['log_loss']}")
    print(f"Log Loss G1 (Poisson):      {res['metrics']['G1_poisson']['log_loss']}")
    print(f"Decisão do Gate:            {res['promotion_gate']['decision']}")
    print(f"Modelo Ativo:               {res['promotion_gate']['active_model']}")
    print(f"Explicação:                 {res['promotion_gate']['explanation']}")
