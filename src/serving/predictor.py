"""
Interface de serviço e inferência do Brasileirão Predictor AI.
Permite executar previsões programáticas diretamente via Python ou CLI sem necessidade de servidor HTTP.
"""
from typing import Any, Dict

from src.config import BASE_DIR
from src.database import SessionLocal
from src.features.contextual_aggregator import (
    aggregate_team_context_features,
    merge_match_context_features,
)
from src.models.contextual_lgbm import ContextualLGBMModel
from src.models.numeric_lgbm import NumericLGBMModel
from src.models.poisson_goals import PoissonGoalModel
from src.schemas.models import FeatureSnapshot, Match


class MatchPredictor:
    """
    Unified predictor supporting B1 (Numeric), M1 (Contextual), and G1 (Poisson) models.
    """

    def __init__(self):
        artifacts = BASE_DIR / "artifacts"
        # Load B1
        b1_path = artifacts / "b1_numeric_lgbm.pkl"
        if b1_path.exists():
            self.model_b1 = NumericLGBMModel.load(b1_path)
        else:
            self.model_b1 = NumericLGBMModel()

        # Load M1
        m1_path = artifacts / "m1_contextual_lgbm.pkl"
        if m1_path.exists():
            self.model_m1 = ContextualLGBMModel.load(m1_path)
        else:
            self.model_m1 = ContextualLGBMModel()

        # Poisson G1
        self.model_g1 = PoissonGoalModel(max_goals=6)

    def predict_by_match_id(self, match_id: int, model_type: str = "numeric_lgbm") -> Dict[str, Any]:
        db = SessionLocal()
        try:
            match = db.query(Match).filter(Match.match_id == match_id).first()
            if not match:
                raise ValueError(f"Match {match_id} not found.")

            snapshot = (
                db.query(FeatureSnapshot)
                .filter(FeatureSnapshot.match_id == match_id)
                .order_by(FeatureSnapshot.as_of_at.desc())
                .first()
            )
            if not snapshot:
                raise ValueError(f"Snapshot for match {match_id} not found.")

            poisson_res = self.model_g1.predict_1x2_and_scores(match.home_team_id, match.away_team_id)

            num_feats = snapshot.features or {}
            empty_ctx = aggregate_team_context_features([], snapshot.as_of_at)
            ctx_feats = merge_match_context_features(num_feats, empty_ctx, empty_ctx)

            if model_type == "contextual_lgbm":
                p_dict = self.model_m1.predict_one(ctx_feats)
                active_id = "contextual_lgbm"
            elif model_type == "poisson":
                p_dict = poisson_res["probabilities"]
                active_id = "poisson_goals"
            else:
                p_dict = self.model_b1.predict_one(num_feats)
                active_id = "numeric_lgbm"

            return {
                "match_id": match_id,
                "season": match.season,
                "round": match.round,
                "model_id": active_id,
                "as_of_at": snapshot.as_of_at.isoformat(),
                "probabilities": p_dict,
                "top_scores": poisson_res["top_scores"][:5]
            }
        finally:
            db.close()

if __name__ == "__main__":
    predictor = MatchPredictor()
    res = predictor.predict_by_match_id(1)
    print("Previsão Programática da Partida 1:")
    print(f"Modelo: {res['model_id']}")
    print(f"Probabilidades: {res['probabilities']}")
    print(f"Placares Prováveis: {res['top_scores'][:3]}")
