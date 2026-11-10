from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.config import BASE_DIR
from src.database import get_db
from src.features.contextual_aggregator import (
    aggregate_team_context_features,
    merge_match_context_features,
)
from src.models.baseline_prior import PriorBaselineModel
from src.models.contextual_lgbm import ContextualLGBMModel
from src.models.numeric_lgbm import NumericLGBMModel
from src.models.poisson_goals import PoissonGoalModel
from src.schemas.api import (
    PredictionResponse,
    PredictionSimulationRequest,
    Probabilities1X2,
    ScoreModelResponse,
    ScoreProbItem,
)
from src.schemas.models import FeatureSnapshot, Match, Prediction

router = APIRouter(prefix="/api/v1", tags=["Predictions"])

# Singleton caches for models
_CACHED_MODELS = {}

def get_loaded_model(model_name: str):
    if model_name in _CACHED_MODELS:
        return _CACHED_MODELS[model_name]

    artifacts_dir = BASE_DIR / "artifacts"
    if model_name == "numeric_lgbm":
        path = artifacts_dir / "b1_numeric_lgbm.pkl"
        if path.exists():
            model = NumericLGBMModel.load(path)
        else:
            model = NumericLGBMModel()
    elif model_name == "contextual_lgbm":
        path = artifacts_dir / "m1_contextual_lgbm.pkl"
        if path.exists():
            model = ContextualLGBMModel.load(path)
        else:
            model = ContextualLGBMModel()
    elif model_name == "prior":
        model = PriorBaselineModel()
    else:
        model = None

    _CACHED_MODELS[model_name] = model
    return model

@router.post("/predictions", response_model=PredictionResponse)
def simulate_prediction(req: PredictionSimulationRequest, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.match_id == req.match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    snapshot = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.match_id == req.match_id)
        .order_by(FeatureSnapshot.as_of_at.desc())
        .first()
    )

    if not snapshot:
        raise HTTPException(status_code=404, detail="Feature snapshot not found for this match.")

    # Calculate Poisson score distribution
    poisson_model = PoissonGoalModel(max_goals=6)
    poisson_res = poisson_model.predict_1x2_and_scores(match.home_team_id, match.away_team_id, top_k=5)
    score_model_resp = ScoreModelResponse(
        name="poisson-1.0",
        top_scores=[ScoreProbItem(**s) for s in poisson_res["top_scores"]]
    )

    # Calculate 1X2 probabilities according to selected model
    model_type = req.model_type
    num_feats = snapshot.features or {}
    empty_ctx = aggregate_team_context_features([], snapshot.as_of_at)
    ctx_feats = merge_match_context_features(num_feats, empty_ctx, empty_ctx)

    if model_type == "poisson":
        p_dict = poisson_res["probabilities"]
        model_id = "poisson-1.0"
    elif model_type == "prior":
        model = get_loaded_model("prior")
        p_dict = model.predict_proba_one()
        model_id = "prior-1.0"
    elif model_type == "contextual_lgbm":
        model = get_loaded_model("contextual_lgbm")
        p_dict = model.predict_one(ctx_feats)
        model_id = "contextual-lgbm-1.0"
    else:
        model = get_loaded_model("numeric_lgbm")
        p_dict = model.predict_one(num_feats)
        model_id = "baseline-lgbm-1.0"

    # Save to predictions table
    pred_record = Prediction(
        match_id=match.match_id,
        snapshot_id=snapshot.snapshot_id,
        model_id=model_id,
        p_home=p_dict["home"],
        p_draw=p_dict["draw"],
        p_away=p_dict["away"],
        score_distribution={"top_scores": poisson_res["top_scores"]}
    )
    db.add(pred_record)
    db.commit()

    return PredictionResponse(
        match_id=match.match_id,
        as_of_at=snapshot.as_of_at.isoformat(),
        model_id=model_id,
        snapshot_id=snapshot.snapshot_id,
        probabilities=Probabilities1X2(
            home=p_dict["home"],
            draw=p_dict["draw"],
            away=p_dict["away"]
        ),
        score_model=score_model_resp,
        news_coverage="INSUFFICIENT" if ctx_feats.get("news_coverage_count", 0) == 0 else "PARTIAL",
        contextual_model_available=True,
        warnings=["Not enough verified pre-match news"] if ctx_feats.get("news_coverage_count", 0) == 0 else []
    )

@router.get("/predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    return {
        "prediction_id": pred.prediction_id,
        "match_id": pred.match_id,
        "snapshot_id": pred.snapshot_id,
        "model_id": pred.model_id,
        "probabilities": {
            "home": pred.p_home,
            "draw": pred.p_draw,
            "away": pred.p_away
        },
        "score_distribution": pred.score_distribution,
        "created_at": pred.created_at.isoformat()
    }
