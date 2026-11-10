from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MatchItemResponse(BaseModel):
    match_id: int
    season: int
    round: int
    kickoff_at: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    goals_home: Optional[int] = None
    goals_away: Optional[int] = None
    status: str

class ScoreProbItem(BaseModel):
    home: int
    away: int
    probability: float

class ScoreModelResponse(BaseModel):
    name: str = "poisson-1.0"
    top_scores: List[ScoreProbItem]

class Probabilities1X2(BaseModel):
    home: float = Field(..., ge=0.0, le=1.0)
    draw: float = Field(..., ge=0.0, le=1.0)
    away: float = Field(..., ge=0.0, le=1.0)

class PredictionSimulationRequest(BaseModel):
    match_id: int
    model_type: Literal["numeric_lgbm", "contextual_lgbm", "prior", "poisson"] = "numeric_lgbm"
    as_of_hours: Optional[int] = 24

class PredictionResponse(BaseModel):
    match_id: int
    as_of_at: str
    model_id: str
    snapshot_id: int
    probabilities: Probabilities1X2
    score_model: Optional[ScoreModelResponse] = None
    news_coverage: str = "INSUFFICIENT" # INSUFFICIENT, PARTIAL, COMPLETE
    contextual_model_available: bool = False
    warnings: List[str] = Field(default_factory=list)

class ManualNewsIngestionRequest(BaseModel):
    url: str
    title: str
    published_at: str # ISO-8601
    first_seen_at: Optional[str] = None # ISO-8601, defaults to published_at for historical ingestion
    source_name: str
    team_slug: str
    match_id: Optional[int] = None
    text_content: str
    rights_status: Literal["ALLOWED", "METADATA_ONLY", "BLOCKED"] = "ALLOWED"

class ManualNewsIngestionResponse(BaseModel):
    article_id: int
    status: str
    message: str
    extracted_features: Optional[Dict[str, Any]] = None

class ExplanationRequest(BaseModel):
    match_id: int
    snapshot_id: Optional[int] = None
    question: Optional[str] = "Por que o modelo atribuiu essas probabilidades?"

class ExplanationResponse(BaseModel):
    match_id: int
    snapshot_id: int
    explanation: str
    evidence_ids: List[str]
    doc_urls: List[str]
    disclaimer: str
