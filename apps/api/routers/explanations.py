from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.rag.explainer import ExplanationGenerator
from src.schemas.api import ExplanationRequest, ExplanationResponse

router = APIRouter(prefix="/api/v1", tags=["Explanations"])
explainer = ExplanationGenerator()

@router.post("/explanations", response_model=ExplanationResponse)
def generate_match_explanation(req: ExplanationRequest, db: Session = Depends(get_db)):
    try:
        res = explainer.generate_explanation(db, match_id=req.match_id, snapshot_id=req.snapshot_id)
        return ExplanationResponse(
            match_id=res["match_id"],
            snapshot_id=res["snapshot_id"],
            explanation=res["explanation"],
            evidence_ids=res["evidence_ids"],
            doc_urls=res["doc_urls"],
            disclaimer=res["disclaimer"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
