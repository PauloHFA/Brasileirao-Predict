from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import get_db

router = APIRouter(tags=["Health"])

@router.get("/healthz")
def health_check():
    return {"status": "ok", "service": "brasileirao-predictor-api"}

@router.get("/readyz")
def readiness_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {str(e)}")
