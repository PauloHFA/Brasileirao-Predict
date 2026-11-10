import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.config import BASE_DIR
from src.database import get_db
from src.schemas.models import ModelRun

router = APIRouter(prefix="/api/v1", tags=["Experiments"])

@router.get("/experiments")
def get_experiments(db: Session = Depends(get_db)):
    # Read latest experiment results from reports
    rep_file = BASE_DIR / "reports" / "experiment_results.json"
    latest_report = {}
    if rep_file.exists():
        with open(rep_file, "r", encoding="utf-8") as f:
            latest_report = json.load(f)

    # Read from model_runs table
    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).limit(10).all()
    runs_data = [
        {
            "run_id": r.run_id,
            "split_id": r.split_id,
            "model_type": r.model_type,
            "metrics": r.metrics,
            "created_at": r.created_at.isoformat()
        }
        for r in runs
    ]

    return {
        "latest_experiment": latest_report,
        "historical_runs": runs_data
    }
