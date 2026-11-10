import pytest
from fastapi import HTTPException

from apps.api.routers import experiments, explanations, health, ingestion, matches, predictions
from src.database import SessionLocal, init_db
from src.schemas.api import (
    ExplanationRequest,
    ManualNewsIngestionRequest,
    PredictionSimulationRequest,
)


@pytest.fixture(scope="module")
def db_session():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_health_and_readiness(db_session):
    h = health.health_check()
    assert h["status"] == "ok"

    r = health.readiness_check(db_session)
    assert r["status"] == "ready"
    assert r["database"] == "connected"

def test_seasons_endpoint(db_session):
    seasons = matches.list_seasons(db_session)
    assert len(seasons) >= 3
    season_years = [s["season"] for s in seasons]
    assert 2024 in season_years
    assert seasons[0]["total_matches"] == 380

def test_matches_and_features(db_session):
    match_list = matches.list_matches(season=2024, round=1, status=None, db=db_session)
    assert len(match_list) == 10

    m1 = match_list[0]
    m_id = m1.match_id

    # Features endpoint
    feat_data = matches.get_match_features(match_id=m_id, as_of=None, db=db_session)
    assert feat_data["match_id"] == m_id
    assert feat_data["feature_version"] == "v1.0"
    assert "elo_diff_pre" in feat_data["features"]
    assert "form_points_5_home" in feat_data["features"]

def test_prediction_simulation(db_session):
    match_list = matches.list_matches(season=2024, round=1, status=None, db=db_session)
    m_id = match_list[0].match_id

    req = PredictionSimulationRequest(match_id=m_id, model_type="numeric_lgbm", as_of_hours=24)
    pred_res = predictions.simulate_prediction(req, db=db_session)

    probs = pred_res.probabilities
    assert 0.0 <= probs.home <= 1.0
    assert 0.0 <= probs.draw <= 1.0
    assert 0.0 <= probs.away <= 1.0
    total = probs.home + probs.draw + probs.away
    assert pytest.approx(total, abs=1e-3) == 1.0
    assert pred_res.score_model is not None
    assert len(pred_res.score_model.top_scores) > 0

def test_explanation_endpoint(db_session):
    match_list = matches.list_matches(season=2024, round=1, status=None, db=db_session)
    m_id = match_list[0].match_id

    req = ExplanationRequest(match_id=m_id)
    exp_res = explanations.generate_match_explanation(req, db=db_session)

    assert exp_res.match_id == m_id
    assert "Análise pré-jogo" in exp_res.explanation
    assert "não implicam nexo de causalidade" in exp_res.disclaimer

def test_manual_ingestion_and_temporal_rejection(db_session):
    match_list = matches.list_matches(season=2024, round=1, status=None, db=db_session)
    m_id = match_list[0].match_id

    # Case A: Valid eligible news (before match cutoff)
    valid_req = ManualNewsIngestionRequest(
        url="https://esporte.exemplo.com/noticia-valida-teste-1",
        title="Flamengo confirma treino com titulares",
        published_at="2024-04-10T12:00:00Z",
        source_name="exemplo_esporte",
        team_slug="flamengo",
        match_id=m_id,
        text_content="Treinamento tático realizado normalmente com todos os titulares.",
        rights_status="ALLOWED"
    )
    res_valid = ingestion.ingest_manual_news(valid_req, x_admin_token="dev-secret-token-123", db=db_session)
    assert res_valid.status == "ACCEPTED_AND_EXTRACTED"
    assert res_valid.extracted_features is not None

    # Case B: Future leaked news (after match cutoff) -> MUST BE REJECTED with HTTPException 422!
    leak_req = ManualNewsIngestionRequest(
        url="https://esporte.exemplo.com/noticia-vazada-futura-teste-2",
        title="Flamengo vence jogo após apito final",
        published_at="2026-12-01T22:00:00Z",
        source_name="exemplo_esporte",
        team_slug="flamengo",
        match_id=m_id,
        text_content="Resultado pós-jogo.",
        rights_status="ALLOWED"
    )
    with pytest.raises(HTTPException) as exc_info:
        ingestion.ingest_manual_news(leak_req, x_admin_token="dev-secret-token-123", db=db_session)
    assert exc_info.value.status_code == 422
    assert "Temporal leakage violation" in str(exc_info.value.detail)

def test_experiments_endpoint(db_session):
    exp_data = experiments.get_experiments(db_session)
    assert "latest_experiment" in exp_data
    assert "historical_runs" in exp_data
