from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from apps.api.routers import explanations, health, ingestion, matches, predictions
from src.config import BASE_DIR
from src.data_quality.auditor import audit_database
from src.database import SessionLocal, init_db
from src.evaluation.run_experiment import run_experiment
from src.schemas.api import (
    ExplanationRequest,
    ManualNewsIngestionRequest,
    PredictionSimulationRequest,
)
from src.schemas.models import Match


def test_complete_demonstration_and_acceptance_workflow():
    """
    Automated verification of the official 10-step Demonstration and Acceptance Script
    specified in Apêndice E of the Technical Specification.
    """
    db = SessionLocal()
    try:
        # ---------------------------------------------------------
        # Passo 1: Inicialização e Readiness
        # ---------------------------------------------------------
        init_db()
        r = health.readiness_check(db)
        assert r["status"] == "ready"
        assert r["database"] == "connected"

        # ---------------------------------------------------------
        # Passo 2: Auditoria de temporadas históricas (CA-01)
        # ---------------------------------------------------------
        report = audit_database()
        assert report["status"] == "PASS"
        assert len(report["seasons"]) >= 3
        for season_year, s_info in report["seasons"].items():
            assert s_info["complete"] is True
            assert s_info["match_count"] == 380

        # ---------------------------------------------------------
        # Passo 3: Verificação de corte temporal histórico T-24h
        # ---------------------------------------------------------
        test_match = db.query(Match).filter(Match.season == 2024, Match.round == 1).first()
        assert test_match is not None

        features_data = matches.get_match_features(test_match.match_id, db=db)
        as_of_at = datetime.fromisoformat(features_data["as_of_at"])
        if as_of_at.tzinfo is None:
            as_of_at = as_of_at.replace(tzinfo=timezone.utc)
        kickoff = test_match.kickoff_at
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        hours_diff = (kickoff - as_of_at).total_seconds() / 3600.0
        assert pytest.approx(hours_diff, abs=0.1) == 24.0

        # ---------------------------------------------------------
        # Passo 4: Tentar injetar notícia pós-cutoff (Deve rejeitar com 422)
        # ---------------------------------------------------------
        future_news = ManualNewsIngestionRequest(
            url="https://esporte.exemplo.com/noticia-pos-jogo-resultado",
            title="Goleada histórica confirmada",
            published_at="2026-11-20T22:00:00Z", # Pós-cutoff do jogo de 2024!
            first_seen_at="2026-11-20T22:05:00Z",
            source_name="exemplo_esporte",
            team_slug="flamengo",
            match_id=test_match.match_id,
            text_content="Fatos acontecidos após a realização da partida.",
            rights_status="ALLOWED"
        )
        with pytest.raises(HTTPException) as exc_info:
            ingestion.ingest_manual_news(future_news, x_admin_token="dev-secret-token-123", db=db)
        assert exc_info.value.status_code == 422
        assert "Temporal leakage violation" in str(exc_info.value.detail)

        # ---------------------------------------------------------
        # Passo 5: Ingerir notícia autorizada com lesão confirmada
        # ---------------------------------------------------------
        injury_news = ManualNewsIngestionRequest(
            url="https://esporte.exemplo.com/noticia-lesao-confirmada-pedro",
            title="Pedro sofre lesão muscular na coxa e desfalca a equipe",
            published_at="2024-04-10T10:00:00Z", # Antes do corte!
            first_seen_at="2024-04-10T10:05:00Z",
            source_name="ge_globo",
            team_slug="flamengo",
            match_id=test_match.match_id,
            text_content="Atacante teve confirmada lesão na coxa e está vetado pelo departamento médico.",
            rights_status="ALLOWED"
        )
        res_injury = ingestion.ingest_manual_news(injury_news, x_admin_token="dev-secret-token-123", db=db)
        assert res_injury.status == "ACCEPTED_AND_EXTRACTED"
        feats = res_injury.extracted_features["features"]
        assert feats["impacto_desfalques"] is not None
        assert feats["impacto_desfalques"] >= 5
        assert len(res_injury.extracted_features["evidence"]) > 0

        # ---------------------------------------------------------
        # Passo 6: Ingerir texto neutro sem fatos (mantém null e reporta cobertura)
        # ---------------------------------------------------------
        neutral_news = ManualNewsIngestionRequest(
            url="https://esporte.exemplo.com/noticia-treino-sem-novidades",
            title="Equipe realiza treino tático em campo reduzido",
            published_at="2024-04-11T09:00:00Z",
            first_seen_at="2024-04-11T09:05:00Z",
            source_name="portal_esporte",
            team_slug="sao_paulo",
            match_id=test_match.match_id,
            text_content="Treino técnico com foco em troca de passes, posse de bola e finalizações.",
            rights_status="ALLOWED"
        )
        res_neutral = ingestion.ingest_manual_news(neutral_news, x_admin_token="dev-secret-token-123", db=db)
        assert res_neutral.status == "ACCEPTED_AND_EXTRACTED"
        feats_neutral = res_neutral.extracted_features["features"]
        assert feats_neutral["pressao_crise"] is None
        assert feats_neutral["impacto_desfalques"] is None
        assert feats_neutral["prioridade_rotacao"] is None

        # ---------------------------------------------------------
        # Passo 7: Execução dos experimentos e métricas
        # ---------------------------------------------------------
        exp_res = run_experiment(BASE_DIR / "configs" / "experiment_v1.yaml")
        assert "B1_numeric" in exp_res["metrics"]
        assert "M1_contextual" in exp_res["metrics"]
        assert "G1_poisson" in exp_res["metrics"]
        assert "promotion_gate" in exp_res
        assert exp_res["promotion_gate"]["active_model"] in ["numeric_lgbm", "contextual_lgbm"]

        # ---------------------------------------------------------
        # Passo 8: Visualizar probabilidades e placares
        # ---------------------------------------------------------
        sim_req = PredictionSimulationRequest(match_id=test_match.match_id, model_type="numeric_lgbm")
        pred_res = predictions.simulate_prediction(sim_req, db=db)
        p = pred_res.probabilities
        assert 0.0 <= p.home <= 1.0 and 0.0 <= p.draw <= 1.0 and 0.0 <= p.away <= 1.0
        assert pytest.approx(p.home + p.draw + p.away, abs=1e-3) == 1.0
        assert len(pred_res.score_model.top_scores) >= 3

        # ---------------------------------------------------------
        # Passo 9: Explicação referenciada com evidências e fallback
        # ---------------------------------------------------------
        exp_req = ExplanationRequest(match_id=test_match.match_id)
        exp_output = explanations.generate_match_explanation(exp_req, db=db)
        assert len(exp_output.explanation) > 50
        assert "não implicam nexo de causalidade" in exp_output.disclaimer

        # ---------------------------------------------------------
        # Passo 10: Repositório sem segredos e com relatórios salvos
        # ---------------------------------------------------------
        assert (BASE_DIR / "reports" / "coverage_report.json").exists()
        assert (BASE_DIR / "reports" / "leakage_report.json").exists()
        assert (BASE_DIR / "reports" / "experiment_results.json").exists()
        assert (BASE_DIR / "artifacts" / "b1_numeric_lgbm.pkl").exists()

    finally:
        db.close()
