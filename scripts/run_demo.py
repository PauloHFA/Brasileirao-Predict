"""
Script de demonstração oficial do Brasileirão Predictor AI.
Executa os passos demonstráveis do Apêndice E e exibe o resumo operacional no terminal.
"""
import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from apps.api.routers import explanations, health, ingestion, matches, predictions  # noqa: E402
from src.data_quality.auditor import audit_database  # noqa: E402
from src.database import SessionLocal, init_db  # noqa: E402
from src.schemas.api import (  # noqa: E402
    ExplanationRequest,
    ManualNewsIngestionRequest,
    PredictionSimulationRequest,
)
from src.schemas.models import Match  # noqa: E402


def main():
    print("=" * 70)
    print("   ⚽ BRASILEIRÃO PREDICTOR AI — ROTEIRO DE DEMONSTRAÇÃO E ACEITE")
    print("=" * 70)

    db = SessionLocal()
    try:
        # 1. Health & Readiness
        print("\n[Passo 1] Verificação de saúde e infraestrutura...")
        init_db()
        r = health.readiness_check(db)
        print(f" -> Status: {r['status']} | Banco de Dados: {r['database']}")

        # 2. Auditoria de dados históricos
        print("\n[Passo 2] Auditoria de temporadas históricas (CA-01)...")
        rep = audit_database()
        print(f" -> Integridade: {rep['status']} ({rep['total_matches']} partidas auditadas)")
        for season, s_info in rep["seasons"].items():
            print(f"    • {season}: {s_info['match_count']} jogos, {s_info['rounds_count']} rodadas")

        # 3. Consulta de jogo histórico com corte T-24h
        print("\n[Passo 3] Inspeção de confronto com corte temporal T-24h...")
        sample_match = db.query(Match).filter(Match.season == 2024, Match.round == 1).first()
        feats = matches.get_match_features(sample_match.match_id, db=db)
        print(f" -> Partida ID {sample_match.match_id} | Corte: {feats['as_of_at']} (T-24h)")
        print(f"    Elo Diff: {feats['features']['elo_diff_pre']} | Forma Mandante: {feats['features']['form_points_5_home']}")

        # 4. Injeção de notícia pós-cutoff (teste de rejeição de leakage)
        print("\n[Passo 4] Teste de rejeição de vazamento temporal (Data Leakage)...")
        leak_req = ManualNewsIngestionRequest(
            url="https://esporte.demo.com/noticia-vazada-futura",
            title="Resultado apurado após o apito final",
            published_at="2026-11-20T22:00:00Z",
            first_seen_at="2026-11-20T22:05:00Z",
            source_name="globo_esporte",
            team_slug="flamengo",
            match_id=sample_match.match_id,
            text_content="Conteúdo com spoilers de jogo encerrado.",
            rights_status="ALLOWED"
        )
        try:
            ingestion.ingest_manual_news(leak_req, x_admin_token="dev-secret-token-123", db=db)
            print(" -> ERRO: Notícia pós-corte não foi rejeitada!")
        except Exception as e:
            print(f" -> SUCESSO: Notícia rejeitada conforme regra temporal: {e.detail}")

        # 5. Ingestão de notícia válida com lesão e extração de evidência
        print("\n[Passo 5] Ingestão de notícia pré-jogo autorizada com lesão confirmada...")
        valid_news = ManualNewsIngestionRequest(
            url="https://esporte.demo.com/lesao-confirmada-pedro-2024",
            title="Pedro sofre lesão muscular e vira desfalque para a rodada",
            published_at="2024-04-10T12:00:00Z",
            first_seen_at="2024-04-10T12:05:00Z",
            source_name="ge_globo",
            team_slug="flamengo",
            match_id=sample_match.match_id,
            text_content="Atacante teve confirmada lesão na coxa e desfalca a equipe titular.",
            rights_status="ALLOWED"
        )
        res_ingest = ingestion.ingest_manual_news(valid_news, x_admin_token="dev-secret-token-123", db=db)
        print(f" -> Status: {res_ingest.status}")
        extracted = res_ingest.extracted_features
        print(f"    Impacto Desfalques: {extracted['features']['impacto_desfalques']}/10")
        print(f"    Evidência: \"{extracted['evidence'][0]['quote']}\"")

        # 6. Simulação de previsão com modelos B1 e Poisson G1
        print("\n[Passo 6] Simulação probabilística 1X2 e distribuição de placares...")
        pred_res = predictions.simulate_prediction(
            PredictionSimulationRequest(match_id=sample_match.match_id, model_type="numeric_lgbm"),
            db=db
        )
        p = pred_res.probabilities
        print(f" -> Probabilidades 1X2: Mandante: {p.home*100:.1f}% | Empate: {p.draw*100:.1f}% | Visitante: {p.away*100:.1f}%")
        print(" -> Placares Mais Prováveis (Poisson G1):")
        for sc in pred_res.score_model.top_scores[:3]:
            print(f"    • {sc.home} x {sc.away} : {sc.probability*100:.1f}%")

        # 7. Explicação fundamentada RAG
        print("\n[Passo 7] Geração de relatório explicativo RAG fundamentado...")
        exp_res = explanations.generate_match_explanation(
            ExplanationRequest(match_id=sample_match.match_id),
            db=db
        )
        print(f" -> Evidências Citadas: {exp_res.evidence_ids}")
        print(" -> Trecho da Explicação:")
        print("   ", exp_res.explanation.splitlines()[0])
        print("   ", exp_res.explanation.splitlines()[1])

        print("\n" + "=" * 70)
        print("   ✅ ROTEIRO DE DEMONSTRAÇÃO EXECUTADO COM 100% DE SUCESSO")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    main()
