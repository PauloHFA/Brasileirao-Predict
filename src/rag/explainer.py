from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.schemas.models import Article, ArticleEntity, FeatureSnapshot, LLMExtraction, Match, Team

DISCLAIMER_CAUSALITY = (
    "Aviso metodológico: Associações estatísticas refletem correlações empíricas no conjunto de treino "
    "e não implicam nexo de causalidade direta. Probabilidades estimadas não constituem certeza de resultado "
    "nem recomendação de apostas."
)

class ExplanationGenerator:
    """
    RAG-grounded pre-match explanation generator.
    Cites verified evidence IDs from before cutoff t*, reports lack of evidence, and disclaims causal claims.
    """

    def generate_explanation(
        self,
        db: Session,
        match_id: int,
        snapshot_id: Optional[int] = None
    ) -> Dict[str, Any]:
        match = db.query(Match).filter(Match.match_id == match_id).first()
        if not match:
            raise ValueError(f"Match {match_id} not found.")

        # Find snapshot
        if snapshot_id:
            snapshot = db.query(FeatureSnapshot).filter(FeatureSnapshot.snapshot_id == snapshot_id).first()
        else:
            snapshot = (
                db.query(FeatureSnapshot)
                .filter(FeatureSnapshot.match_id == match_id)
                .order_by(FeatureSnapshot.as_of_at.desc())
                .first()
            )

        if not snapshot:
            raise ValueError(f"No feature snapshot found for match {match_id}.")

        home_team = db.query(Team).filter(Team.team_id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.team_id == match.away_team_id).first()
        home_name = home_team.canonical_name if home_team else f"Team {match.home_team_id}"
        away_name = away_team.canonical_name if away_team else f"Team {match.away_team_id}"

        feats = snapshot.features or {}
        elo_diff = feats.get("elo_diff_pre", 0.0)
        form_home = feats.get("form_points_5_home", 0.0)
        form_away = feats.get("form_points_5_away", 0.0)

        # Retrieve eligible news articles before cutoff
        cutoff = snapshot.as_of_at
        articles = (
            db.query(Article, LLMExtraction)
            .join(ArticleEntity, Article.article_id == ArticleEntity.article_id)
            .outerjoin(LLMExtraction, Article.article_id == LLMExtraction.article_id)
            .filter(
                (ArticleEntity.match_id == match_id) |
                (ArticleEntity.team_id.in_([match.home_team_id, match.away_team_id])),
                Article.published_at <= cutoff,
                Article.first_seen_at <= cutoff,
                Article.rights_status.in_(["ALLOWED", "METADATA_ONLY"])
            )
            .all()
        )

        evidence_ids: List[str] = []
        doc_urls: List[str] = []
        factual_statements: List[str] = []

        for art, ext in articles:
            doc_urls.append(art.canonical_url)
            if ext and ext.extracted_json:
                ext_payload = ext.extracted_json
                ev_list = ext_payload.get("evidence", [])
                for ev in ev_list:
                    ev_id = ev.get("id", f"DOC_{art.article_id}")
                    evidence_ids.append(ev_id)
                    quote = ev.get("quote", "")
                    field = ev.get("field", "")
                    cert = ev.get("certainty", "reported")
                    factual_statements.append(
                        f"[{ev_id}] Em relação a {field} ({cert}): \"{quote}\" (Fonte: {art.source_name})"
                    )

        # Build structured explanation
        lines = []
        lines.append(
            f"Análise pré-jogo para {home_name} vs {away_name} (Corte temporal: {cutoff.strftime('%d/%m/%Y %H:%M UTC')})."
        )
        lines.append(
            f"Fatores Numéricos (Baseline B1): O confronto apresenta diferença de rating Elo de {elo_diff:+.1f} a favor do "
            f"{home_name if elo_diff >= 0 else away_name}. "
            f"Aproveitamento recente nos últimos 5 jogos: {home_name} ({form_home:.2f} pts/jogo) e {away_name} ({form_away:.2f} pts/jogo)."
        )

        if factual_statements:
            lines.append("Evidências Contextuais Auditadas no Pré-Jogo:")
            for stmt in factual_statements:
                lines.append(f" - {stmt}")
        else:
            lines.append(
                "Evidências Contextuais: Não foram encontradas reportagens autorizadas com fatos documentados antes do corte temporal. "
                "O modelo mantém a estimativa exclusivamente sobre os fundamentos estatísticos e baseline numérico."
            )

        lines.append(DISCLAIMER_CAUSALITY)

        return {
            "match_id": match_id,
            "snapshot_id": snapshot.snapshot_id,
            "cutoff_timestamp": cutoff.isoformat(),
            "explanation": "\n\n".join(lines),
            "evidence_ids": list(set(evidence_ids)),
            "doc_urls": list(set(doc_urls)),
            "disclaimer": DISCLAIMER_CAUSALITY
        }
