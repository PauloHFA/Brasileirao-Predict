import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.config import BASE_DIR
from src.database import SessionLocal
from src.schemas.models import Article, ArticleEntity, FeatureSnapshot, Match


def audit_temporal_leakage_for_test_set(
    db: Session,
    test_seasons: List[int],
    feature_version: str = "v1.0"
) -> Dict[str, Any]:
    """
    Audits 100% of matches in test seasons to verify that zero features, past matches or
    news articles have timestamps posterior to match pre-game cutoff as_of_at.
    """
    matches = (
        db.query(Match, FeatureSnapshot)
        .join(FeatureSnapshot, Match.match_id == FeatureSnapshot.match_id)
        .filter(
            Match.season.in_(test_seasons),
            FeatureSnapshot.feature_version == feature_version
        )
        .all()
    )

    violations: List[Dict[str, Any]] = []
    audited_count = len(matches)

    for match, snapshot in matches:
        as_of_at = snapshot.as_of_at
        if as_of_at.tzinfo is None:
            as_of_at = as_of_at.replace(tzinfo=timezone.utc)

        # 1. Check match kickoff is strictly after as_of_at
        kickoff = match.kickoff_at
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)

        if kickoff <= as_of_at:
            violations.append({
                "match_id": match.match_id,
                "type": "MATCH_KICKOFF_BEFORE_CUTOFF",
                "kickoff_at": kickoff.isoformat(),
                "as_of_at": as_of_at.isoformat()
            })

        # 2. Check all linked articles for this match
        linked_articles = (
            db.query(Article)
            .join(ArticleEntity, Article.article_id == ArticleEntity.article_id)
            .filter(
                (ArticleEntity.match_id == match.match_id) |
                (ArticleEntity.team_id.in_([match.home_team_id, match.away_team_id]))
            )
            .all()
        )

        for art in linked_articles:
            pub_at = art.published_at.replace(tzinfo=timezone.utc) if art.published_at.tzinfo is None else art.published_at
            seen_at = art.first_seen_at.replace(tzinfo=timezone.utc) if art.first_seen_at.tzinfo is None else art.first_seen_at

            if pub_at > as_of_at:
                violations.append({
                    "match_id": match.match_id,
                    "article_id": art.article_id,
                    "type": "ARTICLE_PUBLISHED_AFTER_CUTOFF",
                    "article_published_at": pub_at.isoformat(),
                    "as_of_at": as_of_at.isoformat()
                })

            if seen_at > as_of_at:
                violations.append({
                    "match_id": match.match_id,
                    "article_id": art.article_id,
                    "type": "ARTICLE_FIRST_SEEN_AFTER_CUTOFF",
                    "article_first_seen_at": seen_at.isoformat(),
                    "as_of_at": as_of_at.isoformat()
                })

    status = "PASS" if not violations else "FAIL"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "test_seasons": test_seasons,
        "matches_audited": audited_count,
        "violations_count": len(violations),
        "violations": violations
    }

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "leakage_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    session = SessionLocal()
    try:
        rep = audit_temporal_leakage_for_test_set(session, test_seasons=[2024])
        print(f"Leakage Audit Status: {rep['status']}")
        print(f"Audited {rep['matches_audited']} test matches. Violations: {rep['violations_count']}")
    finally:
        session.close()
