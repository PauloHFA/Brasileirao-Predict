import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import func

from src.config import BASE_DIR
from src.database import SessionLocal
from src.schemas.models import Match, MatchSource, Team


def audit_database() -> Dict[str, Any]:
    db = SessionLocal()
    try:
        total_matches = db.query(Match).count()
        total_teams = db.query(Team).count()
        sources = db.query(MatchSource).all()

        seasons_coverage = {}
        issues: List[str] = []

        # Group by season
        season_stats = (
            db.query(
                Match.season,
                func.count(Match.match_id).label("match_count"),
                func.min(Match.kickoff_at).label("first_match"),
                func.max(Match.kickoff_at).label("last_match"),
                func.count(func.distinct(Match.round)).label("rounds_count")
            )
            .group_by(Match.season)
            .order_by(Match.season)
            .all()
        )

        for s_stat in season_stats:
            season = s_stat.season
            count = s_stat.match_count
            rounds = s_stat.rounds_count

            # Check for exactly 380 matches and 38 rounds for Serie A
            if count != 380:
                issues.append(f"Season {season} has {count} matches (expected 380).")
            if rounds != 38:
                issues.append(f"Season {season} has {rounds} distinct rounds (expected 38).")

            seasons_coverage[str(season)] = {
                "match_count": count,
                "rounds_count": rounds,
                "first_match": s_stat.first_match.isoformat() if s_stat.first_match else None,
                "last_match": s_stat.last_match.isoformat() if s_stat.last_match else None,
                "complete": count == 380 and rounds == 38
            }

        # Check for invalid goals (negative)
        invalid_goals = db.query(Match).filter(
            (Match.goals_home < 0) | (Match.goals_away < 0)
        ).count()
        if invalid_goals > 0:
            issues.append(f"Found {invalid_goals} matches with negative goals.")

        # Check same team playing against itself
        same_team_matches = db.query(Match).filter(Match.home_team_id == Match.away_team_id).count()
        if same_team_matches > 0:
            issues.append(f"Found {same_team_matches} matches where home_team == away_team.")

        # Check duplicate match round combinations
        duplicate_check = (
            db.query(Match.season, Match.round, Match.home_team_id, Match.away_team_id, func.count())
            .group_by(Match.season, Match.round, Match.home_team_id, Match.away_team_id)
            .having(func.count() > 1)
            .all()
        )
        if duplicate_check:
            issues.append(f"Found {len(duplicate_check)} duplicate match entries.")

        sources_info = [
            {
                "source_name": s.source_name,
                "version": s.version,
                "sha256": s.sha256,
                "fetched_at": s.fetched_at.isoformat()
            }
            for s in sources
        ]

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if not issues else "FAIL",
            "total_matches": total_matches,
            "total_teams": total_teams,
            "seasons": seasons_coverage,
            "sources": sources_info,
            "issues": issues,
            "integrity_checks": {
                "unique_matches": len(duplicate_check) == 0,
                "distinct_teams": same_team_matches == 0,
                "non_negative_goals": invalid_goals == 0,
                "at_least_3_seasons": len(seasons_coverage) >= 3,
            }
        }

        # Save to reports
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / "coverage_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
    finally:
        db.close()

if __name__ == "__main__":
    report = audit_database()
    print("=== Relatório de Auditoria de Dados ===")
    print(f"Status: {report['status']}")
    print(f"Total de Partidas: {report['total_matches']}")
    print(f"Temporadas Auditadas: {list(report['seasons'].keys())}")
    for s, d in report['seasons'].items():
        print(f"  - Temporada {s}: {d['match_count']} jogos, {d['rounds_count']} rodadas (Completa: {d['complete']})")
    if report["issues"]:
        print(f"Problemas encontrados: {report['issues']}")
    else:
        print("Integridade verificada: 100% em conformidade com RNF-02 e CA-01.")
