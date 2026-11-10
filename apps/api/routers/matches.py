from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.api import MatchItemResponse
from src.schemas.models import Article, ArticleEntity, FeatureSnapshot, LLMExtraction, Match, Team

router = APIRouter(prefix="/api/v1", tags=["Matches"])

@router.get("/seasons")
def list_seasons(db: Session = Depends(get_db)):
    results = (
        db.query(
            Match.season,
            func.count(Match.match_id).label("total_matches"),
            func.count(func.distinct(Match.round)).label("rounds_count")
        )
        .group_by(Match.season)
        .order_by(Match.season.desc())
        .all()
    )
    return [
        {
            "season": r.season,
            "total_matches": r.total_matches,
            "rounds_count": r.rounds_count,
            "complete": r.total_matches == 380
        }
        for r in results
    ]

@router.get("/matches", response_model=List[MatchItemResponse])
def list_matches(
    season: Optional[int] = Query(None),
    round: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = (
        db.query(
            Match,
            Team.canonical_name.label("home_name")
        )
        .join(Team, Match.home_team_id == Team.team_id)
        .order_by(Match.kickoff_at.asc())
    )

    if season:
        query = query.filter(Match.season == season)
    if round:
        query = query.filter(Match.round == round)
    if status:
        query = query.filter(Match.status == status)

    items = query.all()

    # Preload teams dict for fast lookup
    teams_dict = {t.team_id: t.canonical_name for t in db.query(Team).all()}

    output = []
    for match, _ in items:
        output.append(
            MatchItemResponse(
                match_id=match.match_id,
                season=match.season,
                round=match.round,
                kickoff_at=match.kickoff_at.isoformat(),
                home_team_id=match.home_team_id,
                away_team_id=match.away_team_id,
                home_team_name=teams_dict.get(match.home_team_id, f"Team {match.home_team_id}"),
                away_team_name=teams_dict.get(match.away_team_id, f"Team {match.away_team_id}"),
                goals_home=match.goals_home,
                goals_away=match.goals_away,
                status=match.status
            )
        )
    return output

@router.get("/matches/{match_id}/features")
def get_match_features(match_id: int, as_of: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(FeatureSnapshot).filter(FeatureSnapshot.match_id == match_id)
    if as_of:
        query = query.filter(FeatureSnapshot.as_of_at <= as_of)
    snapshot = query.order_by(FeatureSnapshot.as_of_at.desc()).first()

    if not snapshot:
        raise HTTPException(status_code=404, detail="Feature snapshot not found for this match.")

    return {
        "match_id": match_id,
        "snapshot_id": snapshot.snapshot_id,
        "as_of_at": snapshot.as_of_at.isoformat(),
        "feature_version": snapshot.feature_version,
        "snapshot_hash": snapshot.snapshot_hash,
        "features": snapshot.features,
        "news_coverage": snapshot.news_coverage
    }

@router.get("/matches/{match_id}/evidence")
def get_match_evidence(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    snapshot = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.match_id == match_id)
        .order_by(FeatureSnapshot.as_of_at.desc())
        .first()
    )
    cutoff = snapshot.as_of_at if snapshot else match.kickoff_at

    articles = (
        db.query(Article, LLMExtraction)
        .join(ArticleEntity, Article.article_id == ArticleEntity.article_id)
        .outerjoin(LLMExtraction, Article.article_id == LLMExtraction.article_id)
        .filter(
            (ArticleEntity.match_id == match_id) |
            (ArticleEntity.team_id.in_([match.home_team_id, match.away_team_id])),
            Article.published_at <= cutoff,
            Article.rights_status.in_(["ALLOWED", "METADATA_ONLY"])
        )
        .all()
    )

    evidence_list = []
    for art, ext in articles:
        extracted = ext.extracted_json if ext else {}
        evidence_list.append({
            "article_id": art.article_id,
            "title": art.title,
            "source_name": art.source_name,
            "canonical_url": art.canonical_url,
            "published_at": art.published_at.isoformat(),
            "rights_status": art.rights_status,
            "features_extracted": extracted.get("features", {}),
            "evidence_spans": extracted.get("evidence", [])
        })

    return {
        "match_id": match_id,
        "cutoff_as_of": cutoff.isoformat(),
        "articles_count": len(evidence_list),
        "articles": evidence_list
    }
