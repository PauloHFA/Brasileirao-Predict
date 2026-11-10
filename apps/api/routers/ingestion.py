import hashlib
import html
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.config import settings
from src.data_quality.leakage_guard import verify_news_temporal_validity
from src.database import get_db
from src.entity_resolution.teams import resolver
from src.llm_extraction.extractor import extract_and_save_article
from src.schemas.api import ManualNewsIngestionRequest, ManualNewsIngestionResponse
from src.schemas.models import Article, ArticleEntity, FeatureSnapshot, Match

router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion"])

@router.post("/manual", response_model=ManualNewsIngestionResponse)
def ingest_manual_news(
    req: ManualNewsIngestionRequest,
    x_admin_token: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    # Security check: verify admin token if configured
    if settings.ADMIN_API_TOKEN and x_admin_token != settings.ADMIN_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-Admin-Token")

    # Sanitize inputs
    sanitized_title = html.escape(req.title.strip())
    sanitized_text = html.escape(req.text_content.strip())
    content_sha256 = hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest()

    try:
        pub_dt = datetime.fromisoformat(req.published_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ISO format for published_at.")

    if pub_dt.tzinfo is None:
        pub_dt = pub_dt.replace(tzinfo=timezone.utc)

    if req.first_seen_at:
        try:
            first_seen_dt = datetime.fromisoformat(req.first_seen_at)
            if first_seen_dt.tzinfo is None:
                first_seen_dt = first_seen_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid ISO format for first_seen_at.")
    else:
        first_seen_dt = pub_dt

    # If linked to a specific match, enforce temporal cutoff!
    if req.match_id:
        match = db.query(Match).filter(Match.match_id == req.match_id).first()
        if not match:
            raise HTTPException(status_code=404, detail="Match not found.")

        snapshot = (
            db.query(FeatureSnapshot)
            .filter(FeatureSnapshot.match_id == req.match_id)
            .order_by(FeatureSnapshot.as_of_at.desc())
            .first()
        )
        cutoff = snapshot.as_of_at if snapshot else match.kickoff_at

        is_valid, reason = verify_news_temporal_validity(
            article_published_at=pub_dt,
            article_first_seen_at=first_seen_dt,
            as_of_at=cutoff,
            rights_status=req.rights_status
        )

        if not is_valid:
            # Rejection mandated by Section 9 & 11
            raise HTTPException(status_code=422, detail=f"Rejection: {reason}")

    # Resolve team
    team_info = resolver.resolve(req.team_slug)
    team_id = team_info["id"] if team_info else None
    team_name = team_info["canonical_name"] if team_info else req.team_slug

    # Check duplicate canonical_url
    existing = db.query(Article).filter(Article.canonical_url == req.url).first()
    if existing:
        article = existing
        article.title = sanitized_title
        article.body_summary = sanitized_text[:1000]
    else:
        article = Article(
            canonical_url=req.url,
            title=sanitized_title,
            published_at=pub_dt,
            first_seen_at=first_seen_dt,
            source_name=req.source_name,
            rights_status=req.rights_status,
            content_sha256=content_sha256,
            body_summary=sanitized_text[:1000]
        )
        db.add(article)
        db.flush()

        # Link entity
        if team_id or req.match_id:
            link = ArticleEntity(
                article_id=article.article_id,
                team_id=team_id,
                match_id=req.match_id,
                relevance=1.0
            )
            db.add(link)

    db.commit()

    # Run structured extraction
    as_of_str = pub_dt.isoformat()
    ext_payload = extract_and_save_article(
        db=db,
        article=article,
        team_slug=req.team_slug,
        team_name=team_name,
        as_of_at=as_of_str,
        match_id=str(req.match_id) if req.match_id else None
    )

    return ManualNewsIngestionResponse(
        article_id=article.article_id,
        status="ACCEPTED_AND_EXTRACTED",
        message="Article ingested, verified and extracted successfully.",
        extracted_features=ext_payload.model_dump()
    )
