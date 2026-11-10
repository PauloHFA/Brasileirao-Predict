from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from src.schemas.models import Article, Match


def get_cutoff_time(kickoff_at: datetime, as_of_hours: int = 24) -> datetime:
    """Computes exact pre-match cutoff instant t* = kickoff_at - timedelta(hours=as_of_hours)."""
    if kickoff_at.tzinfo is None:
        kickoff_at = kickoff_at.replace(tzinfo=timezone.utc)
    return kickoff_at - timedelta(hours=as_of_hours)

def get_eligible_past_matches(db: Session, as_of_at: datetime) -> List[Match]:
    """Retrieves exclusively matches finished strictly before cutoff instant t*."""
    return (
        db.query(Match)
        .filter(Match.kickoff_at < as_of_at, Match.status == "FINISHED")
        .order_by(Match.kickoff_at.asc())
        .all()
    )

def is_article_eligible(article: Article, as_of_at: datetime) -> Tuple[bool, Optional[str]]:
    """Strict eligibility check for textual context to prevent data leakage."""
    pub_at = article.published_at
    if pub_at.tzinfo is None:
        pub_at = pub_at.replace(tzinfo=timezone.utc)

    first_seen = article.first_seen_at
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)

    if pub_at > as_of_at:
        return False, f"Article published_at ({pub_at.isoformat()}) is after cutoff ({as_of_at.isoformat()})"

    if first_seen > as_of_at:
        return False, f"Article first_seen_at ({first_seen.isoformat()}) is after cutoff ({as_of_at.isoformat()})"

    if article.rights_status not in ("ALLOWED", "METADATA_ONLY"):
        return False, f"Article rights_status ({article.rights_status}) is blocked or unauthorized"

    return True, None
