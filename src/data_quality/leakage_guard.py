from datetime import datetime, timezone
from typing import Optional, Tuple


def verify_news_temporal_validity(
    article_published_at: datetime,
    article_first_seen_at: datetime,
    as_of_at: datetime,
    rights_status: str
) -> Tuple[bool, Optional[str]]:
    """
    Validates temporal eligibility of news item against cutoff timestamp.
    Returns (True, None) if eligible, or (False, rejection_reason).
    """
    if article_published_at.tzinfo is None:
        article_published_at = article_published_at.replace(tzinfo=timezone.utc)
    if article_first_seen_at.tzinfo is None:
        article_first_seen_at = article_first_seen_at.replace(tzinfo=timezone.utc)
    if as_of_at.tzinfo is None:
        as_of_at = as_of_at.replace(tzinfo=timezone.utc)

    if rights_status not in ("ALLOWED", "METADATA_ONLY"):
        return False, f"Rights violation: rights_status is '{rights_status}', must be ALLOWED or METADATA_ONLY"

    if article_published_at > as_of_at:
        return False, (
            f"Temporal leakage violation: published_at ({article_published_at.isoformat()}) "
            f"is after cutoff as_of_at ({as_of_at.isoformat()})"
        )

    if article_first_seen_at > as_of_at:
        return False, (
            f"Temporal leakage violation: first_seen_at ({article_first_seen_at.isoformat()}) "
            f"is after cutoff as_of_at ({as_of_at.isoformat()})"
        )

    return True, None
