import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.database import SessionLocal
from src.entity_resolution.teams import resolver
from src.llm_extraction.extractor import extract_and_save_article
from src.schemas.models import Article, ArticleEntity, FeatureSnapshot, Match

# Default reliable public sports RSS feeds for Série A news metadata
DEFAULT_RSS_FEEDS = [
    {
        "name": "ge_globo_futebol",
        "url": "https://ge.globo.com/dynamo/futebol/rss2.xml",
        "rights_status": "METADATA_ONLY"
    },
    {
        "name": "uol_esporte",
        "url": "https://rss.uol.com.br/feed/esporte.xml",
        "rights_status": "METADATA_ONLY"
    }
]

def clean_html_tags(raw_html: str) -> str:
    """Strips HTML tags from summaries to preserve clean text."""
    if not raw_html:
        return ""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return " ".join(clean.split())

def parse_rss_feed(feed_url: str, source_name: str, rights_status: str = "METADATA_ONLY") -> List[Dict[str, Any]]:
    """
    Parses a remote or local RSS 2.0 XML feed and extracts standardized article metadata.
    """
    articles = []
    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "BrasileiraoPredictorAI/1.0 (Research bot; +http://localhost)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        channel = root.find("channel")
        if channel is None:
            return articles

        for item in channel.findall("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            description = item.findtext("description", "").strip()
            pub_date_str = item.findtext("pubDate", "").strip()

            if not title or not link:
                continue

            # Parse published datetime
            published_at = datetime.now(timezone.utc)
            if pub_date_str:
                try:
                    dt = parsedate_to_datetime(pub_date_str)
                    published_at = dt.astimezone(timezone.utc)
                except Exception:
                    published_at = datetime.now(timezone.utc)

            content_text = clean_html_tags(description)
            sha256_hash = hashlib.sha256(f"{title} {content_text}".encode("utf-8")).hexdigest()

            articles.append({
                "canonical_url": link,
                "title": title,
                "published_at": published_at,
                "first_seen_at": datetime.now(timezone.utc),
                "source_name": source_name,
                "rights_status": rights_status,
                "content_sha256": sha256_hash,
                "body_summary": content_text[:800]
            })
    except Exception as e:
        print(f"Warning: Could not fetch or parse RSS feed '{feed_url}': {e}")

    return articles

class RSSNewsCollector:
    """
    Automated collector that ingests public RSS feeds, links articles to Série A teams and matches,
    and runs structured LLM extraction within pre-match eligibility windows.
    """

    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self.should_close_db = False

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
            self.should_close_db = True
        return self._db

    def close(self):
        if self.should_close_db and self._db is not None:
            self._db.close()
            self._db = None

    def match_teams_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Identifies any mention of Série A teams in article title and text."""
        detected_teams = []
        lower_text = text.lower()

        for slug, info in resolver.canonical_teams.items():
            canonical = info["canonical_name"].lower()
            if canonical in lower_text:
                detected_teams.append({"slug": slug, "id": info["id"], "name": info["canonical_name"]})
                continue
            for alias in info.get("aliases", []):
                if len(alias) >= 4 and f" {alias} " in f" {lower_text} ":
                    detected_teams.append({"slug": slug, "id": info["id"], "name": info["canonical_name"]})
                    break

        return detected_teams

    def find_associated_match(self, team_id: int, published_at: datetime) -> Optional[int]:
        """
        Finds the closest upcoming match for this team where published_at is strictly
        before the match pre-game cutoff (kickoff - 24h).
        """
        upcoming_matches = (
            self.db.query(Match)
            .filter(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
                Match.kickoff_at > published_at
            )
            .order_by(Match.kickoff_at.asc())
            .limit(3)
            .all()
        )

        for match in upcoming_matches:
            # Check cutoff
            snapshot = (
                self.db.query(FeatureSnapshot)
                .filter(FeatureSnapshot.match_id == match.match_id)
                .order_by(FeatureSnapshot.as_of_at.desc())
                .first()
            )
            cutoff = snapshot.as_of_at if snapshot else match.kickoff_at
            cutoff_aware = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff

            if published_at <= cutoff_aware:
                return match.match_id

        return None

    def collect_and_ingest(self, feeds: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes ingestion run across all configured RSS feeds.
        """
        target_feeds = feeds or DEFAULT_RSS_FEEDS
        total_discovered = 0
        total_inserted = 0
        total_linked = 0
        total_extracted = 0

        for f_info in target_feeds:
            items = parse_rss_feed(
                f_info["url"],
                source_name=f_info["name"],
                rights_status=f_info.get("rights_status", "METADATA_ONLY")
            )
            total_discovered += len(items)

            for art_data in items:
                # Deduplication check
                existing = self.db.query(Article).filter(Article.canonical_url == art_data["canonical_url"]).first()
                if existing:
                    continue

                article = Article(
                    canonical_url=art_data["canonical_url"],
                    title=art_data["title"],
                    published_at=art_data["published_at"],
                    first_seen_at=art_data["first_seen_at"],
                    source_name=art_data["source_name"],
                    rights_status=art_data["rights_status"],
                    content_sha256=art_data["content_sha256"],
                    body_summary=art_data["body_summary"]
                )
                self.db.add(article)
                self.db.flush()
                total_inserted += 1

                # Detect teams
                full_text = f"{art_data['title']} {art_data['body_summary']}"
                teams_found = self.match_teams_in_text(full_text)

                for team in teams_found:
                    match_id = self.find_associated_match(team["id"], art_data["published_at"])
                    link = ArticleEntity(
                        article_id=article.article_id,
                        team_id=team["id"],
                        match_id=match_id,
                        relevance=1.0
                    )
                    self.db.add(link)
                    total_linked += 1

                    # Run structured LLM extraction for linked team
                    try:
                        extract_and_save_article(
                            db=self.db,
                            article=article,
                            team_slug=team["slug"],
                            team_name=team["name"],
                            as_of_at=art_data["published_at"].isoformat(),
                            match_id=str(match_id) if match_id else None
                        )
                        total_extracted += 1
                    except Exception as e:
                        print(f"Extraction error for article {article.article_id}: {e}")

        self.db.commit()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feeds_polled": len(target_feeds),
            "total_discovered": total_discovered,
            "total_inserted": total_inserted,
            "total_linked": total_linked,
            "total_extracted": total_extracted
        }

if __name__ == "__main__":
    collector = RSSNewsCollector()
    try:
        res = collector.collect_and_ingest()
        print("=== RSS Ingestion Completed ===")
        print(f"Feeds Polled:     {res['feeds_polled']}")
        print(f"Discovered:       {res['total_discovered']}")
        print(f"New Articles:     {res['total_inserted']}")
        print(f"Entity Links:     {res['total_linked']}")
        print(f"LLM Extractions:  {res['total_extracted']}")
    finally:
        collector.close()
