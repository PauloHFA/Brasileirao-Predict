from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    aliases: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    home_matches: Mapped[List["Match"]] = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches: Mapped[List["Match"]] = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    round: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.team_id"), nullable=False)
    goals_home: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    goals_away: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED", nullable=False) # SCHEDULED, FINISHED, POSTPONED

    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    snapshots: Mapped[List["FeatureSnapshot"]] = relationship("FeatureSnapshot", back_populates="match", cascade="all, delete-orphan")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="match", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("home_team_id <> away_team_id", name="chk_distinct_teams"),
        CheckConstraint("goals_home IS NULL OR goals_home >= 0", name="chk_goals_home_positive"),
        CheckConstraint("goals_away IS NULL OR goals_away >= 0", name="chk_goals_away_positive"),
        UniqueConstraint("season", "round", "home_team_id", "away_team_id", name="uq_match_round_teams"),
        Index("ix_matches_kickoff", "kickoff_at"),
    )

class MatchSource(Base):
    __tablename__ = "match_sources"

    source_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

class Article(Base):
    __tablename__ = "articles"

    article_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_url: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    rights_status: Mapped[str] = mapped_column(String(20), nullable=False) # ALLOWED, METADATA_ONLY, BLOCKED
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    body_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    extractions: Mapped[List["LLMExtraction"]] = relationship("LLMExtraction", back_populates="article", cascade="all, delete-orphan")
    entities: Mapped[List["ArticleEntity"]] = relationship("ArticleEntity", back_populates="article", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("rights_status IN ('ALLOWED', 'METADATA_ONLY', 'BLOCKED')", name="chk_rights_status"),
        Index("ix_articles_first_seen", "first_seen_at"),
    )

class ArticleEntity(Base):
    __tablename__ = "article_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.article_id"), nullable=False)
    team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.team_id"), nullable=True)
    match_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("matches.match_id"), nullable=True)
    relevance: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    article: Mapped["Article"] = relationship("Article", back_populates="entities")

class LLMExtraction(Base):
    __tablename__ = "llm_extractions"

    extraction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.article_id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False)
    extracted_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="VALID", nullable=False) # VALID, REJECTED, NEEDS_REVIEW
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    article: Mapped["Article"] = relationship("Article", back_populates="extractions")

class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    snapshot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.match_id"), nullable=False)
    as_of_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    feature_version: Mapped[str] = mapped_column(String(20), nullable=False)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    news_coverage: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="snapshots")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="snapshot")

    __table_args__ = (
        UniqueConstraint("match_id", "as_of_at", "feature_version", name="uq_snapshot_match_as_of"),
    )

class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.match_id"), nullable=False)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("feature_snapshots.snapshot_id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String(50), nullable=False)
    p_home: Mapped[float] = mapped_column(Float, nullable=False)
    p_draw: Mapped[float] = mapped_column(Float, nullable=False)
    p_away: Mapped[float] = mapped_column(Float, nullable=False)
    score_distribution: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="predictions")
    snapshot: Mapped["FeatureSnapshot"] = relationship("FeatureSnapshot", back_populates="predictions")
    evidence_links: Mapped[List["EvidenceLink"]] = relationship("EvidenceLink", back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("p_home >= 0.0 AND p_home <= 1.0", name="chk_p_home_range"),
        CheckConstraint("p_draw >= 0.0 AND p_draw <= 1.0", name="chk_p_draw_range"),
        CheckConstraint("p_away >= 0.0 AND p_away <= 1.0", name="chk_p_away_range"),
    )

class EvidenceLink(Base):
    __tablename__ = "evidence_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.prediction_id"), nullable=False)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.article_id"), nullable=False)
    extraction_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("llm_extractions.extraction_id"), nullable=True)
    fact_key: Mapped[str] = mapped_column(String(100), nullable=False)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="evidence_links")

class ModelRun(Base):
    __tablename__ = "model_runs"

    run_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
