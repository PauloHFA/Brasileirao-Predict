import json
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import BASE_DIR, settings
from src.schemas.models import Base, Team

# Support connect_args for SQLite (e.g. check_same_thread)
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(db_engine=None) -> None:
    target_engine = db_engine or engine
    Base.metadata.create_all(bind=target_engine)

    # Seed canonical teams from alias dictionary if empty
    alias_dict_path = BASE_DIR / "configs" / "teams_alias_dictionary.json"
    if alias_dict_path.exists():
        with open(alias_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with Session(target_engine) as session:
            count = session.query(Team).count()
            if count == 0:
                for slug, item in data.items():
                    team = Team(
                        team_id=item["id"],
                        canonical_name=item["canonical_name"],
                        slug=slug,
                        aliases=item.get("aliases", [])
                    )
                    session.add(team)
                session.commit()

if __name__ == "__main__":
    print(f"Initializing database at: {settings.DATABASE_URL}")
    init_db()
    print("Database tables created and canonical teams seeded successfully.")
