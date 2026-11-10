import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import BASE_DIR


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip accents, remove extra punctuation and spaces."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class TeamResolver:
    """Resolves arbitrary team names/aliases to canonical Team objects without collisions."""

    def __init__(self, dict_path: Optional[Path] = None):
        self.dict_path = dict_path or (BASE_DIR / "configs" / "teams_alias_dictionary.json")
        self.canonical_teams: Dict[str, Dict[str, Any]] = {}
        self.alias_to_slug: Dict[str, str] = {}
        self.id_to_slug: Dict[int, str] = {}
        self.load_dictionary()

    def load_dictionary(self) -> None:
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Team alias dictionary not found at {self.dict_path}")

        with open(self.dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.canonical_teams = data
        self.alias_to_slug.clear()
        self.id_to_slug.clear()

        for slug, info in data.items():
            self.id_to_slug[info["id"]] = slug
            # Map canonical name normalized
            norm_canonical = normalize_text(info["canonical_name"])
            self.alias_to_slug[norm_canonical] = slug

            # Map slug
            self.alias_to_slug[normalize_text(slug)] = slug

            # Map all aliases
            for alias in info.get("aliases", []):
                norm_alias = normalize_text(alias)
                self.alias_to_slug[norm_alias] = slug

    def resolve(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """Resolve a name or alias to canonical team dict. Returns None if unresolvable."""
        if not name_or_alias:
            return None

        norm = normalize_text(name_or_alias)
        slug = self.alias_to_slug.get(norm)
        if slug and slug in self.canonical_teams:
            result = self.canonical_teams[slug].copy()
            result["slug"] = slug
            return result

        # Fallback substring matching carefully preventing cross-state collisions
        # Specific guards against dangerous substrings:
        if "botafogo-sp" in norm or "botafogo sp" in norm:
            slug = "botafogo_sp"
        elif "botafogo" in norm:
            slug = "botafogo_rj"
        elif "america-rn" in norm or "america de natal" in norm:
            slug = "america_rn"
        elif "america-mg" in norm or "america mineiro" in norm:
            slug = "america_mg"
        elif "atletico-go" in norm or "atletico goianiense" in norm:
            slug = "atletico_go"
        elif "athletico" in norm or "atletico-pr" in norm or "atletico paranaense" in norm:
            slug = "athletico_pr"
        elif "atletico-mg" in norm or "atletico mineiro" in norm or "galo" in norm:
            slug = "atletico_mg"

        if slug and slug in self.canonical_teams:
            result = self.canonical_teams[slug].copy()
            result["slug"] = slug
            return result

        return None

    def get_by_id(self, team_id: int) -> Optional[Dict[str, Any]]:
        slug = self.id_to_slug.get(team_id)
        if slug and slug in self.canonical_teams:
            result = self.canonical_teams[slug].copy()
            result["slug"] = slug
            return result
        return None

resolver = TeamResolver()
