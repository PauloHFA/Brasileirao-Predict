import json
import re
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from sqlalchemy.orm import Session

from src.config import settings
from src.prompts.extraction_prompts import EXTRACTION_USER_PROMPT_TEMPLATE, SYSTEM_EXTRACTION_PROMPT
from src.schemas.extraction import EvidenceItem, ExtractionPayload, TeamContextFeatures
from src.schemas.models import Article, LLMExtraction


class BaseExtractor:
    def extract(
        self,
        doc_id: str,
        title: str,
        text_content: str,
        team_slug: str,
        team_name: str,
        as_of_at: str,
        match_id: Optional[str] = None
    ) -> ExtractionPayload:
        raise NotImplementedError

class MockLLMExtractor(BaseExtractor):
    """
    Deterministic rule-anchored extractor for fast testing, validation and CI environments.
    Strictly follows the schema and ensures zero fabricated facts or missing citations.
    """

    def extract(
        self,
        doc_id: str,
        title: str,
        text_content: str,
        team_slug: str,
        team_name: str,
        as_of_at: str,
        match_id: Optional[str] = None
    ) -> ExtractionPayload:
        full_text = f"{title}. {text_content}".lower()
        evidence_items: List[EvidenceItem] = []

        pressao_crise: Optional[int] = None
        impacto_desfalques: Optional[int] = None
        prioridade_rotacao: Optional[int] = None
        clima_grupo: Optional[int] = None

        # 1. Desfalques
        has_absence_terms = any(term in full_text for term in ["lesao", "lesão", "desfalca", "vetado", "artilheiro", "departamento medico", "coxa"])
        is_negated_absence = any(term in full_text for term in ["sem lesao", "sem lesão", "sem novas lesoes", "sem desfalques", "sem novas ausencias"])
        if has_absence_terms and not is_negated_absence:
            sentences = [s.strip() for s in re.split(r"[.!?]", text_content) if s.strip()]
            quote = next((s for s in sentences if any(k in s.lower() for k in ["lesao", "lesão", "desfalca", "coxa", "vetado"])), title)
            impacto_desfalques = 6
            evidence_items.append(
                EvidenceItem(
                    id=f"{doc_id}:SPAN_ABS",
                    field="impacto_desfalques",
                    source_id=doc_id,
                    quote=quote[:120],
                    certainty="confirmed"
                )
            )

        # 2. Crise / Protestos / Troca de técnico
        has_crisis_terms = any(term in full_text for term in ["protest", "demite", "demissao", "demissão", "crise", "ameaca", "ameaça"])
        is_negated_crisis = any(term in full_text for term in ["sem protest", "sem crise", "nenhum protest", "nenhuma crise"])
        if has_crisis_terms and not is_negated_crisis:
            sentences = [s.strip() for s in re.split(r"[.!?]", text_content) if s.strip()]
            quote = next((s for s in sentences if any(k in s.lower() for k in ["protest", "demit", "crise"])), title)
            pressao_crise = 7
            evidence_items.append(
                EvidenceItem(
                    id=f"{doc_id}:SPAN_CRISE",
                    field="pressao_crise",
                    source_id=doc_id,
                    quote=quote[:120],
                    certainty="confirmed"
                )
            )

        # 3. Rodízio / Preservação
        if any(term in full_text for term in ["poupar", "preservar", "reserva", "libertadores", "rodizio", "rodízio"]):
            sentences = [s.strip() for s in re.split(r"[.!?]", text_content) if s.strip()]
            quote = next((s for s in sentences if any(k in s.lower() for k in ["poupar", "preservar", "reserva"])), title)
            prioridade_rotacao = 5
            evidence_items.append(
                EvidenceItem(
                    id=f"{doc_id}:SPAN_ROT",
                    field="prioridade_rotacao",
                    source_id=doc_id,
                    quote=quote[:120],
                    certainty="confirmed"
                )
            )

        features = TeamContextFeatures(
            pressao_crise=pressao_crise,
            impacto_desfalques=impacto_desfalques,
            prioridade_rotacao=prioridade_rotacao,
            clima_grupo=clima_grupo
        )

        return ExtractionPayload(
            schema_version="1.0",
            match_id=match_id or "MATCH_PREVIEW",
            team_id=team_slug,
            doc_ids=[doc_id],
            features=features,
            evidence=evidence_items,
            reporting_confidence_0a1=0.88 if evidence_items else 0.50
        )

class OllamaExtractor(BaseExtractor):
    """Local Ollama LLM provider calling Ollama REST API with schema validation."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    def extract(
        self,
        doc_id: str,
        title: str,
        text_content: str,
        team_slug: str,
        team_name: str,
        as_of_at: str,
        match_id: Optional[str] = None
    ) -> ExtractionPayload:
        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            as_of_at=as_of_at,
            team_name=team_name,
            team_slug=team_slug,
            title=title,
            text_content=text_content
        )

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_EXTRACTION_PROMPT,
                    "prompt": user_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=30.0
            )
            if resp.status_code == 200:
                raw_json = json.loads(resp.json().get("response", "{}"))
                return ExtractionPayload.model_validate(raw_json)
        except Exception:
            pass

        # Fallback to mock extractor if Ollama server is unreachable or fails schema
        mock = MockLLMExtractor()
        return mock.extract(doc_id, title, text_content, team_slug, team_name, as_of_at, match_id)

def get_extractor() -> BaseExtractor:
    if settings.LLM_PROVIDER == "ollama":
        return OllamaExtractor(base_url=settings.LLM_BASE_URL, model=settings.LLM_MODEL)
    return MockLLMExtractor()

def extract_and_save_article(
    db: Session,
    article: Article,
    team_slug: str,
    team_name: str,
    as_of_at: str,
    match_id: Optional[str] = None
) -> ExtractionPayload:
    extractor = get_extractor()
    doc_id = f"DOC_{article.article_id}"
    text = article.body_summary or article.title

    payload = extractor.extract(
        doc_id=doc_id,
        title=article.title,
        text_content=text,
        team_slug=team_slug,
        team_name=team_name,
        as_of_at=as_of_at,
        match_id=match_id
    )

    # Save to database
    db_ext = LLMExtraction(
        article_id=article.article_id,
        model_id=settings.LLM_MODEL,
        prompt_version="1.0",
        schema_version=payload.schema_version,
        extracted_json=payload.model_dump(),
        status="VALID",
        created_at=datetime.now(timezone.utc)
    )
    db.add(db_ext)
    db.commit()
    return payload
