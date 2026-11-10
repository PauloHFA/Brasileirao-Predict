from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EvidenceItem(BaseModel):
    id: str = Field(..., description="Unique evidence identifier, e.g. DOC_42:SPAN_03")
    field: Literal["pressao_crise", "impacto_desfalques", "prioridade_rotacao", "clima_grupo"]
    source_id: str = Field(..., description="Source document or URL ID")
    quote: str = Field(..., min_length=3, description="Verbatim text span supporting the score")
    certainty: Literal["confirmed", "reported", "rumor", "speculation"] = "reported"

class TeamContextFeatures(BaseModel):
    pressao_crise: Optional[int] = Field(None, ge=0, le=10)
    impacto_desfalques: Optional[int] = Field(None, ge=0, le=10)
    prioridade_rotacao: Optional[int] = Field(None, ge=1, le=5)
    clima_grupo: Optional[int] = Field(None, ge=0, le=10)

class ExtractionPayload(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    match_id: str
    team_id: str
    doc_ids: List[str]
    features: TeamContextFeatures
    evidence: List[EvidenceItem] = Field(default_factory=list)
    reporting_confidence_0a1: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_evidence_linkage(self) -> "ExtractionPayload":
        """
        Rule from Appendix A:
        Every non-null feature score MUST have at least one corresponding evidence item for that field.
        """
        evidence_fields = {e.field for e in self.evidence}
        feat_dict = self.features.model_dump()

        for field_name, score in feat_dict.items():
            if score is not None:
                if field_name not in evidence_fields:
                    raise ValueError(
                        f"Validation error: Feature '{field_name}' has non-null score ({score}) "
                        f"but no evidence item linked to this field in evidence list."
                    )
        return self
