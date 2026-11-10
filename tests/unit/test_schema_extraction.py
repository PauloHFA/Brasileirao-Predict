import pytest
from pydantic import ValidationError

from src.schemas.extraction import EvidenceItem, ExtractionPayload, TeamContextFeatures


def test_valid_extraction_payload_with_nulls():
    payload = ExtractionPayload(
        schema_version="1.0",
        match_id="BR-2024-101",
        team_id="flamengo",
        doc_ids=["DOC_1"],
        features=TeamContextFeatures(
            pressao_crise=None,
            impacto_desfalques=None,
            prioridade_rotacao=None
        ),
        evidence=[],
        reporting_confidence_0a1=0.85
    )
    assert payload.features.pressao_crise is None
    assert payload.reporting_confidence_0a1 == 0.85

def test_valid_extraction_payload_with_evidence():
    payload = ExtractionPayload(
        schema_version="1.0",
        match_id="BR-2024-101",
        team_id="flamengo",
        doc_ids=["DOC_1"],
        features=TeamContextFeatures(
            pressao_crise=6,
            impacto_desfalques=None,
            prioridade_rotacao=None
        ),
        evidence=[
            EvidenceItem(
                id="DOC_1:SPAN_01",
                field="pressao_crise",
                source_id="DOC_1",
                quote="Torcida protesta no CT pedindo demissao do tecnico",
                certainty="confirmed"
            )
        ],
        reporting_confidence_0a1=0.9
    )
    assert payload.features.pressao_crise == 6

def test_reject_non_null_score_without_evidence():
    with pytest.raises(ValidationError) as exc_info:
        ExtractionPayload(
            schema_version="1.0",
            match_id="BR-2024-101",
            team_id="flamengo",
            doc_ids=["DOC_1"],
            features=TeamContextFeatures(
                pressao_crise=7, # Non-null but no evidence!
                impacto_desfalques=None,
                prioridade_rotacao=None
            ),
            evidence=[], # Empty evidence list!
            reporting_confidence_0a1=0.9
        )
    assert "no evidence item linked to this field" in str(exc_info.value)
