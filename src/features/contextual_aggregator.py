from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np


def aggregate_team_context_features(
    extractions: List[Dict[str, Any]],
    as_of_at: datetime
) -> Dict[str, Any]:
    """
    Aggregates LLM extractions for a team prior to cutoff as_of_at.
    Computes recency-weighted mean of non-null scores and missingness indicators.
    """
    if as_of_at.tzinfo is None:
        as_of_at = as_of_at.replace(tzinfo=timezone.utc)

    if not extractions:
        return {
            "crisis_score": 0.0,
            "absences_score": 0.0,
            "rotation_score": 1.0,
            "missing_crisis_flag": 1.0,
            "missing_absences_flag": 1.0,
            "missing_rotation_flag": 1.0,
            "news_count": 0,
            "evidence_count": 0
        }

    crisis_vals, crisis_weights = [], []
    absences_vals, absences_weights = [], []
    rotation_vals, rotation_weights = [], []
    total_evidences = 0

    for ext in extractions:
        payload = ext.get("extracted_json", {})
        feats = payload.get("features", {})
        evidences = payload.get("evidence", [])
        total_evidences += len(evidences)

        # Recency weight: default 1.0
        weight = 1.0
        pub_date_str = ext.get("published_at")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str)
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                hours_diff = max(0.0, (as_of_at - pub_date).total_seconds() / 3600.0)
                # Exponential decay over 48 hours
                weight = float(np.exp(-hours_diff / 48.0))
            except Exception:
                weight = 1.0

        if feats.get("pressao_crise") is not None:
            crisis_vals.append(float(feats["pressao_crise"]))
            crisis_weights.append(weight)

        if feats.get("impacto_desfalques") is not None:
            absences_vals.append(float(feats["impacto_desfalques"]))
            absences_weights.append(weight)

        if feats.get("prioridade_rotacao") is not None:
            rotation_vals.append(float(feats["prioridade_rotacao"]))
            rotation_weights.append(weight)

    # Compute weighted averages or default + flag
    if crisis_vals:
        crisis_score = float(np.average(crisis_vals, weights=crisis_weights))
        missing_crisis = 0.0
    else:
        crisis_score = 0.0
        missing_crisis = 1.0

    if absences_vals:
        absences_score = float(np.average(absences_vals, weights=absences_weights))
        missing_absences = 0.0
    else:
        absences_score = 0.0
        missing_absences = 1.0

    if rotation_vals:
        rotation_score = float(np.average(rotation_vals, weights=rotation_weights))
        missing_rotation = 0.0
    else:
        rotation_score = 1.0
        missing_rotation = 1.0

    return {
        "crisis_score": round(crisis_score, 2),
        "absences_score": round(absences_score, 2),
        "rotation_score": round(rotation_score, 2),
        "missing_crisis_flag": missing_crisis,
        "missing_absences_flag": missing_absences,
        "missing_rotation_flag": missing_rotation,
        "news_count": len(extractions),
        "evidence_count": total_evidences
    }

def merge_match_context_features(
    numerical_features: Dict[str, Any],
    home_context: Dict[str, Any],
    away_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Combines numerical baseline features with aggregated home and away contextual features."""
    merged = dict(numerical_features)
    merged.update({
        "crisis_home": home_context["crisis_score"],
        "crisis_away": away_context["crisis_score"],
        "absences_home": home_context["absences_score"],
        "absences_away": away_context["absences_score"],
        "rotation_home": home_context["rotation_score"],
        "rotation_away": away_context["rotation_score"],
        "missing_crisis_home": home_context["missing_crisis_flag"],
        "missing_crisis_away": away_context["missing_crisis_flag"],
        "missing_absences_home": home_context["missing_absences_flag"],
        "missing_absences_away": away_context["missing_absences_flag"],
        "news_coverage_count": home_context["news_count"] + away_context["news_count"],
        "evidence_count": home_context["evidence_count"] + away_context["evidence_count"]
    })
    return merged
