from typing import Any, Dict


class PromotionGate:
    """
    Evaluates scientific promotion criteria for contextual model M1 vs baseline B1.
    Rule: Relative Log Loss reduction >= 2% AND no severe calibration degradation (ECE <= ECE_B1 + 0.03).
    """

    def __init__(self, min_rel_improvement: float = 0.02, max_ece_degradation: float = 0.03):
        self.min_rel_improvement = min_rel_improvement
        self.max_ece_degradation = max_ece_degradation

    def evaluate(
        self,
        metrics_b1: Dict[str, float],
        metrics_m1: Dict[str, float],
        bootstrap_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        loss_b1 = metrics_b1["log_loss"]
        loss_m1 = metrics_m1["log_loss"]
        ece_b1 = metrics_b1.get("ece", 0.0)
        ece_m1 = metrics_m1.get("ece", 0.0)

        # Relative improvement: (loss_b1 - loss_m1) / loss_b1
        rel_improvement = (loss_b1 - loss_m1) / loss_b1
        meets_loss_target = rel_improvement >= self.min_rel_improvement
        meets_calibration = (ece_m1 - ece_b1) <= self.max_ece_degradation
        ci_supports = bootstrap_result.get("ci_lower_95", 0.0) >= 0.0

        promoted = meets_loss_target and meets_calibration

        if promoted:
            decision = "PROMOTE_M1_TO_PRODUCTION"
            explanation = (
                f"M1 achieved {rel_improvement:.2%} relative reduction in Log Loss (>= {self.min_rel_improvement:.1%}) "
                f"without degrading calibration. Hypothesis confirmed."
            )
            active_model = "contextual_lgbm"
        else:
            decision = "RETAIN_B1_BASELINE"
            explanation = (
                f"Null hypothesis not rejected: M1 relative change ({rel_improvement:.2%}) "
                f"did not meet the {self.min_rel_improvement:.1%} threshold. "
                f"B1 retained as default production model. Negative result registered successfully."
            )
            active_model = "numeric_lgbm"

        return {
            "decision": decision,
            "active_model": active_model,
            "relative_improvement": round(float(rel_improvement), 4),
            "threshold_required": self.min_rel_improvement,
            "meets_loss_target": meets_loss_target,
            "meets_calibration": meets_calibration,
            "ci_supports": ci_supports,
            "explanation": explanation
        }
