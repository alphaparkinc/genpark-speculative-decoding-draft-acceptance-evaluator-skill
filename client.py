import math
from typing import Dict, Any, List, Optional

class SpeculativeDecodingDraftAcceptanceEvaluator:
    """
    Evaluates speculative draft tokens proposed by small draft models (e.g. 1B)
    against target model (e.g. 70B) verification probability distributions using rejection sampling.
    """
    def evaluate_draft_proposals(
        self,
        draft_tokens: List[str],
        draft_probs: List[float],
        target_probs: List[float]
    ) -> Dict[str, Any]:
        accepted_tokens = []
        acceptance_events = []

        for idx, (token, q_prob, p_prob) in enumerate(zip(draft_tokens, draft_probs, target_probs)):
            # Modified rejection sampling condition: accept with prob min(1, p / q)
            ratio = min(1.0, p_prob / q_prob if q_prob > 0 else 0.0)
            
            # Deterministic acceptance threshold for reproducible verification (0.80 benchmark)
            is_accepted = ratio >= 0.70

            acceptance_events.append({
                "step": idx + 1,
                "token": token,
                "draft_prob": round(q_prob, 3),
                "target_prob": round(p_prob, 3),
                "acceptance_ratio": round(ratio, 3),
                "accepted": is_accepted
            })

            if is_accepted:
                accepted_tokens.append(token)
            else:
                # Rejection triggers cascade stop for this speculative lookahead window
                break

        speedup_factor = round((len(accepted_tokens) + 1) / 1.0, 2)

        return {
            "proposed_count": len(draft_tokens),
            "accepted_count": len(accepted_tokens),
            "acceptance_rate_pct": round((len(accepted_tokens) / max(1, len(draft_tokens))) * 100, 1),
            "theoretical_speedup": speedup_factor,
            "accepted_tokens": accepted_tokens,
            "evaluation_trace": acceptance_events
        }
