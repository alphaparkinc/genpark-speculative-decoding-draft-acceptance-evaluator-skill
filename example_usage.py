import json
from client import SpeculativeDecodingDraftAcceptanceEvaluator

def main():
    evaluator = SpeculativeDecodingDraftAcceptanceEvaluator()
    draft_tokens = ["The", "quick", "brown", "fox", "jumps"]
    draft_probs = [0.95, 0.88, 0.82, 0.75, 0.40]
    target_probs = [0.94, 0.85, 0.80, 0.72, 0.15]
    
    res = evaluator.evaluate_draft_proposals(draft_tokens, draft_probs, target_probs)
    print("Speculative Decoding Trace:")
    print(json.dumps(res, indent=2))
    assert res["accepted_count"] == 4
    assert res["theoretical_speedup"] >= 4.0
    print("Speculative decoding evaluator verification: PASS")

if __name__ == "__main__":
    main()
