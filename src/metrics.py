"""F_0.5 macro-averaged scoring, matching the competition spec exactly:

- Computed per Source1 entity, then averaged (macro) across all entities.
- Correct empty prediction on a true singleton -> 1.0
- Any predicted match on a true singleton -> 0.0
- Predicting empty when there ARE true matches -> 0.0 (recall = 0)
- Otherwise standard F_beta with beta=0.5 (precision weighted 2x recall).

Use this SAME function for local validation as you use to pick your
threshold, so what you optimize locally is what the leaderboard scores.
"""
from typing import Dict, Set


def parse_id_list(s) -> Set[str]:
    if s is None:
        return set()
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return set()
    return set(x.strip() for x in s.split(",") if x.strip())


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred_ids: Set[str], true_ids: Set[str], beta: float = 0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids:
        return 0.0
    if not true_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f_beta(precision, recall, beta)


def macro_f_beta(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]],
                  beta: float = 0.5) -> float:
    """predictions / ground_truth: source1_entity_id -> set of matched ids.
    Every key in ground_truth is scored; missing from predictions == empty prediction."""
    if not ground_truth:
        return 0.0
    total = 0.0
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        total += score_entity(pred_ids, true_ids, beta)
    return total / len(ground_truth)
