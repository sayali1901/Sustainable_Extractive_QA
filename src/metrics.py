"""Metrics for extractive answers."""

import re
from collections import Counter


def normalize(text):
    """Normalize text for SQuAD-style matching."""
    # Match case and spacing before comparison.
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction, gold_answers):
    """Check for an exact normalized match."""
    # A refused answer cannot match the gold text.
    if prediction is None:
        return False
    pred_norm = normalize(prediction)
    return any(normalize(gold) == pred_norm for gold in gold_answers)


def answer_containment(prediction, gold_answers):
    """Check whether the prediction contains a gold answer."""
    # Compare normalized text to ignore formatting.
    if prediction is None:
        return False
    pred_norm = normalize(prediction)
    return any(normalize(gold) in pred_norm for gold in gold_answers)


def token_f1(prediction, gold_answers):
    """Return the best token F1 across the gold answers."""
    # Score against every accepted answer.
    if prediction is None:
        return 0.0
    pred_tokens = normalize(prediction).split()
    best_f1 = 0.0

    for gold in gold_answers:
        gold_tokens = normalize(gold).split()
        # Counter intersection keeps repeated-token counts.
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        # Harmonic mean balances precision and recall.
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1
