"""
Evaluation metrics for extractive QA.

Metrics:
- Exact Match (EM): Strict string match after normalisation.
- Answer Containment: Whether extracted sentence contains the gold span.
- Token F1: Token-level precision/recall harmonic mean.
"""

import re
from collections import Counter


def normalize(text):
    """
    Normalise text for comparison: lowercase, strip articles and punctuation.
    Follows the SQuAD evaluation script conventions.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction, gold_answers):
    """
    Check if prediction exactly matches any gold answer after normalisation.

    Note: For sentence-level extraction, EM will be near zero because
    the system returns full sentences, not minimal answer spans.
    """
    if prediction is None:
        return False
    pred_norm = normalize(prediction)
    return any(normalize(gold) == pred_norm for gold in gold_answers)


def answer_containment(prediction, gold_answers):
    """
    Check if any gold answer span appears within the extracted sentence.
    This is the primary utility metric for sentence-level extractive QA.
    """
    if prediction is None:
        return False
    pred_norm = normalize(prediction)
    return any(normalize(gold) in pred_norm for gold in gold_answers)


def token_f1(prediction, gold_answers):
    """
    Compute maximum token-level F1 across all gold answers.

    For extractive systems returning full sentences, Token F1 is expected
    to be lower than for span-extraction systems because extracted text
    contains many tokens beyond the gold answer.
    """
    if prediction is None:
        return 0.0
    pred_tokens = normalize(prediction).split()
    best_f1 = 0.0

    for gold in gold_answers:
        gold_tokens = normalize(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1
