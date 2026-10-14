"""External ARI and arithmetic NMI, including every negative prediction label."""

from collections import Counter
from math import fsum, log
from numbers import Integral


def integer_labels(labels):
    values = list(labels)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("Labels must be integers")
    return [int(v) for v in values]


def _arithmetic_nmi(cells, rows, cols, n):
    # Match arithmetic-mean entropy normalization; log base cancels in the ratio.
    if len(rows) == len(cols) == 1:
        return 1.0
    if len(rows) == 1 or len(cols) == 1:
        return 0.0
    entropy_true = fsum((count / n) * log(n / count) for count in rows.values())
    entropy_predicted = fsum((count / n) * log(n / count) for count in cols.values())
    mutual_information = fsum(
        (count / n) * log((count * n) / (rows[true] * cols[pred]))
        for (true, pred), count in cells.items()
    )
    # Bound only floating-point roundoff at the theoretical endpoints.
    return min(1.0, max(0.0, 2 * mutual_information / (entropy_true + entropy_predicted)))


def evaluate(truth, predicted):
    truth, predicted = integer_labels(truth), integer_labels(predicted)
    if not truth or len(truth) != len(predicted):
        raise ValueError("Nonempty truth and prediction must have equal lengths")
    choose2 = lambda n: n * (n - 1) // 2
    cells = Counter(zip(truth, predicted))
    rows, cols = Counter(truth), Counter(predicted)
    a = sum(map(choose2, rows.values()))
    b = sum(map(choose2, cols.values()))
    c = sum(map(choose2, cells.values()))
    pairs = choose2(len(truth))
    # Integer arithmetic avoids cancellation and handles degenerate partitions.
    denominator = (a + b) * pairs - 2 * a * b
    ari = 1.0 if denominator == 0 else 2 * (c * pairs - a * b) / denominator
    return {
        "ari_all_observations": ari,
        "nmi_all_observations": _arithmetic_nmi(cells, rows, cols, len(truth)),
        "n_observations": len(predicted),
        "n_predicted_clusters": sum(label >= 0 for label in cols),
        "negative_label_counts": {str(k): cols[k] for k in sorted(cols) if k < 0},
    }
