"""McNemar's test on paired predictions, and Holm-Bonferroni across a family of them.

Two models scored on the same items can be compared item by item. What matters is
not how many each got right but the items they disagree on: the ones the first got
right and the second got wrong, and the ones the other way round. If the two models
were equally good those two counts are two halves of the same coin, and McNemar's
test asks how far from an even split the observed counts are.

The test says nothing at all about two models scored on different items. There is no
pairing to compute, so `mcnemar` requires both prediction arrays to be the same
length as the label array and refuses anything else.

Running one test is a decision at the stated alpha. Running eleven and reporting the
smallest is not, because with eleven independent tests at alpha 0.05 the chance of at
least one false positive is over forty percent. `holm_bonferroni` adjusts a family of
p-values so the chance of any false positive among them stays at alpha. It is uniformly
more powerful than dividing alpha by the number of tests, and it assumes nothing about
whether the tests are independent.
"""

from __future__ import annotations

import numpy as np

EXACT_BELOW = 25


# --------------------------------------------------------------------------
# one comparison
# --------------------------------------------------------------------------


def mcnemar(y_true, pred_a, pred_b, *, name_a="a", name_b="b", exact_below: int = EXACT_BELOW) -> dict:
    """McNemar's test on two sets of predictions over the same items.

    `b` counts the items the first model got right and the second got wrong, and
    `c` the items the other way round. Items both models got right and items both
    got wrong carry no information about which is better and are not in the test,
    which is why a large accuracy difference on a small number of discordant items
    can still be inconclusive.

    Below `exact_below` discordant items the two-sided binomial test is used, which
    is exact. Above it the chi-square approximation with Edwards's continuity
    correction is used, which is what the correction is for: without it the
    approximation is anti-conservative on a discrete statistic.
    """
    from scipy import stats

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if pred_a.shape != y_true.shape or pred_b.shape != y_true.shape:
        raise ValueError(
            f"{name_a} has {pred_a.shape} predictions and {name_b} has {pred_b.shape} against "
            f"{y_true.shape} labels. McNemar pairs item with item, so two runs scored on "
            "different items or different numbers of them cannot be compared by it at all."
        )

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c

    if discordant == 0:
        statistic, p_value, method = 0.0, 1.0, "no discordant items"
    elif discordant < int(exact_below):
        statistic = float(min(b, c))
        p_value = float(stats.binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue)
        method = f"exact binomial, {discordant} discordant items"
    else:
        statistic = float((abs(b - c) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, 1))
        method = "chi-square with continuity correction"

    return {
        "a": name_a,
        "b": name_b,
        "n_items": int(len(y_true)),
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "only_a_correct": b,
        "only_b_correct": c,
        "discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
        "favours": name_a if b > c else (name_b if c > b else "neither"),
    }


# --------------------------------------------------------------------------
# a family of comparisons
# --------------------------------------------------------------------------


def holm_bonferroni(p_values, *, alpha: float = 0.05):
    """Holm-adjusted p-values and the reject flags at `alpha`, in the input order.

    The p-values are ranked from smallest to largest, the smallest is multiplied by
    the number of tests, the next by one fewer, and so on. Each adjusted value is
    then raised to at least the one before it, so the adjusted values never decrease
    as the raw ones increase; without that step a test could be rejected while a
    test with a larger p-value was not, which is not what the procedure means.
    """
    p_values = np.asarray(p_values, dtype="float64")
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    adjusted = np.empty(m, dtype="float64")
    running = 0.0
    for rank, index in enumerate(np.argsort(p_values, kind="stable")):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted, adjusted <= float(alpha)


def compare_family(comparisons, *, alpha: float = 0.05, family: str | None = None) -> list:
    """Run Holm-Bonferroni over a list of `mcnemar` results and return them annotated.

    A family is the set of comparisons a claim is read from together. Correcting
    within one is what keeps the claim at `alpha`; which comparisons belong in one
    family is a judgement the caller makes and this function records under
    `family` rather than decides.
    """
    rows = [dict(row) for row in comparisons]
    adjusted, reject = holm_bonferroni([row["p_value"] for row in rows], alpha=alpha)
    for row, value, flag in zip(rows, adjusted, reject):
        row["holm_p"] = float(value)
        row["significant_at_alpha"] = bool(flag)
        row["alpha"] = float(alpha)
        row["family"] = family
        row["family_size"] = len(rows)
    return rows
