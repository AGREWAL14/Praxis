"""The five ways of making the rare classes visible, one per run.

Each of these is one change to a run that is otherwise the sequence model exactly
as its parent trained it: same windows, same split, same architecture, same ten
epochs at batch 32, same seed. They divide into two kinds, and the difference
matters when reading a result.

Three of them change what the model learns. Weighting the classes in the loss and
replacing the loss with a focal one both change the size of the gradient a rare
window contributes. Oversampling changes how many times the model sees one.

Two of them change nothing about training and only change how a trained model's
output is turned into an answer. Logit adjustment subtracts the class priors from
the scores. Per-class thresholds divide each score by a number chosen for that
class. A run of either kind trains the same model its parent did and then decides
differently.

Nothing here reads the test partition. The priors and the class weights come from
the training rows, and the thresholds are chosen on validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from src import sequence as sq

# --------------------------------------------------------------------------
# counting the classes
# --------------------------------------------------------------------------


def class_counts(y, n_classes: int) -> np.ndarray:
    """How many rows each class holds, including the classes holding none."""
    return np.bincount(np.asarray(y).astype("int64"), minlength=int(n_classes))


def priors(y, n_classes: int) -> np.ndarray:
    """Each class's share of the rows."""
    counts = class_counts(y, n_classes)
    return counts / max(counts.sum(), 1)


# --------------------------------------------------------------------------
# the three that change what the model learns
# --------------------------------------------------------------------------


def inverse_frequency_weights(y, n_classes: int) -> dict:
    """Weight per class, `N / (K * n_c)`, so the mean weight is 1.

    A class with a thousandth of the rows carries a thousand times the weight per
    row, and the average row carries the same weight it did without any weighting
    at all. Normalising matters: weights that average to something other than 1
    scale every gradient in the run, which would be a second change on top of the
    reweighting.

    A class with no rows would divide by zero. It gets weight 1, which affects
    nothing, because a class with no rows contributes no term to the loss.
    """
    counts = class_counts(y, n_classes)
    total = counts.sum()
    weights = np.ones(int(n_classes), dtype="float64")
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    return {int(code): float(weight) for code, weight in enumerate(weights)}


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Categorical focal cross-entropy, Keras's own implementation.

    Cross-entropy asks every window for the same amount of attention. Focal loss
    multiplies each window's term by `(1 - p)**gamma`, where `p` is the
    probability the model gave the right answer, so a window it already answers
    confidently contributes almost nothing and a window it gets wrong contributes
    nearly all of its term. On a corpus where a handful of classes hold most of
    the rows, most of the confident windows belong to those classes.

    Keras's version is used rather than one written here, because a loss written
    here would have to be registered by name before a saved model could be loaded
    again, and a run whose model cannot be read back is not much of a record.
    """
    import keras

    return keras.losses.CategoricalFocalCrossentropy(gamma=float(gamma), alpha=float(alpha))


def oversample_index(y, n_classes: int, *, target: int | None = None, seed: int = 42):
    """An index into `y` drawing every class up to `target` rows, with replacement.

    Classes at or above `target` keep every row they have and gain nothing.
    Classes below it are drawn from repeatedly until they reach it, so the model
    sees a rare window several times per epoch rather than once.

    `target` defaults to the median class size. The median is not the largest
    class on purpose: drawing every class up to the largest would multiply the
    training set five times over, and drawing the smallest class up to it would
    repeat two dozen windows sixty thousand times, which is a statement about
    those two dozen windows rather than about the class.

    The returned index is sorted, so the rows keep the order the file has and the
    only difference from the training set it came from is which rows appear more
    than once.
    """
    y = np.asarray(y).astype("int64")
    counts = class_counts(y, n_classes)
    present = counts[counts > 0]
    if target is None:
        target = int(np.median(present))
    rng = np.random.default_rng(int(seed))

    keep = [np.arange(len(y))]
    for code in range(int(n_classes)):
        where = np.flatnonzero(y == code)
        short = int(target) - len(where)
        if len(where) and short > 0:
            keep.append(rng.choice(where, size=short, replace=True))
    return np.sort(np.concatenate(keep)), int(target)


def resampling_record(y_before, y_after, n_classes: int, *, target: int, classes) -> dict:
    """What the oversampling did, per class, for the run's metrics."""
    before = class_counts(y_before, n_classes)
    after = class_counts(y_after, n_classes)
    largest = int(after.max())
    smallest_before = int(before[before > 0].min())
    ratio_before = float(before.max()) / max(smallest_before, 1)
    ratio_after = float(largest) / max(int(target), 1)
    return {
        "rule": "every class drawn up to the median class size, with replacement",
        "target_per_class": int(target),
        "n_train_before": int(before.sum()),
        "n_train_after": int(after.sum()),
        "largest_class_windows": largest,
        "imbalance_before": round(ratio_before, 1),
        "imbalance_after": round(ratio_after, 1),
        "not_class_parity": (
            f"Classes at or above the target keep the count they had, so this is not full "
            f"class parity. The largest class holds {largest:,} windows against a target of "
            f"{int(target):,}, which is {ratio_after:,.1f} to 1 after the resampling against "
            f"{ratio_before:,.0f} to 1 before it. The classes above the target are untouched."
        ),
        "per_class": {
            str(label): {
                "before": int(before[code]),
                "after": int(after[code]),
                "times_repeated": round(float(after[code]) / max(int(before[code]), 1), 2),
            }
            for code, label in enumerate(classes)
        },
    }


# --------------------------------------------------------------------------
# the two that change only how a trained model decides
# --------------------------------------------------------------------------


def argmax_decision():
    """The rule every run uses unless it is the thing being changed."""

    def decide(probabilities):
        return np.asarray(probabilities).argmax(axis=1)

    return decide


def logit_adjusted_decision(prior, tau: float = 1.0):
    """Subtract the class priors from the scores before taking the largest.

    A softmax trained on this corpus learns the class sizes along with everything
    else, so a rare class has to clear a bar the common ones do not. Taking
    `log p - tau * log prior` removes that bar: a class is chosen when its score
    is high relative to how often it appears, rather than high outright. At
    `tau` 0 the rule is the ordinary largest-score rule.

    The priors come from the training rows. No validation or test row is read.
    """
    prior = np.asarray(prior, dtype="float64")
    offset = float(tau) * np.log(np.clip(prior, 1e-12, None))

    def decide(probabilities):
        scores = np.log(np.clip(np.asarray(probabilities, dtype="float64"), 1e-12, None))
        return (scores - offset).argmax(axis=1)

    return decide


def threshold_decision(thresholds):
    """Divide each class's score by its own threshold, then take the largest.

    A threshold below 1 makes a class easier to choose and a threshold of 1
    leaves it where it was, so a run with every threshold at 1 decides exactly
    the way the ordinary rule does. Dividing rather than testing each score
    against its threshold in turn means every window still gets exactly one
    answer, with no window left undecided and none claimed by two classes.
    """
    thresholds = np.asarray(thresholds, dtype="float64")

    def decide(probabilities):
        return (np.asarray(probabilities, dtype="float64") / thresholds).argmax(axis=1)

    return decide


def tune_thresholds(probabilities, y_true, *, n_classes: int, grid, labels=None) -> tuple:
    """Choose a threshold per class on the partition it is handed, one pass.

    Every threshold starts at 1, which is the ordinary rule, and the classes are
    visited from the rarest upward. Each class's threshold is set to the grid
    value that gives the best macro-F1 with the other thresholds held where they
    are, and it is only moved if it improves on what the run already has. One
    pass rather than repeated passes, fixed before the run, so the number of
    times the partition is read is not decided by looking at the result.

    Macro-F1 is what is maximised because macro-F1 is what this project reports.

    The caller passes validation probabilities. Nothing here knows what partition
    it has been given, so the caller is the one place that has to be right about
    it.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    y_true = np.asarray(y_true).astype("int64")
    n_classes = int(n_classes)
    labels_range = list(range(n_classes))
    grid = [float(v) for v in grid]

    def macro(thresholds):
        predicted = threshold_decision(thresholds)(probabilities)
        return float(f1_score(y_true, predicted, labels=labels_range, average="macro", zero_division=0))

    thresholds = np.ones(n_classes, dtype="float64")
    started = macro(thresholds)
    best = started
    order = list(np.argsort(class_counts(y_true, n_classes), kind="stable"))

    trace = []
    for code in order:
        scores = []
        for value in grid:
            trial = thresholds.copy()
            trial[code] = value
            scores.append(macro(trial))
        position = int(np.argmax(scores))
        if scores[position] > best:
            thresholds[code] = grid[position]
            best = scores[position]
        trace.append(
            {
                "class": str(labels[code]) if labels is not None else int(code),
                "threshold": float(thresholds[code]),
                "macro_f1": round(float(best), 6),
            }
        )

    record = {
        "rule": "one score divided by one threshold per class, largest wins",
        "tuned_on": "validation",
        "grid": grid,
        "order": "rarest class first, by count in the partition tuned on",
        "thresholds": {
            (str(labels[code]) if labels is not None else int(code)): float(thresholds[code])
            for code in range(n_classes)
        },
        "validation_macro_f1_before": round(started, 6),
        "validation_macro_f1_after": round(best, 6),
        "n_thresholds_moved": int((thresholds != 1.0).sum()),
        "caveats": [
            "One greedy pass in an order fixed before the run, rarest class first. Each "
            "threshold is chosen with the others held where they are, so a different order "
            "could reach different thresholds. These are not a derived optimum.",
            "The thresholds were chosen to raise macro-F1 on validation. That they raise it "
            "on test is not established here. The test partition is scored once, with these "
            "thresholds already fixed.",
        ],
        "trace": trace,
    }
    return thresholds, record


# --------------------------------------------------------------------------
# the comparison blocks a run reports
# --------------------------------------------------------------------------


def comparison_block(metrics, parent, *, run_id, classes, threshold, note, with_mean=False) -> dict:
    """One comparison against the parent run, on one set of classes.

    The per-class figures come from `sequence.comparison_against`, which is what
    every comparison in this project is built with. Two things are done to what
    it returns. Its line about the two runs being scored on different partitions
    is removed, because here they are not: a run and its parent are scored on the
    same windows, and `note` says so. And a group mean is added where the set of
    classes is one whose average is worth reading as a number in its own right.
    """
    block = sq.comparison_against(
        metrics, parent, run_id=run_id, classes=list(classes), threshold=threshold
    )
    block.pop("test_partitions_differ", None)
    block["test_items"] = note
    if with_mean:
        block["mean_f1"] = mean_f1(metrics, parent, classes)
    return block


def mean_f1(metrics, parent, classes) -> dict:
    """Mean per-class F1 across `classes`, on both sides, and the difference."""
    classes = list(classes)
    here = float(np.mean([metrics["per_class_f1"][c] for c in classes]))
    there = float(np.mean([parent["per_class_f1"][c] for c in classes]))
    return {
        "n_classes": len(classes),
        "classes": classes,
        "parent": there,
        "this_run": here,
        "difference": here - there,
    }


# --------------------------------------------------------------------------
# the sixth, which changes what the run optimises rather than what it sees
#
# Nothing in the current programme calls anything below this line. It was written
# for NB07b, which is withdrawn under DECISIONS.md 2026-08-11. It is kept because
# PREREGISTRATION.md Amendments 14, 16 and 17 fix the intervention, the group
# variable, the group counts and the chance rates, and those amendments stand, so
# the direction is specified rather than speculative. tools/dry_run/profiles/nb07b.py
# is what still exercises it.
# --------------------------------------------------------------------------


def group_codes(names) -> tuple:
    """Integer codes per window, and the group names those codes index.

    The names are whatever the caller decides a group is. For NB07b they are
    capture identifiers derived from the recording each window was cut from, so
    the objective's groups are capture sessions and nothing here has to know
    that.
    """
    names = np.asarray(names).astype(str)
    groups, codes = np.unique(names, return_inverse=True)
    return codes.astype("int64"), [str(g) for g in groups]


def group_sizes(codes, n_groups: int) -> np.ndarray:
    """How many windows each group holds, including any holding none."""
    return np.bincount(np.asarray(codes).astype("int64"), minlength=int(n_groups))


class GroupWeights:
    """Weights over the groups, updated by exponentiated gradient and carried on.

    The objective is a weighted mean of the groups' mean losses rather than the
    mean loss over the windows. `q` is what does the weighting, and how it starts
    decides what the run is optimising before any update has happened.

    Three starts, because the same class serves the run and the check that the
    training loop is sound.

    `uniform` gives every group 1/n. This is the standard group-DRO start and it
    is what NB07b uses. Note what it means: a group of 24 windows and a group of
    8,290 carry the same weight from the first batch, so the run departs from its
    parent by group balancing before worst-group weighting has moved anything.

    `proportional` gives each group its share of the training windows. Under this
    weighting the weighted mean of group means is the mean over the windows in
    expectation, so it is the parent's objective written in this notation.

    `batch_share` takes the weights from the batch in front of it rather than
    carrying any state, which makes the objective exactly the mean loss over the
    batch. That is the parent's objective exactly rather than in expectation, and
    it is what the reproduction check runs: any difference from `model.fit` under
    it belongs to the loop and not to the weighting.

    The update is `q_g <- q_g * exp(eta * L_g)` over the groups present in the
    batch, followed by renormalising the whole vector. A group not in the batch
    keeps its unnormalised weight, so nothing about it is inferred from a batch it
    did not appear in; the renormalisation then rescales every group by the same
    constant, which leaves the absent groups' weights unchanged relative to each
    other and moves them relative to the groups that were updated. There is no way
    to renormalise that avoids that, and it is stated here rather than left for a
    reader to work out.

    `eta` of zero freezes `q` where it started, which is how the check runs.
    """

    def __init__(self, sizes, *, eta: float, start: str = "uniform"):
        sizes = np.asarray(sizes, dtype="float64")
        if start not in ("uniform", "proportional", "batch_share"):
            raise ValueError(f"start is {start!r}, not uniform, proportional or batch_share")
        if start == "batch_share" and float(eta) != 0.0:
            raise ValueError("batch_share carries no state, so eta has to be 0")

        self.sizes = sizes
        self.n_groups = int(len(sizes))
        self.eta = float(eta)
        self.start = str(start)

        if start == "proportional":
            self.q = sizes / max(sizes.sum(), 1.0)
        else:
            self.q = np.full(self.n_groups, 1.0 / max(self.n_groups, 1), dtype="float64")

        self.n_updates = np.zeros(self.n_groups, dtype="int64")
        self.trajectory = []

    def batch_weights(self, codes) -> np.ndarray:
        """The weight each window in the batch carries, summing to 1 over the batch.

        A group's weight is divided among its windows, so a group contributes its
        weight whether it brought one window or ten. The weights of the groups
        present are renormalised to sum to 1, so the size of the loss does not
        depend on how many groups happened to turn up.
        """
        codes = np.asarray(codes).astype("int64")
        present, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        share = counts / counts.sum() if self.start == "batch_share" else self.q[present]
        share = share / max(share.sum(), 1e-12)
        return (share[inverse] / counts[inverse]).astype("float64")

    def update(self, codes, losses) -> None:
        """One exponentiated-gradient step from the group losses in this batch."""
        if self.eta == 0.0:
            return
        codes = np.asarray(codes).astype("int64")
        losses = np.asarray(losses, dtype="float64")
        present, inverse = np.unique(codes, return_inverse=True)
        means = np.bincount(inverse, weights=losses) / np.bincount(inverse)
        self.q[present] *= np.exp(self.eta * means)
        self.q /= max(self.q.sum(), 1e-12)
        self.n_updates[present] += 1

    def snapshot(self, *, epoch: int, group_names) -> dict:
        """The weights as they stand, recorded once an epoch."""
        order = np.argsort(-self.q)
        entry = {
            "epoch": int(epoch),
            "max_weight": float(self.q.max()),
            "min_weight": float(self.q.min()),
            "effective_groups": float(1.0 / max((self.q**2).sum(), 1e-12)),
            "heaviest": [
                {"group": str(group_names[i]), "weight": float(self.q[i]),
                 "windows": int(self.sizes[i])}
                for i in order[:5]
            ],
            "weights": {str(group_names[i]): float(self.q[i]) for i in range(self.n_groups)},
        }
        self.trajectory.append(entry)
        return entry

    def record(self, group_names) -> dict:
        """What the weighting did over the run, for the run's metrics."""
        return {
            "rule": "q_g <- q_g * exp(eta * L_g) over the groups in the batch, then renormalised",
            "start": self.start,
            "eta": self.eta,
            "n_groups": self.n_groups,
            "carried_across_batches": True,
            "absent_groups": (
                "keep their unnormalised weight; the renormalisation rescales every group "
                "by the same constant"
            ),
            "no_floor": (
                "no group-size floor, no weight cap and no warmup. The trajectory below is "
                "what shows whether the weighting collapsed onto the smallest groups"
            ),
            "group_windows": {
                str(name): int(n) for name, n in zip(group_names, self.sizes.astype("int64"))
            },
            "batches_each_group_appeared_in": {
                str(name): int(n) for name, n in zip(group_names, self.n_updates)
            },
            "trajectory": self.trajectory,
        }


def worst_group_dro(sizes, *, eta: float, start: str = "uniform") -> GroupWeights:
    """Weights over the groups for a worst-group objective, in one call."""
    return GroupWeights(sizes, eta=float(eta), start=str(start))
