"""Numbers the exploratory pass needs, computed without holding the data.

Eight and three quarter million rows do not fit anywhere convenient, so every
quantity here is built by reading one file at a time and adding to an
accumulator. Two structures do all the work.

`Moments` carries a running count, sum, sum of squares and sum of cubes per
feature, which is enough for the mean, the standard deviation and the skew, plus
the minimum, the maximum and how many values were exactly zero.

`BinnedCounts` carries, for every class and every feature, how many rows fell in
each bin of a fixed set of bin edges. That one table answers three different
questions later: how much a feature tells you about the label, what the median
is, and how well a single feature separates any two classes. All of those are
rank statistics, and binning preserves rank, so none of them needs the raw
values back.

The bin edges are quantiles taken from a sample. The sample decides only where
the boundaries fall. The counts inside them are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# streaming moments
# --------------------------------------------------------------------------


class Moments:
    """Running mean, standard deviation, skew, range and zero count per feature.

    The central moments are accumulated around a fixed offset rather than
    around zero. Some of these columns run to millions and cubing a raw value
    of that size loses most of the precision in the sum. Subtracting a rough
    guess at the mean first keeps the numbers small, and because the offset is
    constant the exact central moments come back out afterwards.
    """

    def __init__(self, features, offsets=None):
        self.features = list(features)
        n = len(self.features)
        if offsets is None:
            self.offset = np.zeros(n, dtype=np.float64)
        else:
            self.offset = np.asarray(
                [float(offsets[f]) for f in self.features], dtype=np.float64
            )
        self.n = np.zeros(n, dtype=np.int64)
        self.s1 = np.zeros(n, dtype=np.float64)
        self.s2 = np.zeros(n, dtype=np.float64)
        self.s3 = np.zeros(n, dtype=np.float64)
        self.lo = np.full(n, np.inf, dtype=np.float64)
        self.hi = np.full(n, -np.inf, dtype=np.float64)
        self.zeros = np.zeros(n, dtype=np.int64)
        self.nonfinite = np.zeros(n, dtype=np.int64)

    def update(self, values: np.ndarray) -> "Moments":
        """Add a block of rows, shaped (rows, features) in the feature order."""
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f"expected (rows, {len(self.features)}), got {x.shape}")
        if x.shape[0] == 0:
            return self

        good = np.isfinite(x)
        self.nonfinite += (~good).sum(axis=0)
        x = np.where(good, x, 0.0)

        d = np.where(good, x - self.offset, 0.0)
        self.n += good.sum(axis=0)
        self.s1 += d.sum(axis=0)
        self.s2 += (d * d).sum(axis=0)
        self.s3 += (d * d * d).sum(axis=0)
        self.zeros += ((x == 0.0) & good).sum(axis=0)
        self.lo = np.minimum(self.lo, np.where(good, x, np.inf).min(axis=0))
        self.hi = np.maximum(self.hi, np.where(good, x, -np.inf).max(axis=0))
        return self

    # -- readings ---------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.offset + np.where(self.n > 0, self.s1 / np.maximum(self.n, 1), np.nan)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self._m2, 0.0))

    @property
    def skew(self) -> np.ndarray:
        sd = self.std
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, self._m3 / sd**3, 0.0)

    @property
    def zero_share(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n > 0, self.zeros / np.maximum(self.n, 1), np.nan)

    @property
    def _m1(self) -> np.ndarray:
        return np.where(self.n > 0, self.s1 / np.maximum(self.n, 1), np.nan)

    @property
    def _m2(self) -> np.ndarray:
        n = np.maximum(self.n, 1)
        return self.s2 / n - self._m1**2

    @property
    def _m3(self) -> np.ndarray:
        n = np.maximum(self.n, 1)
        m1 = self._m1
        return self.s3 / n - 3.0 * m1 * (self.s2 / n) + 2.0 * m1**3

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.features,
                "rows": self.n,
                "mean": self.mean,
                "std": self.std,
                "min": np.where(np.isfinite(self.lo), self.lo, np.nan),
                "max": np.where(np.isfinite(self.hi), self.hi, np.nan),
                "skew": self.skew,
                "zero_share": self.zero_share,
                "nonfinite": self.nonfinite,
            }
        )


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------


def quantile_edges(values, n_bins: int) -> np.ndarray:
    """Interior bin boundaries for one feature, at evenly spaced quantiles.

    Quantiles rather than an even grid, so the bins land where the values are.
    Duplicate boundaries are collapsed, which means a column that is zero in
    most rows ends up with few bins rather than with a hundred empty ones, and
    a column taking a handful of distinct values ends up with a boundary at
    each of them and no approximation at all.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    return np.unique(np.quantile(x, qs)).astype(np.float64)


class BinnedCounts:
    """How many rows of each class fell in each bin of each feature.

    Bin `i` of a feature holds the values `v` with `edges[i-1] < v <= edges[i]`,
    reading the ends as open, so there is one more bin than there are edges.
    Features with fewer edges leave the tail of their row unused.
    """

    def __init__(self, classes, features, edges: dict):
        self.classes = list(classes)
        self.features = list(features)
        self._edges = [np.asarray(edges[f], dtype=np.float64) for f in self.features]
        self.n_bins = np.array([len(e) + 1 for e in self._edges], dtype=np.int64)
        self.width = int(self.n_bins.max()) if len(self.n_bins) else 1
        self.counts = np.zeros((len(self.classes), len(self.features), self.width), dtype=np.int64)
        self._class_index = {c: i for i, c in enumerate(self.classes)}

    def add(self, label: str, values: np.ndarray) -> "BinnedCounts":
        """Add a block of rows, all of one class, shaped (rows, features)."""
        if label not in self._class_index:
            raise KeyError(f"unknown class {label!r}")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f"expected (rows, {len(self.features)}), got {x.shape}")
        if x.shape[0] == 0:
            return self
        ci = self._class_index[label]
        for fi, edges in enumerate(self._edges):
            idx = np.searchsorted(edges, x[:, fi], side="right")
            self.counts[ci, fi] += np.bincount(idx, minlength=self.width)[: self.width]
        return self

    @property
    def rows_per_class(self) -> np.ndarray:
        """Rows seen per class. Every feature sees the same rows."""
        if not len(self.features):
            return np.zeros(len(self.classes), dtype=np.int64)
        return self.counts[:, 0, :].sum(axis=1)

    def edges_of(self, feature: str) -> np.ndarray:
        return self._edges[self.features.index(feature)]

    def subset(self, classes) -> np.ndarray:
        """The count table restricted to some classes, in the order given."""
        return self.counts[[self._class_index[c] for c in classes]]

    def quantile(self, feature: str, q: float = 0.5, classes=None) -> float:
        """A quantile read off the bins, so accurate to the width of one bin."""
        fi = self.features.index(feature)
        counts = (self.counts if classes is None else self.subset(classes))[:, fi, :].sum(axis=0)
        return quantile_from_counts(counts, self._edges[fi], q)


def quantile_from_counts(counts, edges, q: float = 0.5) -> float:
    """Read a quantile out of one binned row, returning the bin's upper edge."""
    counts = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or edges.size == 0:
        return float("nan")
    i = int(np.searchsorted(np.cumsum(counts), q * total, side="left"))
    return float(edges[min(i, edges.size - 1)])


# --------------------------------------------------------------------------
# what a feature says about the label
# --------------------------------------------------------------------------


def mutual_information(counts) -> np.ndarray:
    """Mutual information in nats between each binned feature and the class.

    Both variables are discrete once the feature is binned, so this is the
    definition summed over the table rather than an estimate: how much shorter
    the description of the class gets once the bin is known. Zero means the bin
    a row lands in says nothing about which class it belongs to. The ceiling is
    the entropy of the class variable, `log(k)` for `k` equally sized classes
    and lower when they are not.
    """
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum(axis=(0, 2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(total > 0, c / np.where(total > 0, total, 1.0), 0.0)
        p_class = p.sum(axis=2, keepdims=True)
        p_bin = p.sum(axis=0, keepdims=True)
        denom = p_class * p_bin
        term = np.where((p > 0) & (denom > 0), p * np.log(np.where(denom > 0, p / denom, 1.0)), 0.0)
    return term.sum(axis=(0, 2))


def label_entropy(counts) -> float:
    """The entropy of the class variable in nats, which is the ceiling above."""
    c = np.asarray(counts, dtype=np.float64)
    if not c.size:
        return float("nan")
    n = c[:, 0, :].sum(axis=1)
    p = n / max(n.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def pair_auc(counts_a, counts_b) -> np.ndarray:
    """Best-direction AUC of every feature for one pair of classes.

    The area under the ROC curve is the chance that a row drawn from one class
    scores above a row drawn from the other, counting a tie as half. Written
    that way it comes straight out of the two binned rows without any sorting:
    for each bin, how many of the other class fell below it, plus half of those
    that fell in the same bin. Rows sharing a bin count as tied, so a feature
    with coarse bins is scored slightly low rather than slightly high.

    The larger of the two directions is returned, because a feature that runs
    the other way separates the pair just as well.
    """
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    below = np.cumsum(b, axis=-1) - b
    na, nb = a.sum(axis=-1), b.sum(axis=-1)
    pairs = na * nb
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((a * below).sum(axis=-1) + 0.5 * (a * b).sum(axis=-1)) / np.where(pairs > 0, pairs, 1.0)
    auc = np.where(pairs > 0, auc, np.nan)
    return np.maximum(auc, 1.0 - auc)


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------


def high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Every unordered pair of features correlating above the threshold."""
    values = corr.to_numpy()
    names = list(corr.columns)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = values[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"feature_a": names[i], "feature_b": names[j], "r": float(r)})
    frame = pd.DataFrame(rows, columns=["feature_a", "feature_b", "r"])
    if len(frame):
        frame = frame.reindex(frame["r"].abs().sort_values(ascending=False).index)
    return frame.reset_index(drop=True)


def cluster_order(corr: pd.DataFrame) -> list:
    """Order features so that the ones saying the same thing sit together.

    Distance is one minus the absolute correlation, so a pair carrying the same
    information ends up adjacent whichever sign it carries it with. Falls back
    to the order given if SciPy is not available or the matrix has holes in it.
    """
    names = list(corr.columns)
    values = np.nan_to_num(corr.to_numpy(dtype=np.float64), nan=0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return names
    distance = 1.0 - np.abs(values)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    if not np.isfinite(distance).all():
        return names
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    return [names[i] for i in order]


# --------------------------------------------------------------------------
# separability between two groups of rows held in memory
# --------------------------------------------------------------------------


def auc_by_column(rows_a, rows_b) -> np.ndarray:
    """Best-direction AUC of every column, for two groups of rows.

    `pair_auc` above answers the same question from binned counts, because the
    exploratory pass never holds the data. This one is for the case where both
    groups do fit in memory and the exact value is wanted: ranks are taken over
    the two groups together, ties share the average rank, and the area follows
    from the rank sum of the first group.

    The larger of the two directions is returned, matching `pair_auc`, so 0.5
    means the column does not separate the two groups at all and 1.0 means it
    separates them completely.
    """
    a = np.asarray(rows_a, dtype=np.float64)
    b = np.asarray(rows_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected two 2-D arrays of equal width, got {a.shape} and {b.shape}")
    n_a, n_b = len(a), len(b)
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    if n_a == 0 or n_b == 0:
        return out
    for j in range(a.shape[1]):
        both = np.concatenate([a[:, j], b[:, j]])
        if not np.isfinite(both).all():
            continue
        _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
        # Average rank of each distinct value: how many values sit strictly below
        # it, plus the middle of the block of ties it belongs to.
        below = np.cumsum(counts) - counts
        average_rank = below + (counts + 1) / 2.0
        ranks = average_rank[inverse]
        auc = (ranks[:n_a].sum() - n_a * (n_a + 1) / 2.0) / (n_a * n_b)
        out[j] = max(auc, 1.0 - auc)
    return out


def smallest_consistent_denominator(values, candidates=(10, 100), tolerance=1e-3):
    """The smallest N for which every value is a multiple of one over N.

    A column that records whether each packet carried some property, averaged over
    a fixed number of packets, can only take values that are multiples of one over
    that number. Reading the values back therefore says how many packets went into
    a row, without anything having to be assumed about how the file was built.

    Returns None when no candidate fits, which is the honest answer for a column
    that is not an average of a per-packet indicator.
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    for n in sorted(candidates):
        scaled = finite * n
        if np.abs(scaled - np.round(scaled)).max() <= tolerance:
            return int(n)
    return None
