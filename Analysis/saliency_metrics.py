from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score


ArrayLike = np.ndarray | Sequence[float]


@dataclass
class SaliencyMetrics:
    """Shared implementation of gaze-saliency alignment metrics.

    The class assumes a human gaze saliency map G and a model attention map S on
    the same grid. Distributional metrics use normalized maps P=norm(G) and
    Q=norm(S). Support-based metrics use a binary high-gaze support F derived
    from G.
    """

    eps: float = 1e-12
    fixation_mode: str = "nonzero"
    fixation_topk: int = 20
    fixation_percentile: float = 90.0
    auc_neg_samples: Optional[int] = None
    sauc_neg_samples: Optional[int] = None
    emd_mode: str = "exact"
    random_state: Optional[int] = None

    metric_names: Tuple[str, ...] = ("AUC", "sAUC", "NSS", "CC", "EMD", "SIM", "KL", "IG")
    lower_is_better: Tuple[str, ...] = ("EMD", "KL")

    def __post_init__(self) -> None:
        self.fixation_mode = str(self.fixation_mode).lower().strip()
        self.emd_mode = str(self.emd_mode).lower().strip()
        self.rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random.RandomState()

    @staticmethod
    def as_2d(x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D map after squeeze, got shape {arr.shape}.")
        return arr

    def probability_map(self, x: ArrayLike) -> np.ndarray:
        arr = self.as_2d(x)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= self.eps:
            return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float64)
        return arr / total

    def zscore(self, x: ArrayLike) -> np.ndarray:
        arr = self.as_2d(x)
        sigma = float(np.std(arr))
        if sigma <= self.eps:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - float(np.mean(arr))) / (sigma + self.eps)

    def minmax01(self, x: ArrayLike) -> np.ndarray:
        arr = self.as_2d(x)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if (hi - lo) <= self.eps:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo + self.eps)

    def fixation_mask(self, gaze_map: ArrayLike) -> np.ndarray:
        g = self.as_2d(gaze_map)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = g.shape
        flat = g.reshape(-1)
        mode = self.fixation_mode

        if mode == "nonzero":
            mask = (g > 0).reshape(-1)
        elif mode == "topk":
            k = max(1, min(int(self.fixation_topk), flat.size))
            idx = np.argpartition(flat, -k)[-k:]
            mask = np.zeros(flat.size, dtype=bool)
            mask[idx] = True
        elif mode == "percentile":
            thr = np.percentile(flat, float(self.fixation_percentile))
            mask = flat >= thr
        else:
            raise ValueError("fixation_mode must be one of: nonzero, topk, percentile.")

        # Keep support-based metrics defined when maps are empty or constant.
        if int(mask.sum()) == 0:
            mask[int(np.argmax(flat))] = True
        if int(mask.sum()) == int(mask.size) and int(mask.size) > 1:
            keep = int(np.argmax(flat))
            mask[:] = False
            mask[keep] = True
        return mask.reshape(h, w).astype(np.uint8)

    @staticmethod
    def fixation_pool_from_gaze_maps(gaze_maps: Iterable[ArrayLike], engine: "SaliencyMetrics") -> np.ndarray:
        coords = []
        for gaze_map in gaze_maps:
            xy = np.argwhere(engine.fixation_mask(gaze_map) > 0)
            if xy.size:
                coords.append(xy)
        return np.concatenate(coords, axis=0).astype(np.int64) if coords else np.zeros((0, 2), dtype=np.int64)

    def average_baseline(self, gaze_maps: Iterable[ArrayLike], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        maps = [self.probability_map(m) for m in gaze_maps]
        if maps:
            return self.probability_map(np.mean(maps, axis=0))
        if shape is None:
            raise ValueError("shape is required when no gaze maps are available for the baseline.")
        return np.full(shape, 1.0 / float(np.prod(shape)), dtype=np.float64)

    def auc(self, gaze_map: ArrayLike, model_map: ArrayLike, fixation_mask: Optional[np.ndarray] = None) -> float:
        f = self.fixation_mask(gaze_map) if fixation_mask is None else np.asarray(fixation_mask, dtype=np.uint8)
        y_true = f.reshape(-1).astype(np.uint8)
        if int(y_true.sum()) == 0 or int(y_true.sum()) == int(y_true.size):
            return float("nan")

        scores = self.as_2d(model_map).reshape(-1).astype(np.float64)
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        if self.auc_neg_samples is not None and int(self.auc_neg_samples) < int(neg_idx.size):
            neg_idx = self.rng.choice(neg_idx, size=int(self.auc_neg_samples), replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        return float(roc_auc_score(y_true[idx], scores[idx]))

    def sauc(
        self,
        gaze_map: ArrayLike,
        model_map: ArrayLike,
        shuffled_fixation_xy: Optional[np.ndarray],
        fixation_mask: Optional[np.ndarray] = None,
        neg_samples: Optional[int] = None,
    ) -> float:
        f = self.fixation_mask(gaze_map) if fixation_mask is None else np.asarray(fixation_mask, dtype=np.uint8)
        pos_xy = np.argwhere(f > 0)
        if pos_xy.shape[0] == 0 or shuffled_fixation_xy is None or len(shuffled_fixation_xy) == 0:
            return float("nan")

        model = self.as_2d(model_map)
        h, w = model.shape
        pool = np.asarray(shuffled_fixation_xy, dtype=np.int64)
        n_neg = min(int(neg_samples or self.sauc_neg_samples or pool.shape[0]), int(pool.shape[0]))
        neg_xy = pool[self.rng.choice(pool.shape[0], size=n_neg, replace=False)]
        neg_y = np.clip(neg_xy[:, 0], 0, h - 1)
        neg_x = np.clip(neg_xy[:, 1], 0, w - 1)

        pos_scores = model[pos_xy[:, 0], pos_xy[:, 1]]
        neg_scores = model[neg_y, neg_x]
        y_true = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)]).astype(np.uint8)
        scores = np.concatenate([pos_scores, neg_scores]).astype(np.float64)
        if int(y_true.sum()) == 0 or int(y_true.sum()) == int(y_true.size):
            return float("nan")
        return float(roc_auc_score(y_true, scores))

    def nss(self, gaze_map: ArrayLike, model_map: ArrayLike, fixation_mask: Optional[np.ndarray] = None) -> float:
        f = self.fixation_mask(gaze_map) if fixation_mask is None else np.asarray(fixation_mask, dtype=np.uint8)
        values = self.zscore(model_map)[f > 0]
        return float(np.mean(values)) if values.size else float("nan")

    def cc(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        g = self.as_2d(gaze_map).reshape(-1)
        s = self.as_2d(model_map).reshape(-1)
        g = g - float(np.mean(g))
        s = s - float(np.mean(s))
        denom = float(np.linalg.norm(g) * np.linalg.norm(s))
        if denom <= self.eps:
            return float("nan")
        return float(np.dot(g, s) / (denom + self.eps))

    def sim(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        p = self.probability_map(gaze_map)
        q = self.probability_map(model_map)
        return float(np.minimum(p, q).sum())

    def kl(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        p = np.clip(self.probability_map(gaze_map).reshape(-1), self.eps, None)
        q = np.clip(self.probability_map(model_map).reshape(-1), self.eps, None)
        return float(np.sum(p * np.log(p / q)))

    def ig(self, gaze_map: ArrayLike, model_map: ArrayLike, baseline: Optional[ArrayLike] = None, fixation_mask: Optional[np.ndarray] = None) -> float:
        f = self.fixation_mask(gaze_map) if fixation_mask is None else np.asarray(fixation_mask, dtype=np.uint8)
        if int(f.sum()) == 0:
            return float("nan")
        q = self.probability_map(model_map)
        b = self.probability_map(baseline) if baseline is not None else np.full(q.shape, 1.0 / float(q.size), dtype=np.float64)
        return float(np.mean(np.log2((q[f > 0] + self.eps) / (b[f > 0] + self.eps))))

    def emd(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        if self.emd_mode == "marginal":
            return self.emd_marginal(gaze_map, model_map)
        if self.emd_mode != "exact":
            raise ValueError("emd_mode must be 'exact' or 'marginal'.")
        try:
            return self.emd_exact(gaze_map, model_map)
        except Exception:
            return self.emd_marginal(gaze_map, model_map)

    def emd_marginal(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        p = self.probability_map(gaze_map)
        q = self.probability_map(model_map)
        h, w = p.shape
        ys = np.arange(h, dtype=np.float64)
        xs = np.arange(w, dtype=np.float64)
        emd_y = wasserstein_distance(ys, ys, u_weights=p.sum(axis=1), v_weights=q.sum(axis=1))
        emd_x = wasserstein_distance(xs, xs, u_weights=p.sum(axis=0), v_weights=q.sum(axis=0))
        return float(emd_x + emd_y)

    def emd_exact(self, gaze_map: ArrayLike, model_map: ArrayLike) -> float:
        p = self.probability_map(gaze_map).reshape(-1)
        q = self.probability_map(model_map).reshape(-1)
        shape = self.as_2d(gaze_map).shape
        cost, a_eq = _transport_problem_matrices(shape)
        b_eq = np.concatenate([q, p])
        result = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        return float(result.fun)

    def compute_all(
        self,
        gaze_map: ArrayLike,
        model_map: ArrayLike,
        fixation_mask: Optional[np.ndarray] = None,
        shuffled_fixation_xy: Optional[np.ndarray] = None,
        baseline: Optional[ArrayLike] = None,
        include_sauc: bool = True,
    ) -> Dict[str, float]:
        f = self.fixation_mask(gaze_map) if fixation_mask is None else np.asarray(fixation_mask, dtype=np.uint8)
        out = {
            "AUC": self.auc(gaze_map, model_map, f),
            "NSS": self.nss(gaze_map, model_map, f),
            "CC": self.cc(gaze_map, model_map),
            "EMD": self.emd(gaze_map, model_map),
            "SIM": self.sim(gaze_map, model_map),
            "KL": self.kl(gaze_map, model_map),
            "IG": self.ig(gaze_map, model_map, baseline=baseline, fixation_mask=f),
        }
        out["sAUC"] = self.sauc(gaze_map, model_map, shuffled_fixation_xy, f) if include_sauc else float("nan")
        return out

    @staticmethod
    def normalize_distribution_dict(d: Mapping, ignore: Optional[Iterable] = None) -> Dict[str, float]:
        ignore_set = {str(x).strip().lower() for x in (ignore or set())}
        out: Dict[str, float] = {}
        if not isinstance(d, Mapping):
            return out
        for key, value in d.items():
            k = str(key).strip().lower() if key is not None else ""
            if k in ignore_set:
                continue
            v = float(value)
            if np.isfinite(v) and v > 0:
                out[k] = out.get(k, 0.0) + v
        total = float(sum(out.values()))
        return {k: v / total for k, v in out.items()} if total > 0 else {}

    @staticmethod
    def align_distribution_dicts(
        reference: Mapping,
        prediction: Mapping,
        ignore: Optional[Iterable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        p_dict = SaliencyMetrics.normalize_distribution_dict(reference, ignore=ignore)
        q_dict = SaliencyMetrics.normalize_distribution_dict(prediction, ignore=ignore)
        keys = sorted(set(p_dict) | set(q_dict))
        p = np.array([p_dict.get(k, 0.0) for k in keys], dtype=np.float64)
        q = np.array([q_dict.get(k, 0.0) for k in keys], dtype=np.float64)
        return p, q, keys

    def distribution_metrics(
        self,
        reference: Mapping,
        prediction: Mapping,
        ignore: Optional[Iterable] = None,
    ) -> Dict[str, float]:
        p, q, _ = self.align_distribution_dicts(reference, prediction, ignore=ignore)
        if p.size == 0 or q.size == 0:
            return {"KL": float("nan"), "CC": float("nan"), "SIM": float("nan")}

        p = np.clip(p, 0.0, None)
        q = np.clip(q, 0.0, None)
        p = p / max(float(p.sum()), self.eps)
        q = q / max(float(q.sum()), self.eps)

        pc = p - float(p.mean())
        qc = q - float(q.mean())
        denom = float(np.linalg.norm(pc) * np.linalg.norm(qc))
        cc = float(np.dot(pc, qc) / (denom + self.eps)) if denom > self.eps else float("nan")
        return {
            "KL": float(np.sum(np.clip(p, self.eps, None) * np.log(np.clip(p, self.eps, None) / np.clip(q, self.eps, None)))),
            "CC": cc,
            "SIM": float(np.minimum(p, q).sum()),
        }


@lru_cache(maxsize=16)
def _transport_problem_matrices(shape: Tuple[int, int]):
    h, w = shape
    n = int(h * w)
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.column_stack([yy.reshape(-1), xx.reshape(-1)]).astype(np.float64)
    cost = cdist(coords, coords, metric="euclidean").reshape(-1)

    a_eq = lil_matrix((2 * n, n * n), dtype=np.float64)
    for i in range(n):
        a_eq[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        a_eq[n + j, j::n] = 1.0
    return cost, a_eq.tocsr()
