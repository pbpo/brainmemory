
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import math

try:
    import torch
    _HAS_TORCH = True
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    _HAS_TORCH = False
    _TORCH_CUDA = False

from .xopts import XOpts
from .utils import normalize

def _tiled_topk_cosine(mem, x: XOpts, K: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (topk_idx[L, k], topk_vals[L, k], q_quantile) using tiled matmul.
       Also returns approximate tail-quantile q based on sampling.
    """
    leaf = getattr(mem, "_leaf_mat", None)
    if leaf is None or leaf.size == 0:
        return np.zeros((0, K), dtype=np.int64), np.zeros((0, K), dtype=np.float32), 0.0
    X = leaf  # rows assumed normalized? Not guaranteed -> compute cosine properly below
    L, D = X.shape
    # Precompute row norms
    norms = np.linalg.norm(X, axis=1).astype(np.float32); norms = np.maximum(norms, 1e-8)
    # Prepare containers
    k = min(K, L-1) if L > 1 else 0
    if k <= 0:
        return np.zeros((L, 0), dtype=np.int64), np.zeros((L, 0), dtype=np.float32), 0.0
    top_idx = np.full((L, k), -1, dtype=np.int64)
    top_val = np.full((L, k), -np.inf, dtype=np.float32)

    # Sampling for quantile q
    sample_pairs = int(x.maint_sample_pairs)
    if L > 1 and sample_pairs > 0:
        rng = np.random.default_rng(getattr(mem, "_seed", 42))
        i = rng.integers(0, L, size=sample_pairs, endpoint=False)
        j = rng.integers(0, L, size=sample_pairs, endpoint=False)
        mask = (i != j)
        i, j = i[mask], j[mask]
        s = (X[i] @ X[j]) / (norms[i] * norms[j])
        s = np.clip(s.astype(np.float32), -1.0, 1.0)
        q = float(np.quantile(s, float(getattr(mem.hyper, "interference_q", 0.95))))
    else:
        q = float(getattr(mem.hyper, "interference_q", 0.95))

    # Decide device
    use_gpu = bool(getattr(mem, "_use_torch", False) and _HAS_TORCH and _TORCH_CUDA)
    tile = int(max(256, int(x.maint_tile_rows)))
    for start in range(0, L, tile):
        end = min(start + tile, L)
        A = X[start:end]  # [T, D]
        if use_gpu:
            try:
                device = getattr(mem, "_device", torch.device("cuda"))
                At = torch.from_numpy(np.ascontiguousarray(A)).pin_memory().to(device, non_blocking=True)
                Xt = torch.from_numpy(np.ascontiguousarray(X)).pin_memory().to(device, non_blocking=True)
                # cosine = (A @ X^T)/(||A||*||X||), norms broadcast
                res = torch.matmul(At, Xt.t()).detach().cpu().numpy().astype(np.float32)
            except Exception:
                res = (A @ X.T).astype(np.float32)
        else:
            res = (A @ X.T).astype(np.float32)

        # cosine normalization
        denom = (norms[start:end][:, None] * norms[None, :])
        res = res / np.maximum(denom, 1e-8)
        np.clip(res, -1.0, 1.0, out=res)

        # For each row, update top-k excluding self
        for r in range(end - start):
            row = res[r]
            row[start + r] = -np.inf  # exclude self
            if k < L:
                idx = np.argpartition(row, -(k))[-k:]
            else:
                idx = np.arange(L, dtype=np.int64)
            vals = row[idx]
            # merge with existing
            cur_idx = np.concatenate([top_idx[start + r][top_val[start + r] > -np.inf], idx])
            cur_vals = np.concatenate([top_val[start + r][top_val[start + r] > -np.inf], vals])
            # unique by index, keep best value
            if cur_idx.size:
                order = np.argsort(-cur_vals)
                cur_idx = cur_idx[order]; cur_vals = cur_vals[order]
                # dedupe keeping first occurrence
                seen = set(); dedup_idx = []
                for tI in cur_idx:
                    if int(tI) in seen: continue
                    seen.add(int(tI)); dedup_idx.append(True)
                dedup_idx = np.array(dedup_idx, dtype=bool)
                cur_idx = cur_idx[dedup_idx]; cur_vals = cur_vals[dedup_idx]
                take = min(k, cur_idx.size)
                top_idx[start + r, :take] = cur_idx[:take]
                top_val[start + r, :take] = cur_vals[:take]

    return top_idx, top_val, q

def wrap_maintenance(mem, x: XOpts):
    if getattr(mem, "_brain_wrapped_maint", False): return
    if not hasattr(mem, "_maintenance"): return
    orig = mem._maintenance

    def new_maintenance():
        if not x.maintenance_brain_enable:
            return orig()

        if not getattr(mem, "leaves", None):
            return

        # Homeostasis (same idea as original)
        try:
            if mem.step % int(getattr(mem.hyper, "homeo_interval", 50)) == 0:
                V_int = np.array([max(0.0, mem.nodes[i].V - mem.nodes[i].E) for i in mem.leaves], dtype=np.float32)
                if V_int.size >= 4:
                    q1, q3 = np.percentile(V_int, [25, 75])
                    iqr = max(1e-6, q3 - q1)
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    W = np.clip(V_int, lo, hi)
                    mu, sd = float(np.mean(W)), float(np.std(W) + 1e-6)
                    z = (W - mu) / sd
                    s = 2.0 * (1.0 / (1.0 + np.exp(-z)))
                    for idx, nid in enumerate(mem.leaves):
                        mem.nodes[nid].V = min(2.0, float(s[idx] + mem.nodes[nid].E))
        except Exception:
            pass

        # Interference via tiled Top-K
        leaf = getattr(mem, "_leaf_mat", None)
        if leaf is None:
            if hasattr(mem, "_refresh_leaf_cache"):
                mem._refresh_leaf_cache()
            leaf = getattr(mem, "_leaf_mat", None)
        L = 0 if leaf is None else leaf.shape[0]
        if L >= 8 and leaf is not None:
            K = int(x.maint_topk or max(1, int(round(math.log(L + 1)))))
            idx, vals, q = _tiled_topk_cosine(mem, x, K + 1)  # +1 candidate list (exclude self already)
            # density & decay
            for r in range(L):
                nbrs = [j for j in idx[r] if j >= 0 and j != r]
                strong_vals = [float(vals[r][t]) for t, j in enumerate(idx[r]) if j >= 0 and float(vals[r][t]) >= q and j != r]
                avg = float(np.mean(strong_vals)) if strong_vals else 0.0
                density = 0.5 * math.log(1.0 + len(strong_vals)) + 0.5 * avg
                nid = mem.leaves[r]
                protection = math.tanh(mem.nodes[nid].E)
                dec = max(0.0, 0.05 * density - 0.01 * protection)
                if dec > 0:
                    mem.nodes[nid].V = max(0.0, mem.nodes[nid].V - dec)

        # thresholds
        try:
            occupancy = float(len(mem.leaves) / max(1, mem.hyper.max_leaves))
            mem.auto.update_thresholds(rl=np.clip(occupancy, 0.0, 2.0))
        except Exception:
            pass

        # over-capacity merges
        merges = 0
        while len(mem.leaves) > mem.hyper.max_leaves and merges < mem.hyper.max_merges_per_cycle:
            if not mem._merge_once():
                break
            merges += 1

        # Phase-R
        try:
            mem._phase_reorg()
        except Exception:
            pass

        # Opportunistic merge
        try:
            if getattr(mem.hyper, "opportunistic_merge", False) and len(mem.leaves) >= mem.hyper.anchor_recent + 2:
                mem._merge_once()
        except Exception:
            pass

        # Metrics log
        try:
            parents_in_leaves = sum(1 for nid in mem.leaves if mem.nodes[nid].children)
            mem.log.write({"t": "metrics", "leaves": len(mem.leaves),
                           "parents_in_leaves": parents_in_leaves,
                           "tau_w": mem.auto.tau_w, "tau_R": mem.auto.tau_R})
        except Exception:
            pass

    mem._orig_maintenance = orig
    mem._maintenance = new_maintenance
    mem._brain_wrapped_maint = True
