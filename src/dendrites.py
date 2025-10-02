
from __future__ import annotations
import math
import numpy as np
from .xopts import XOpts
from .utils import normalize, cos

def wrap_update_contexts(mem, x: XOpts):
    if getattr(mem, "_brain_wrapped_update_ctx", False): return
    if not hasattr(mem, "_update_contexts"): return
    orig = mem._update_contexts

    def new_update_contexts(emb: np.ndarray):
        prev = mem.M_agg.copy() if getattr(mem, "M_agg", None) is not None else None
        out = orig(emb)
        try:
            if prev is not None:
                d = normalize(mem.M_agg - prev, 1e-8)
                if not hasattr(mem, "_delta_M") or mem._delta_M is None or mem._delta_M.size != d.size:
                    mem._delta_M = np.zeros_like(d, dtype=np.float32)
                beta = float(x.predictive_delta_ema)
                mem._delta_M = normalize(beta * mem._delta_M + (1.0 - beta) * d, 1e-8)
        except Exception:
            pass
        return out

    mem._orig_update_contexts = orig
    mem._update_contexts = new_update_contexts
    mem._brain_wrapped_update_ctx = True

def wrap_ingest_batch_freeze(mem, x: XOpts):
    if getattr(mem, "_brain_wrapped_batch", False): return
    if not hasattr(mem, "_ingest_batch_freeze"): return
    orig = mem._ingest_batch_freeze

    def new_ingest_batch_freeze(batch):
        if not hasattr(mem, "auto") or not hasattr(mem.auto, "update_alpha"):
            return orig(batch)
        orig_update = mem.auto.update_alpha
        def mod_update_alpha(pe: float) -> float:
            a = orig_update(pe)
            if x.neuromod_enable:
                s = float(np.tanh(float(x.neuromod_scale) * float(max(0.0, pe))))
                a = float(np.clip(a * (1.0 - float(x.neuromod_depth) * s), float(x.neuromod_alpha_floor), 0.98))
            return a
        mem.auto.update_alpha = mod_update_alpha
        try:
            return orig(batch)
        finally:
            mem.auto.update_alpha = orig_update

    mem._orig_ingest_batch_freeze = orig
    mem._ingest_batch_freeze = new_ingest_batch_freeze
    mem._brain_wrapped_batch = True

def wrap_dendritic_gating(mem, x: XOpts):
    if getattr(mem, "_brain_wrapped_dgate", False): return
    if not hasattr(mem, "_dendritic_gating"): return
    orig = mem._dendritic_gating

    def new_dendritic_gating(na, nb):
        try:
            dm = getattr(mem, "_delta_M", None)
            M = getattr(mem, "M_agg", None)
            if x.predictive_merge_enable and dm is not None and M is not None:
                if float(np.linalg.norm(dm)) > float(x.predictive_delta_min):
                    Ri = cos(na.emb, M); Rj = cos(nb.emb, M)
                    si = cos(na.emb, dm); sj = cos(nb.emb, dm)
                    diff = 3.0 * (Ri - Rj) + float(x.predictive_beta) * (si - sj)
                    if diff >= 20: gate_i = 1.0
                    elif diff <= -20: gate_i = 0.0
                    else: gate_i = 1.0 / (1.0 + math.exp(-diff))
                    gate_j = 1.0 - gate_i
                    Va = float(getattr(na, "V", 1.0)); Vb = float(getattr(nb, "V", 1.0))
                    gated = gate_i * na.emb + gate_j * nb.emb
                    s = max(1e-6, Va + Vb)
                    vw = (Va / s) * na.emb + (Vb / s) * nb.emb
                    new = 0.5 * gated + 0.5 * vw
                    lam = float(getattr(mem.hyper, "shrink_lambda", 0.0))
                    new = (1 - lam) * new + lam * M
                    return normalize(new, 1e-8)
        except Exception:
            pass
        return orig(na, nb)

    mem._orig_dendritic_gating = orig
    mem._dendritic_gating = new_dendritic_gating
    mem._brain_wrapped_dgate = True
