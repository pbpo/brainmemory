
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .xopts import XOpts
from .utils import normalize, cos, safe_call, DummyLock
from .sdr import SDRIndexer
from .streams import GpuStreamer
from .inhibition import LateralInhibitor

def wrap_retrieve(mem, x: XOpts, sdr: SDRIndexer, streamer: GpuStreamer, inhibitor: LateralInhibitor):
    if getattr(mem, "_brain_wrapped_retrieve", False): return
    if not hasattr(mem, "retrieve_for_query"): return
    orig = mem.retrieve_for_query

    def _wrap_meta_if_needed(res, K_cap: int, return_meta: bool):
        if not x.meta_wrap_enable or not return_meta:
            return res
        if isinstance(res, list) and (len(res) == 0 or isinstance(res[0], dict)):
            return res
        # string list -> meta wrap
        out = []
        for s in (res or [])[:K_cap]:
            out.append({"content": s, "score": 0.0, "node_id": -1, "tier": "L1",
                        "chapter": 0, "source": "HTNR", "doc_id": None, "chunk_id": None,
                        "meta": {"tier": "L1"}})
        return out

    def new_retrieve_for_query(query: str, *, K_cap: int = 8, mutate: bool = False,
                               flush: bool = False, return_meta: bool = False,
                               use_rag: bool = True):
        if int(K_cap) <= 0:
            return [] if return_meta else []

        if not x.sdr_filter_enable:
            res = safe_call(orig, query, K_cap=K_cap, mutate=mutate, flush=flush, return_meta=True, use_rag=use_rag)
            res = _wrap_meta_if_needed(res, K_cap, return_meta)
            # inhibition on L1 only
            try:
                chosen = [(int(r.get("node_id", -1)), float(r.get("score", 0.0)))
                          for r in res if r.get("tier") == "L1" and int(r.get("node_id", -1)) >= 0]
                if x.inhibit_enable and len(chosen) >= 3:
                    chosen2 = inhibitor.apply(chosen)
                    score_map = {nid: sc for nid, sc in chosen2}
                    for r in res:
                        nid = int(r.get("node_id", -1))
                        if r.get("tier") == "L1" and nid in score_map:
                            r["score"] = float(score_map[nid])
                    res = sorted(res, key=lambda z: -float(z.get("score", 0.0)))
            except Exception:
                pass
            return res if return_meta else [r.get("content", "") for r in res]

        # ---- SDR-enabled fast path ----
        if flush and hasattr(mem, "flush_buffer"):
            mem.flush_buffer()

        with getattr(mem, "_lock", DummyLock()):
            leaf = getattr(mem, "_leaf_mat", None)
            leaf_ids = list(getattr(mem, "_leaf_ids", [])) if getattr(mem, "_leaf_ids", None) is not None else []
            M_agg = getattr(mem, "M_agg", None)
        if leaf is None or not leaf_ids:
            res = safe_call(orig, query, K_cap=K_cap, mutate=mutate, flush=False, return_meta=return_meta, use_rag=use_rag)
            return res

        q = mem.model.encode([query])[0]; q = normalize(q, 1e-8)

        # temporal anchor (history best) + blend
        anchor = M_agg.copy() if M_agg is not None else None
        hist = getattr(mem, "M_history", None)
        if isinstance(hist, list) and len(hist) > 0:
            sims = [float(np.dot(q, normalize(np.asarray(h, dtype=np.float32), 1e-8))) for h in hist[-8:]]
            if len(sims) > 0:
                best = int(np.argmax(np.array(sims)))
                anchor = normalize(np.asarray(hist[-8:][best], dtype=np.float32), 1e-8)

        mode = (getattr(mem.hyper, "query_blend", "auto") or "none").lower()
        comp = q.copy()
        if anchor is not None and mode != "none":
            cos_qm = cos(q, anchor, 1e-8)
            if mode == "fixed":
                w = float(np.clip(getattr(mem.hyper, "query_blend_fixed", 0.4), 0.0, 0.95))
            else:
                lo = float(getattr(mem.hyper, "query_blend_cos_low", 0.20))
                hi = float(getattr(mem.hyper, "query_blend_cos_high", 0.80))
                t = (cos_qm - lo) / max(1e-6, (hi - lo))
                w = float(np.clip(t, 0.0, 1.0)) * float(np.clip(getattr(mem.hyper, "query_blend_fixed", 0.4), 0.0, 0.95))
            comp = normalize((1.0 - w) * q + w * anchor, 1e-8)

        # SDR prefilter
        L = len(leaf_ids)
        if L > 2 * int(x.sdr_prefilter):
            rows = sdr.filter_rows(comp, desired_M=int(x.sdr_prefilter))
        else:
            rows = np.arange(L, dtype=np.int64)

        # scores (cosine on either CPU/GPU)
        scores = streamer.dot_rows(rows, comp, return_cosine=True)
        if scores.size == 0:
            return [] if return_meta else []
        k = min(int(K_cap), scores.size)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        chosen = [(leaf_ids[int(rows[int(i)])], float(scores[int(i)])) for i in idx]

        # inhibition
        if x.inhibit_enable and len(chosen) >= 3:
            chosen = inhibitor.apply(chosen)

        # build outputs
        outs = []
        for nid, sc in chosen[:K_cap]:
            node = mem.nodes.get(int(nid)) if hasattr(mem, "nodes") else None
            if node is None:
                continue
            text = getattr(node, "content", "") or ""
            outs.append({
                "content": text, "score": float(sc), "node_id": int(nid),
                "chapter": int(getattr(node, "chapter", 0)), "source": getattr(node, "source", "HTNR"),
                "tier": "L1", "doc_id": None, "chunk_id": None,
                "meta": {"node_id": int(nid), "tier": "L1", "chapter": int(getattr(node, "chapter", 0))},
            })

        # RAG augment if top1 low
        if use_rag and hasattr(mem, "external_rag") and mem.external_rag is not None:
            s = np.array([o["score"] for o in outs], dtype=np.float32)
            if s.size == 0 or (float(np.max(s)) < float(getattr(mem.hyper, "rag_top1_thr", 0.65))):
                try:
                    rag = mem._do_rag(query, K_cap, return_meta=True)
                    seen, final = set(), []
                    for r in outs + rag:
                        key = (r.get("node_id", -1), r.get("tier","L1"), r.get("chapter",0), int(np.int64(abs(hash(r.get("content",""))))))
                        if key in seen: 
                            continue
                        seen.add(key); final.append(r)
                        if len(final) >= K_cap: break
                    outs = final
                except Exception:
                    pass

        if mutate:
            for r in outs:
                nid = int(r["node_id"])
                if hasattr(mem, "nodes") and nid in mem.nodes:
                    mem.nodes[nid].V = min(2.0, float(getattr(mem.nodes[nid], "V", 0.0)) + 0.05 * float(r["score"]))

        return outs if return_meta else [r["content"] for r in outs]

    mem._orig_retrieve_for_query = orig
    mem.retrieve_for_query = new_retrieve_for_query
    mem._brain_wrapped_retrieve = True
