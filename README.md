
# htnr_brain — Brain‑inspired runtime upgrades for HTNRMemoryV55X

> **What you get**
>
> 1) **SDR 2‑stage search** (multi‑hash Hamming prefilter → cosine re‑rank)  
> 2) **Predictive dendrite gating** (ΔM‑aligned merge)  
> 3) **Lateral inhibition** at retrieval (soft‑WTA for diversity)  
> 4) **Neuromodulated plasticity** (surprise→α)  
> 5) **Tiled Top‑K interference** (OOC; avoids O(n²) pairwise memory blow‑up)  
> 6) **GPU OOC streaming** matvec with pinned safeguards & dynamic chunking

All features are **drop‑in**: no change to your `HTNRMemoryV55X` core.  
Safe fallbacks: CPU when CUDA/Torch unavailable; original methods preserved as `_orig_*`.

---

## Install

```python
from htnr_brain import install, XOpts

x = XOpts(
    # SDR
    sdr_filter_enable=True, sdr_bits=256, sdr_multi_hash=2, sdr_prefilter=4096,
    sdr_pack_uint64=True, sdr_union_cap=16384, sdr_rebuild_every=200,
    # Gating / inhibition / plasticity
    predictive_merge_enable=True, predictive_beta=0.35,
    inhibit_enable=True, inhibit_topN=32, inhibit_gamma=0.22, inhibit_iters=1,
    neuromod_enable=True, neuromod_depth=0.5, neuromod_scale=2.0, neuromod_alpha_floor=0.60,
    # Maintenance
    maintenance_brain_enable=True, maint_tile_rows=1536, maint_sample_pairs=50000,
    # Streaming / pinned safeguards
    stream_cover_ratio=0.5, stream_chunk_rows=0, stream_use_fp16=False,
    pinned_bytes_max=256*1024*1024,
)
mem = HTNRMemoryV55X(embedder, hyper=Hyper(use_torch=True, batch_freeze_enable=True))
install(mem, x)
```

> Uses your existing `mem.hyper` for thresholds (e.g., `interference_q`, `rag_top1_thr`).

---

## Key Design Notes

### SDR (Sparse Distributed Representation)

- **Multi‑hash**: use `sdr_multi_hash ≥ 2` to union top‑M per hash (higher recall).  
- **Packing**: `sdr_pack_uint64=True` packs to `uint64` and uses `np.bit_count` (NumPy 1.24+) for fast Hamming.  
- **Dynamic M**: if `sdr_dynamic_prefilter=True`, target `M` grows ~`sqrt(L)` for large indexes.  
- Codes rebuild when `step` cadence hits or leaf size changes.

### GPU OOC streaming

- Host subset is pinned only when size ≤ `pinned_bytes_max` (defaults 256MB) to avoid OS page‑lock OOM.  
- Two CUDA streams with ping‑pong buffers overlap H2D copies with GEMV.  
- We always return **cosine** scores (dot / row‑norm).

### Lateral inhibition

- Applies to top‑N only (default 32), sequential soft‑WTA (earlier winners inhibit later).  
- Robust to missing nodes and clamps negatives.

### Predictive dendrite gating (ΔM)

- Adds ΔM‑alignment term to gate logits; stable sigmoid; defaults for Va/Vb if absent.  
- ΔM updated as EMA in `_update_contexts`; neuromod plasticity modulates α in batch ingest.

### Maintenance: tiled Top‑K interference

- Avoids building full `L×L` similarity. For each tile of rows, multiply by full matrix and keep per‑row top‑K.  
- Tail quantile `q` estimated via random pair sampling; decay policy matches original form.  
- Falls back to CPU if CUDA fails.

---

## Compatibility

- Original methods preserved as `_orig_*`.  
- Retrieval wrapper honors original signature via `inspect.signature`.  
- Meta wrapping (`meta_wrap_enable=True`) converts string lists to meta dicts when `return_meta=True` is expected upstream.

---

## Tuning Cheatsheet

- **Speed first**: `sdr_multi_hash=2`, `sdr_prefilter=4096`, `maint_tile_rows=1536`  
- **Recall first**: increase `sdr_bits` (→ 384/512), `sdr_union_cap`, `sdr_prefilter`  
- **Diversity**: `inhibit_gamma=0.18~0.25`, `inhibit_iters=1`  
- **Stability**: decrease `predictive_beta`, increase `neuromod_alpha_floor`

---

## Rollback

If you need to disable parts:

```python
# restore originals
mem._update_contexts = mem._orig_update_contexts
mem._ingest_batch_freeze = mem._orig_ingest_batch_freeze
mem._dendritic_gating = mem._orig_dendritic_gating
mem.retrieve_for_query = mem._orig_retrieve_for_query
mem._maintenance = mem._orig_maintenance
```

---

## License

MIT (include this module alongside your project).
