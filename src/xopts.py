
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class XOpts:
    # ---- SDR filter ----
    sdr_filter_enable: bool = False
    sdr_bits: int = 256
    sdr_multi_hash: int = 1              # 1~3 권장
    sdr_prefilter: int = 4096            # 총합 목표 M
    sdr_prefilter_per_hash: Optional[int] = None  # None이면 자동 분배
    sdr_rebuild_every: int = 200
    sdr_pack_uint64: bool = True         # uint64 + np.bit_count 최적화
    sdr_union_cap: int = 16384           # 해시 합집합 상한
    sdr_dynamic_prefilter: bool = True   # L 큰 경우 sqrt(L) 기반 동적 산정
    sdr_dynamic_alpha: float = 1.0       # 동적 계수 (M = max(min, alpha*sqrt(L)))
    sdr_min_prefilter: int = 2048        # 동적 M 하한

    # ---- Predictive dendrite gating ----
    predictive_merge_enable: bool = True
    predictive_beta: float = 0.35
    predictive_delta_ema: float = 0.7
    predictive_delta_min: float = 1e-3

    # ---- Lateral inhibition ----
    inhibit_enable: bool = True
    inhibit_topN: int = 32
    inhibit_gamma: float = 0.22
    inhibit_iters: int = 1

    # ---- Neuromodulated plasticity ----
    neuromod_enable: bool = True
    neuromod_depth: float = 0.5
    neuromod_scale: float = 2.0
    neuromod_alpha_floor: float = 0.60

    # ---- Maintenance (tiled Top-K interference) ----
    maintenance_brain_enable: bool = True
    maint_tile_rows: int = 1536
    maint_topk: Optional[int] = None       # None -> use ~log(L)+1
    maint_sample_pairs: int = 50000
    maint_quantile_q: Optional[float] = None  # None -> use hyper.interference_q

    # ---- Streaming / pinned safeguards ----
    stream_cover_ratio: float = 0.5
    stream_chunk_rows: int = 0
    stream_use_fp16: bool = False
    pinned_bytes_max: int = 256 * 1024 * 1024

    # ---- Meta wrapping (compat) ----
    meta_wrap_enable: bool = True
