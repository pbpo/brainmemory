# htnr_memory_v56_ann_plus.py
# -*- coding: utf-8 -*-
"""
HTNR Memory — V56-ANN+ (EmbeddingGemma edition, patched)
- 임베딩: google/embeddinggemma-300m (SentenceTransformers)
- 쿼리/문서 임베딩 분리 사용(encode_query / encode_document)
- ANN(HNSW) 기반 간섭 계산 → O(L log L)
- 머지 후보도 ANN 우선, 실패 시 소형 서브매트릭스 폴백(메모리 예산 기반)
- MMR 문장 임베딩 사전 계산 및 재사용 (중복 억제: char 3-gram Jaccard)
- OOC GPU 스트리밍(matvec) 어댑티브 청크 튜닝 + 버퍼 재사용
- HNSW capacity slack & resize + lazy 삭제 누적 시 리빌드
- EVT: PWM(L-moment) 기본(Weibull ξ≤0 강제 + 끝점 보장), Pickands 보조(ξ≤0로 클립)
- Page–Hinkley: 지수망각 분산 갱신식 교정 + var_warmup + min_std
- 홈오스타시스: μ + σ·sigmoid(z) 형태로 재정의(E는 감쇠 보호에만 사용)
- 스냅샷: float16 압축 저장 옵션 (로드시 float32 복원)
- 캐시 키: 임베더 식별자 포함

필수:
    pip install -U sentence-transformers transformers hnswlib

예시:
    mem = HTNRMemory(
        EmbeddingGemmaEmbedder("google/embeddinggemma-300m", device=None, batch_size=64, normalize_embeddings=True),
        hyper=Hyper(use_torch=True, ann_backend="auto"),
        xopts=XOpts(proactive_compress_enable=False, lsh_density_enable=False),
    )
    mem.process_and_add("hello world")
    mem.flush_buffer()
    print(mem.retrieve_for_query("greeting", K_cap=5))
"""

from __future__ import annotations

import os, io, re, math, json, time, logging, hashlib, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Protocol, Deque, Sequence
from collections import deque, OrderedDict, defaultdict

import numpy as np

# ---------- Optional ANN ----------
try:
    import hnswlib as _hnswlib
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

# ---------- Optional Torch (GPU) ----------
try:
    import torch
    _HAS_TORCH = True
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _TORCH_CUDA = False

# ---------- Logging ----------
logger = logging.getLogger("HTNRV56ANN+")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- Pinned host tensor allocator ----------
_HOST_PIN_STORE: Dict[int, "torch.Tensor"] = {}

def allocate_pinned_host_tensor(shape: Tuple[int, ...], *, dtype: "torch.dtype" = None):
    if not (_HAS_TORCH and _TORCH_CUDA):
        return None
    import torch as _torch
    if dtype is None:
        dtype = _torch.float32
    try:
        t = _torch.empty(shape, dtype=dtype, pin_memory=True)  # CPU pinned
        _HOST_PIN_STORE[id(t)] = t  # prevent GC
        return t
    except Exception as e:
        logger.warning(f"allocate_pinned_host_tensor failed: {e}")
        return None

def free_pinned_host_tensor(t):
    if t is not None:
        _HOST_PIN_STORE.pop(id(t), None)

# ---------- Utils ----------
def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n <= eps:
            return arr
        return arr / n
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.divide(arr, (n + eps), out=np.zeros_like(arr), where=(n > eps))
    return out

def _fast_dot(a: np.ndarray, b: np.ndarray) -> float:
    v = float(np.asarray(a, dtype=np.float32).dot(np.asarray(b, dtype=np.float32)))
    if v > 1.0: return 1.0
    if v < -1.0: return -1.0
    return v

def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]

# 문장 분할: 마침표/개행 + 유니코드 따옴표(”)까지 처리
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n\u3002\uFF01\uFF1F\"”])[\s\)\]\u3001\u3000\u201d]*")

def sent_tokenize(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s and s.strip()]

# 한국어 호환 중복 억제: 문자 3‑그램 자카드
def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set:
        s2 = "".join(s.split())
        if not s2:
            return set()
        if len(s2) <= n:
            return {s2}
        return {s2[i:i+n] for i in range(len(s2)-n+1)}
    A, B = grams(a), grams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def _stable_hash(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

# ---------- Summarization (MMR) ----------
def mmr_select(query_vec: np.ndarray,
               sentences: List[str],
               encode_fn=None,
               precomputed_embs: Optional[np.ndarray] = None,
               *,
               topk: int = 2,
               lambda_mmr: float = 0.7,
               max_sentences: int = 64) -> List[str]:
    if not sentences:
        return []
    if len(sentences) > max_sentences:
        sentences = sorted(sentences, key=len, reverse=True)[:max_sentences]

    if precomputed_embs is not None and precomputed_embs.shape[0] == len(sentences):
        embs = precomputed_embs
    elif encode_fn is not None:
        try:
            embs = encode_fn(sentences)
        except Exception:
            return sentences[:topk]
    else:
        logger.warning("MMR called without embeddings; falling back to length-based pick.")
        return sorted(sentences, key=len, reverse=True)[:topk]

    embs = _normalize(np.asarray(embs, dtype=np.float32), 1e-8)
    q = _normalize(query_vec, 1e-8)
    sims = embs @ q

    N = len(sentences)
    cand_mask = np.ones(N, dtype=bool)
    selected: List[int] = []

    while np.any(cand_mask) and len(selected) < topk:
        remaining = np.nonzero(cand_mask)[0]
        rel = sims[remaining]
        if selected:
            red = np.max(embs[remaining] @ embs[selected].T, axis=1)
        else:
            red = np.zeros_like(rel)
        score = lambda_mmr * rel - (1 - lambda_mmr) * red

        best_local = int(np.argmax(score))
        j = int(remaining[best_local])
        selected.append(j)
        cand_mask[j] = False

    out = []
    for s in [sentences[i] for i in selected]:
        # 문자 n‑그램 자카드로 중복 억제
        if not out or jaccard_char_ngrams(out[-1], s, n=3) < 0.65:
            out.append(s)
    return out[:topk]

# ------------------------------------------------------------
# Page–Hinkley (with optional forgetting and variance guard)
# ------------------------------------------------------------
@dataclass
class PageHinkley:
    delta: float = 0.0
    lam: float = 4.0
    alpha: float = 1.0              # 지수망각(1.0이면 Welford)
    var_guard: bool = True
    var_warmup: int = 64            # 워밍업 길이: 이 전에는 분산 가드 미적용
    min_std: float = 0.05           # 표준편차 하한(초기 과민 방지)
    reset_on_change: bool = True

    # state
    mean: float = 0.0
    var_m2: float = 0.0             # 지수망각 하에서 누적 2차 모멘트
    count: int = 0
    gp: float = 0.0
    gn: float = 0.0
    stable_run: int = 0

    def reset(self) -> None:
        self.mean = 0.0
        self.var_m2 = 0.0
        self.count = 0
        self.gp = 0.0
        self.gn = 0.0
        self.stable_run = 0

    def _update_moments(self, x: float) -> Tuple[float, float]:
        self.count += 1
        if self.alpha >= 1.0 or self.count == 1:
            # Welford (무망각)
            delta = x - self.mean
            self.mean += delta / max(1, self.count)
            self.var_m2 += delta * (x - self.mean)
            std = math.sqrt(max(0.0, self.var_m2 / max(1, self.count - 1)))
            return self.mean, std
        else:
            # 지수망각 분산 갱신 (교차항 포함)
            a = float(np.clip(self.alpha, 1e-6, 1.0))
            prev_mean = self.mean
            self.mean = a * self.mean + (1.0 - a) * x
            diff_new = x - self.mean
            diff_old = x - prev_mean
            self.var_m2 = max(0.0, a * (self.var_m2 + (1.0 - a) * diff_old * diff_new))
            std = math.sqrt(self.var_m2)
            return self.mean, std

    def update(self, x: float) -> bool:
        if not np.isfinite(x):
            self.stable_run += 1
            return False
        mu, std = self._update_moments(float(x))
        # NOTE: 워밍업 전에는 분산 표준화(z-score)를 적용하지 않습니다.
        if self.var_guard and (self.count >= max(2, int(self.var_warmup))):
            std = max(std, float(self.min_std))
            z = (x - mu) / std
        else:
            z = x - mu

        y = z - self.delta
        a = float(np.clip(self.alpha, 1e-6, 1.0))
        self.gp = max(0.0, a * self.gp + y)
        self.gn = min(0.0, a * self.gn + y)
        fired = (self.gp > self.lam) or (abs(self.gn) > self.lam)
        if fired:
            if self.reset_on_change:
                self.gp = 0.0; self.gn = 0.0
            self.stable_run = 0
            return True
        else:
            self.stable_run += 1
            return False

# ------------------------------------------------------------
# EVT tail fitter (GPD via PWM / Pickands with Weibull constraint)
# ------------------------------------------------------------
@dataclass
class EVTConfig:
    q0: float = 0.90
    target_q: float = 0.95
    min_tail: int = 200
    clip_low: float = -1.0
    clip_high: float = 1.0
    method: str = "pwm"         # {"pwm","pickands"}

@dataclass
class EVTFit:
    u: float; xi: float; beta: float; tail_n: int

class EVTFitter:
    def __init__(self, cfg: EVTConfig):
        self.cfg = cfg

    # --- Pickands: xi = log((x_{n-2k}-x_{n-4k})/(x_{n-k}-x_{n-2k})) / log 2
    def _pickands(self, xs: np.ndarray) -> Tuple[float, float]:
        # xs: 초과분(>u) 오름차순
        n = xs.size
        k = max(1, n // 4)
        x_nk = xs[n - k - 1]
        x_2k = xs[n - 2 * k - 1] if (n - 2 * k - 1) >= 0 else xs[0]
        x_4k = xs[n - 4 * k - 1] if (n - 4 * k - 1) >= 0 else xs[0]
        num = max(1e-12, x_2k - x_4k)
        den = max(1e-12, x_nk - x_2k)
        xi = float(np.log(num / den) / np.log(2.0))
        xi = float(np.clip(xi, -0.9, 0.0))  # 권장: 유계 코사인 꼬리 → Weibull 영역만 허용(ξ≤0)
        beta = float(max(1e-8, (1.0 - xi) * (x_nk)))
        return xi, beta

    # --- PWM(L-moment) for GPD (Hosking 1987 표준형, Weibull 제약)
    def _pwm(self, xs: np.ndarray) -> Tuple[float, float]:
        n = xs.size
        if n < 2:
            return self._pickands(xs)

        y = np.sort(xs)
        b0 = float(np.mean(y))
        i = np.arange(1, n + 1, dtype=np.float64)
        w = (i - 1.0) / max(1.0, (n - 1.0))
        b1 = float(np.sum(w * y) / n)

        # L-moments: λ1 = b0, λ2 = 2 b1 - b0, t = λ2/λ1 = (2 b1 - b0)/b0
        t = float((2.0 * b1 - b0) / max(1e-12, b0))
        if not np.isfinite(t) or t <= 0:
            return self._pickands(xs)

        xi = 2.0 - 1.0 / max(1e-6, t)
        beta = b0 * (1.0 - xi)

        # 수치 안정화 + Weibull 영역 강제
        xi = float(np.clip(xi, -0.9, 0.0))    # 꼭: ξ ≤ 0
        beta = float(max(1e-8, beta))
        return xi, beta

    def fit(self, data: np.ndarray) -> Optional[EVTFit]:
        cfg = self.cfg
        arr = np.asarray(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < cfg.min_tail:
            return None
        u = float(np.quantile(arr, float(np.clip(cfg.q0, 0.0, 0.9999))))
        excess = arr[arr > u] - u
        if excess.size < max(50, int(0.1 * cfg.min_tail)):
            return None
        xs = np.sort(excess).astype(np.float64)

        # 우선 PWM(Weibull 제약), 실패 시 Pickands(역시 ξ≤0로 클립)
        try:
            if (cfg.method or "pwm").lower() == "pwm":
                xi, beta = self._pwm(xs)
            else:
                xi, beta = self._pickands(xs)
        except Exception:
            xi, beta = self._pickands(xs)

        # 꼭: Weibull 영역 강제 + 상단 끝점 보정
        xi = float(min(0.0, xi))
        u = float(min(cfg.clip_high - 1e-6, u))
        xF_cap = cfg.clip_high - 1e-6
        if xi < 0.0:
            beta_cap = max(1e-8, (xF_cap - u) * (-xi))  # β ≤ (1−ε − u)|ξ|
            beta = float(min(beta, beta_cap))

        return EVTFit(u=u, xi=xi, beta=beta, tail_n=xs.size)

    def quantile(self, fit: EVTFit, q: float) -> float:
        cfg = self.cfg
        q = float(np.clip(q, cfg.q0 + 1e-6, 1.0 - 1e-9))
        q0 = float(np.clip(cfg.q0, 1e-6, 1.0 - 1e-6))
        xi, beta, u = fit.xi, fit.beta, fit.u
        if abs(xi) < 1e-6:
            z = -math.log((1.0 - q) / (1.0 - q0))
            tau = u + beta * z
        else:
            z = ((1.0 - q) / (1.0 - q0)) ** (-xi) - 1.0
            tau = u + (beta / xi) * z
        # 꼭: 상단 끝점 보장
        if xi < 0.0:
            xF = u - beta / xi
            tau = min(tau, xF - 1e-6)
        return float(np.clip(tau, cfg.clip_low, cfg.clip_high))

# ------------------------------------------------------------
# AutoCal (EVT + PI + PH)
# ------------------------------------------------------------
@dataclass
class AutoCal:
    cos_lower: float = -0.999
    cos_upper: float = 0.999
    evt: EVTConfig = EVTConfig()
    ema_gamma_fast: float = 0.2
    ema_gamma_slow: float = 0.85
    max_rate_tau_w: float = 0.03
    max_rate_tau_R: float = 0.02
    mix_evt_weight: float = 0.65
    buffer_maxlen: int = 8192
    ph_lambda: float = 4.0

    tau_w: float = 0.20
    tau_R: float = math.sqrt((1.0 + 0.20) / 2.0)
    alpha_t: float = 0.90

    def __post_init__(self):
        self.tail_buf: Deque[float] = deque(maxlen=int(self.buffer_maxlen))
        self.ph_alpha = PageHinkley(
            delta=0.0, lam=float(self.ph_lambda), alpha=1.0,
            var_guard=True, var_warmup=64, min_std=0.05, reset_on_change=True
        )
        self._evt = EVTFitter(self.evt)

    # ---------- tail management ----------
    def add_tail_samples(self, sims_flat) -> None:
        sims = np.asarray(sims_flat, dtype=np.float32).ravel()
        lo, hi = float(self.cos_lower), float(self.cos_upper)
        sims = sims[np.isfinite(sims)]
        if sims.size == 0:
            return
        sims = np.clip(sims, lo, hi)
        qmed = float(np.quantile(sims, 0.5))
        for v in sims.tolist():
            if v >= qmed:
                self.tail_buf.append(float(v))

    def _evt_tau(self) -> Optional[float]:
        if len(self.tail_buf) < self.evt.min_tail:
            return None
        data = np.array(self.tail_buf, dtype=np.float32)
        fit = self._evt.fit(data)
        if fit is None:
            return None
        try:
            return float(self._evt.quantile(fit, self.evt.target_q))
        except Exception:
            return None

    def update_alpha(self, pe: float) -> float:
        pe = float(np.clip(pe, 0.0, 10.0))
        _ = self.ph_alpha.update(pe)
        sr = max(5.0, float(self.ph_alpha.stable_run))
        self.alpha_t = float(np.clip(math.exp(-1.0 / sr), 0.60, 0.98))
        return self.alpha_t

    def update_thresholds(self, rl: float) -> Tuple[float, float]:
        rl = float(np.clip(rl, 0.0, 2.0))
        tau_evt = self._evt_tau()
        hat_pi = float(np.clip(self.tau_w + 0.05 * (rl - 0.5), 0.10, 0.98))
        fill = len(self.tail_buf) / max(1.0, float(self.buffer_maxlen))
        w_evt = self.mix_evt_weight * (1.0 - math.exp(-4.0 * fill))
        w_pi = 1.0 - 0.5 * w_evt
        Z = max(1e-6, w_evt + w_pi)
        w_evt /= Z; w_pi /= Z
        blended = w_pi * hat_pi + w_evt * (tau_evt if tau_evt is not None else self.tau_w)

        prev = self.tau_w
        g = float(np.clip(self.ema_gamma_slow, 0.0, 0.999))
        tau_w_new = g * prev + (1 - g) * blended
        step = float(abs(tau_w_new - prev))
        if step > self.max_rate_tau_w:
            tau_w_new = prev + math.copysign(self.max_rate_tau_w, tau_w_new - prev)
        tau_w_new = float(np.clip(tau_w_new, 0.10, 0.98))
        self.tau_w = tau_w_new

        prevR = self.tau_R
        target_R = math.sqrt((1.0 + self.tau_w) / 2.0)
        gR = float(np.clip(self.ema_gamma_fast, 0.0, 0.999))
        tau_R_new = gR * prevR + (1 - gR) * target_R
        dR = float(abs(tau_R_new - prevR))
        if dR > self.max_rate_tau_R:
            tau_R_new = prevR + math.copysign(self.max_rate_tau_R, tau_R_new - prevR)
        self.tau_R = float(np.clip(tau_R_new, 0.0, 1.0))
        return self.tau_w, self.tau_R

    def snapshot(self) -> dict:
        return {
            "tau_w": self.tau_w,
            "tau_R": self.tau_R,
            "alpha_t": self.alpha_t,
            "tail_len": len(self.tail_buf),
            "evt": self.evt.__dict__,
            "cfg": {
                "ema_gamma_fast": self.ema_gamma_fast,
                "ema_gamma_slow": self.ema_gamma_slow,
                "max_rate_tau_w": self.max_rate_tau_w,
                "max_rate_tau_R": self.max_rate_tau_R,
                "mix_evt_weight": self.mix_evt_weight,
            },
        }

    def reset(self) -> None:
        self.tail_buf.clear()
        self.tau_w = 0.20
        self.tau_R = math.sqrt((1.0 + self.tau_w) / 2.0)
        self.alpha_t = 0.90
        self.ph_alpha.reset()

# ------------------------------------------------------------
# Numerics / Hyper / XOpts
# ------------------------------------------------------------
@dataclass
class Numerics:
    eps_norm: float = 1e-8
    eps_cov: float = 1e-6
    ema_gamma_fast: float = 0.2
    ema_gamma_slow: float = 0.85
    max_rate_tau_w: float = 0.03
    max_rate_tau_R: float = 0.02
    cos_lower: float = -0.999
    cos_upper: float = 0.999

@dataclass
class Hyper:
    max_leaves: int = 48
    buffer_size: int = 16
    K_contexts: int = 3
    anchor_recent: int = 6

    cand_prefilter: int = 256
    knn_topk_pairs: int = 4
    max_merges_per_cycle: int = 4
    max_phaseR_per_cycle: int = 3

    shrink_lambda: float = 0.08

    interference_q: float = 0.95
    homeo_interval: int = 50

    rag_topk: int = 6
    rag_entropy_tau: float = 1.0
    rag_entropy_thr: float = 1.2
    rag_top1_thr: float = 0.65

    ann_backend: str = "auto"
    ann_min_leaves: int = 64
    ann_M: int = 32
    ann_efC: int = 120
    ann_efS: int = 64
    ann_capacity_slack: float = 1.5
    ann_capacity_extra: int = 1024

    profile: str = "QA"

    query_blend: str = "auto"        # {"none","fixed","auto"}
    query_blend_fixed: float = 0.40
    query_blend_cos_low: float = 0.20
    query_blend_cos_high: float = 0.80

    mmr_topk_parent: int = 6
    mmr_topk_parent_dynamic: bool = True

    opportunistic_merge: bool = False
    merge_cross_doc: bool = False

    use_torch: bool = False
    torch_force_cpu: bool = False
    gpu_max_pairwise: int = 12000

    batch_freeze_enable: bool = False
    batch_freeze_size: int = 128

    approx_interference_enable: bool = True
    approx_knn_k: int = 32
    approx_tail_sample: int = 50000
    approx_sample_pairs_when_no_ann: int = 200000

    snapshot_store_sent_embs: bool = False
    snapshot_float16: bool = True          # 저장 시 임베딩 float16 압축

    # 정확경로 메모리 예산(MB) — S=sub @ sub.T 상한
    exact_max_sim_matrix_mb: int = 256

@dataclass
class XOpts:
    proactive_compress_enable: bool = False
    proactive_compress_method: str = "lsh"
    proactive_compress_tau_scale: float = 0.90
    proactive_compress_max_clusters: int = 256
    proactive_lsh_bits: int = 12

    lsh_density_enable: bool = False
    lsh_bits: int = 20
    lsh_seed: int = 1337

    dynamic_batch_enable: bool = True
    dynamic_batch_min: int = 64
    dynamic_batch_max: int = 512

# ---------- Data model ----------
@dataclass
class HTNRNode:
    id: int
    content: str
    emb: np.ndarray
    V: float
    D: float
    S: float
    R: float
    E: float
    Chi: float
    parent: Optional[int]
    children: List[int]
    birth_step: int
    chapter: int = 0
    source: str = "Unknown"
    sentences: List[str] = field(default_factory=list)
    sent_embs: Optional[np.ndarray] = None

@dataclass
class BufferItem:
    content: str
    emb: np.ndarray
    E_init: float
    source: str
    chapter_hint: Optional[int]

# ---------- Log ----------
class EventLog:
    def __init__(self, to_file: Optional[str] = None, buffer_size: int = 128,
                 truncate_len: int = 16384, enabled: bool = False):
        self.enabled = bool(enabled)
        self.to_file = to_file
        self.buffer_size = max(8, int(buffer_size))
        self.truncate_len = int(truncate_len)
        self._buf: List[str] = []
        self._lock = threading.RLock()
        self._fh: Optional[io.TextIOBase] = None
        if not self.enabled:
            return
        if to_file:
            try:
                parent = os.path.dirname(os.path.abspath(to_file))
                if parent and parent != ".":
                    os.makedirs(parent, exist_ok=True)
                self._fh = open(to_file, "a", encoding="utf-8")
            except Exception:
                logging.exception("EventLog open failed; logs disabled")
                self.enabled = False
                self._fh = None

    def write(self, rec: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(rec, ensure_ascii=False)
        if len(line) > self.truncate_len:
            line = line[: self.truncate_len] + "...(truncated)"
        do_flush = False
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self.buffer_size:
                do_flush = True
        if do_flush:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._buf:
                return
            data = "\n".join(self._buf) + "\n"
            self._buf.clear()
            if self._fh:
                self._fh.write(data)
                self._fh.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None

# ---------- LRU Cache ----------
class LRUCache:
    def __init__(self, capacity: int = 20000):
        self.capacity = int(capacity)
        self.store: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0; self.misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self.store:
                val = self.store.pop(key)
                self.store[key] = val
                self.hits += 1
                return val.copy()
            self.misses += 1
            return None

    def put(self, key: str, val: np.ndarray) -> None:
        with self._lock:
            arr = np.asarray(val, dtype=np.float32).copy()
            arr.setflags(write=False)
            if key in self.store:
                self.store.pop(key)
            self.store[key] = arr
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)

# ---------- RAG ----------
@dataclass
class RAGResult:
    content: str
    score: float
    metadata: Dict[str, Any]

class ExternalRAGProtocol(Protocol):
    def search(self, query: str, top_k: int) -> List[RAGResult]: ...

class FunctionRAG(ExternalRAGProtocol):
    def __init__(self, fn):
        self.fn = fn
    def search(self, query: str, top_k: int) -> List[RAGResult]:
        try:
            items = self.fn(query, top_k)
            out = []
            for i, it in enumerate(items[:top_k]):
                if isinstance(it, RAGResult):
                    out.append(it); continue
                title = it.get("title") if isinstance(it, dict) else str(it)
                snippet = it.get("snippet") if isinstance(it, dict) else str(it)
                score = float(it.get("score", 1.0 / (i + 1))) if isinstance(it, dict) else 1.0 / (i + 1)
                url = it.get("url") if isinstance(it, dict) else None
                out.append(RAGResult(
                    content=f"[{title}] {snippet}", score=score, metadata={"url": url, "type": "rag"}
                ))
            return out
        except Exception as e:
            logger.warning(f"External RAG error: {e}")
            return []

# ---------- Embedder protocol ----------
class EmbedderProtocol(Protocol):
    def get_sentence_embedding_dimension(self) -> int: ...
    def encode(self, texts: Any) -> np.ndarray: ...
    # 선택: 쿼리/문서 분리 API (EmbeddingGemma가 제공)
    def encode_query(self, texts: Any) -> np.ndarray: ...
    def encode_document(self, texts: Any) -> np.ndarray: ...

# ---------- EmbeddingGemma embedder ----------
class EmbeddingGemmaEmbedder(EmbedderProtocol):
    """
    google/embeddinggemma-300m 전용 임베더.
    - 쿼리/문서 임베딩 분리 사용
    - 캐시도 쿼리/문서 분리
    """
    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 *, device: Optional[str] = None, batch_size: int = 32,
                 normalize_embeddings: bool = True, cache_capacity: int = 40000):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("pip install -U sentence-transformers 필요") from e

        self.model_name = model_name
        self._batch_size = int(max(1, batch_size))
        self._normalize = bool(normalize_embeddings)
        self._lock = threading.RLock()
        self.model = SentenceTransformer(model_name, device=device)
        dim_fn = getattr(self.model, "get_sentence_embedding_dimension", None)
        if not callable(dim_fn):
            raise RuntimeError("SentenceTransformer 모델이 임베딩 차원을 제공하지 않습니다.")
        self.dim = int(dim_fn())

        # 쿼리/문서 캐시 분리 (동일 문자열 충돌 방지)
        self.cache_doc = LRUCache(capacity=cache_capacity)
        self.cache_q = LRUCache(capacity=cache_capacity)
        self.cache_tag = f"EG:{self.model_name}"

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def _encode_with(self, fn, texts: List[str], **extra) -> np.ndarray:
        try:
            result = fn(
                texts,
                convert_to_numpy=True,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                **extra,
            )
        except TypeError:
            try:
                result = fn(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                    **extra,
                )
            except TypeError:
                result = fn(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _encode_cached(self, texts: List[str], cache: LRUCache, encode_fn) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        misses = []
        order = []
        for s in texts:
            v = cache.get(s)
            order.append((s, v))
            if v is None:
                misses.append(s)
        if misses:
            with self._lock:
                pend = []
                for s in misses:
                    if cache.get(s) is None:
                        pend.append(s)
                if pend:
                    arr = encode_fn(pend)
                    if hasattr(arr, "detach"):
                        arr = arr.detach().cpu().numpy()
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    for s, vec in zip(pend, arr):
                        cache.put(s, vec)
        out = []
        for s, v in order:
            if v is not None:
                out.append(v); continue
            v2 = cache.get(s)
            if v2 is None:
                v2 = np.zeros(self.dim, dtype=np.float32)
            out.append(v2)
        return np.stack(out, axis=0)

    def _encode_via_prompts(self, texts: List[str], prompt_candidates: List[str], cache: LRUCache) -> np.ndarray:
        for name in prompt_candidates:
            try:
                return self._encode_cached(
                    texts,
                    cache,
                    lambda xs, pn=name: self._encode_with(self.model.encode, xs, prompt_name=pn),
                )
            except Exception:
                continue
        for prompt in prompt_candidates:
            try:
                return self._encode_cached(
                    texts,
                    cache,
                    lambda xs, pr=prompt: self._encode_with(self.model.encode, xs, prompt=pr),
                )
            except Exception:
                continue
        for prefix in ("query", "passage", "document"):
            try:
                return self._encode_cached(
                    texts,
                    cache,
                    lambda xs, pf=prefix: self._encode_with(
                        self.model.encode, [f"{pf}: {t}" for t in xs]
                    ),
                )
            except Exception:
                continue
        return self._encode_cached(texts, cache, lambda xs: self._encode_with(self.model.encode, xs))

    # ---- Public APIs ----
    def encode_document(self, texts: Any) -> np.ndarray:
        inputs = [texts] if isinstance(texts, str) else list(texts)
        fn = getattr(self.model, "encode_document", None)
        if callable(fn):
            return self._encode_cached(inputs, self.cache_doc, lambda xs: self._encode_with(fn, xs))
        return self._encode_via_prompts(inputs, ["document", "passage", "retrieval_document"], self.cache_doc)

    def encode_query(self, texts: Any) -> np.ndarray:
        inputs = [texts] if isinstance(texts, str) else list(texts)
        fn = getattr(self.model, "encode_query", None)
        if callable(fn):
            return self._encode_cached(inputs, self.cache_q, lambda xs: self._encode_with(fn, xs))
        return self._encode_via_prompts(inputs, ["query", "retrieval_query", "search_query"], self.cache_q)

    def encode(self, texts: Any) -> np.ndarray:
        # 호환성: 기본은 문서 임베딩
        return self.encode_document(texts)

# ---------- (fallback) General SentenceTransformer embedder ----------
class SentenceTransformerEmbedder(EmbedderProtocol):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 *, device: Optional[str] = None, batch_size: int = 32,
                 normalize_embeddings: bool = False, cache_capacity: int = 20000):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("pip install 'sentence-transformers' 필요") from e
        self.model_name = model_name
        self._batch_size = int(max(1, batch_size))
        self._normalize = bool(normalize_embeddings)
        self._lock = threading.RLock()
        self.model = SentenceTransformer(model_name, device=device)
        dim = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            self.dim = int(dim())
        else:
            raise RuntimeError("SentenceTransformer 모델이 임베딩 차원을 제공하지 않습니다.")
        self.cache = LRUCache(capacity=cache_capacity)
        self.cache_tag = f"ST:{self.model_name}"

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: Any) -> np.ndarray:
        inputs = [texts] if isinstance(texts, str) else list(texts)
        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)
        misses, order = [], []
        for s in inputs:
            cached = self.cache.get(s)
            order.append((s, cached))
            if cached is None:
                misses.append(s)
        if misses:
            with self._lock:
                pending = []
                for s in misses:
                    if self.cache.get(s) is None:
                        pending.append(s)
                if pending:
                    vectors = self.model.encode(
                        pending, batch_size=self._batch_size, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=self._normalize,
                    )
                    arr = np.asarray(vectors, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    for text, vec in zip(pending, arr):
                        self.cache.put(text, vec)
        out = []
        for s, cached in order:
            if cached is not None:
                out.append(cached); continue
            vec = self.cache.get(s)
            if vec is None:
                vec = np.zeros(self.dim, dtype=np.float32)
            out.append(vec)
        return np.stack(out, axis=0)

# ---------- HNSW wrapper ----------
class HNSWIndex:
    def __init__(self, dim: int, *, space: str = "cosine",
                 M: int = 32, efC: int = 120, efS: int = 64,
                 capacity: int = 0):
        if not _HAS_HNSW:
            raise RuntimeError("hnswlib not available")
        self.dim = int(dim); self.space = space
        self.M = int(M); self.efC = int(efC); self.efS = int(efS)
        self._idx = _hnswlib.Index(space=space, dim=dim)
        cap = max(1, int(capacity))
        self._idx.init_index(max_elements=cap, ef_construction=self.efC, M=self.M)
        self._idx.set_ef(self.efS)
        self.capacity = cap
    def resize_if_needed(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        new_cap = int(max(needed, self.capacity * 1.5 + 1024))
        self._idx.resize_index(new_cap)
        self.capacity = new_cap
    def add_items(self, data: np.ndarray, labels: np.ndarray) -> None:
        if data.size == 0 or labels.size == 0:
            return
        try:
            current = int(self._idx.get_current_count())
        except Exception:
            current = 0
        needed = current + int(data.shape[0])
        self.resize_if_needed(needed)
        self._idx.add_items(data, labels)
    def mark_deleted(self, label: int) -> None:
        try:
            self._idx.mark_deleted(int(label))
        except Exception:
            pass
    def knn_query(self, vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._idx.knn_query(vecs, k=k)
    def set_ef(self, efS: int) -> None:
        self._idx.set_ef(int(efS))

# ---------- Main ----------
class HTNRMemory:
    def __init__(self, embedder: "EmbedderProtocol", *,
                 external_rag: Optional[ExternalRAGProtocol] = None,
                 hyper: Optional[Hyper] = None,
                 numerics: Optional[Numerics] = None,
                 log_path: Optional[str] = None,
                 seed: Optional[int] = 42,
                 xopts: Optional[XOpts] = None):
        self.model = embedder
        self.dim = self.model.get_sentence_embedding_dimension()
        self.hyper = hyper or Hyper()
        self.num = numerics or Numerics()
        self.external_rag = external_rag
        self._x = xopts or XOpts()

        # profile defaults
        self._apply_profile_defaults()

        # Context
        H = self.hyper.K_contexts
        self.M_heads = np.zeros((H, self.dim), dtype=np.float32)
        self.M_weights = np.ones(H, dtype=np.float32) / H
        self.M_agg = np.zeros(self.dim, dtype=np.float32)
        self.M_long = np.zeros(self.dim, dtype=np.float32)

        # State
        self.nodes: Dict[int, HTNRNode] = {}
        self.leaves: List[int] = []
        self.next_id = 0
        self.step = 0
        self.current_chapter = 0
        self.last_chapter_step = -10**9
        self.smoothed_lambda = 0.05

        # Cache
        self._leaf_mat: Optional[np.ndarray] = None
        self._leaf_ids: Optional[List[int]] = None
        self._soa: Optional[Dict[str, np.ndarray]] = None
        self._soa_index: Dict[int, int] = {}

        # Pinned host holder
        our = allocate_pinned_host_tensor  # noqa
        self._leaf_mat_host_t = None  # torch pinned CPU

        # ANN
        self._ann_index: Optional[HNSWIndex] = None
        self._ann_dirty = True
        self._ann_added_since_build = 0
        self._ann_deleted_since_build = 0

        # Buffer / AutoCal / Chapter-PH
        self.buffer: Deque[BufferItem] = deque(maxlen=self.hyper.buffer_size)
        self.auto = AutoCal(
            cos_lower=self.num.cos_lower, cos_upper=self.num.cos_upper,
            ema_gamma_fast=self.num.ema_gamma_fast, ema_gamma_slow=self.num.ema_gamma_slow,
            max_rate_tau_w=self.num.max_rate_tau_w, max_rate_tau_R=self.num.max_rate_tau_R,
        )
        self.ph_chapter = PageHinkley(delta=0.0, lam=3.0, var_guard=True, var_warmup=64, min_std=0.05)

        # Log / lock / rng
        self.log = EventLog(log_path, enabled=False)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)

        # sentence embedding cache for MMR
        self._sent_cache = LRUCache(capacity=5000)
        # 캐시 태그(임베더 식별자)
        self._sent_cache_tag = getattr(self.model, "cache_tag", "EMB")

        # Torch / Device
        self._use_torch = bool(self.hyper.use_torch and _HAS_TORCH and not self.hyper.torch_force_cpu)
        if self._use_torch:
            if _TORCH_CUDA:
                self._device = torch.device("cuda")
                torch.set_float32_matmul_precision("high")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = None

        # OOC streaming cache state
        self._ooc = {
            "copy_stream": None, "comp_stream": None,
            "buf0": None, "buf1": None, "out_host": None,
            "chunk_rows": None, "buf_rows": 0, "tuned": False
        }

        self._roots: set[int] = set()
        self.stats = defaultdict(int)

        self._rng_local = np.random.default_rng(self._x.lsh_seed if self._x.lsh_seed is not None else 1337)
        self._lsh_H: Optional[np.ndarray] = None
        self._lsh_H_comp: Optional[np.ndarray] = None

        # 간섭 샘플 풀(ANN 미사용 시)
        self._pair_pool: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # dissolve 직후 병합 제외 스케줄
        self._merge_excl_until: Dict[int, int] = {}

        self._sanity_check_hyper()

    # ----- Profile defaults -----
    def _apply_profile_defaults(self) -> None:
        p = (self.hyper.profile or "QA").lower()
        if p == "qa":
            self.hyper.query_blend = "auto"
            self.hyper.opportunistic_merge = False
            self.hyper.merge_cross_doc = False
            self.hyper.mmr_topk_parent = max(4, self.hyper.mmr_topk_parent)
        elif p == "streaming":
            if self.hyper.query_blend == "auto":
                self.hyper.query_blend_fixed = 0.20
            self.hyper.opportunistic_merge = True
        if self.hyper.batch_freeze_enable:
            self.hyper.buffer_size = max(self.hyper.buffer_size, self.hyper.batch_freeze_size)

    def _sanity_check_hyper(self) -> None:
        h = self.hyper
        def _ensure(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)
        _ensure(1 <= h.K_contexts <= 8, "hyper.K_contexts must be within [1,8]")
        _ensure(8 <= h.buffer_size <= 4096, "hyper.buffer_size must be within [8,4096]")
        _ensure(16 <= h.max_leaves <= 4096, "hyper.max_leaves must be within [16,4096]")
        _ensure(0.0 <= h.shrink_lambda <= 0.5, "hyper.shrink_lambda must be within [0.0,0.5]")
        _ensure(h.ann_backend in {"auto", "hnsw", "exact", "none"}, "hyper.ann_backend invalid")
        _ensure(0.0 < h.rag_entropy_tau <= 5.0, "hyper.rag_entropy_tau must be within (0,5]")

    # ---------- GPU OOC helpers ----------
    def _choose_chunk_rows(self, D: int) -> int:
        # 처음엔 작게 시작 → 성공하면 지수 증가
        if self._ooc["chunk_rows"] is not None:
            return int(self._ooc["chunk_rows"])
        base = 4096
        if _TORCH_CUDA:
            try:
                free_b, _ = torch.cuda.mem_get_info()
                # 행당 float32 D bytes, 두 버퍼 + 결과/여유
                bytes_per_row = D * 4
                est = int((free_b * 0.35) // (bytes_per_row * 2 + 1))
                base = max(2048, min(est, 65536))
            except Exception:
                base = 32768
        self._ooc["chunk_rows"] = base
        return base

    def _ensure_ooc_buffers(self, rows: int, D: int):
        import torch as _torch
        # 필요시 재할당 (권장 개선: 버퍼 재사용)
        if self._ooc["buf0"] is None or self._ooc["buf_rows"] < rows:
            self._ooc["buf0"] = _torch.empty((rows, D), dtype=_torch.float32, device=self._device)
            self._ooc["buf1"] = _torch.empty((rows, D), dtype=_torch.float32, device=self._device)
            self._ooc["out_host"] = _torch.empty(0, dtype=_torch.float32, pin_memory=True)  # 크기는 실행 시 세팅
            self._ooc["buf_rows"] = rows
        if self._ooc["copy_stream"] is None:
            self._ooc["copy_stream"] = _torch.cuda.Stream(device=self._device)
        if self._ooc["comp_stream"] is None:
            self._ooc["comp_stream"] = _torch.cuda.Stream(device=self._device)

    def _chunked_gpu_matvec_cached(self, host_pinned_t, device_vec) -> np.ndarray:
        import torch as _torch
        assert host_pinned_t is not None and host_pinned_t.device.type == "cpu"
        assert device_vec is not None and device_vec.is_cuda
        L, D = host_pinned_t.shape
        if L == 0:
            return np.zeros((0,), dtype=np.float32)

        chunk = self._choose_chunk_rows(D)

        # 어댑티브 실행 루프: 실패 시 반감하며 재시도
        while True:
            try:
                self._ensure_ooc_buffers(chunk, D)
                copy_stream: torch.cuda.Stream = self._ooc["copy_stream"]
                comp_stream: torch.cuda.Stream = self._ooc["comp_stream"]
                buf0 = self._ooc["buf0"]; buf1 = self._ooc["buf1"]

                # out_host 크기 보장
                if self._ooc["out_host"].numel() != L:
                    self._ooc["out_host"] = _torch.empty(L, dtype=_torch.float32, pin_memory=True)
                out_host = self._ooc["out_host"]
                out_np = out_host.numpy()

                use_buf0 = True
                offset = 0

                with _torch.cuda.stream(copy_stream):
                    end = min(offset + chunk, L)
                    if end > offset:
                        (buf0 if use_buf0 else buf1)[:end - offset].copy_(host_pinned_t[offset:end], non_blocking=True)

                while offset < L:
                    with _torch.cuda.stream(comp_stream):
                        comp_stream.wait_stream(copy_stream)
                        cur_buf = buf0 if use_buf0 else buf1
                        cur_rows = min(chunk, L - offset)
                        if cur_rows <= 0:
                            break
                        result_gpu = _torch.matmul(cur_buf[:cur_rows], device_vec)
                        out_host[offset:offset + cur_rows].copy_(result_gpu, non_blocking=True)
                    next_offset = offset + cur_rows
                    if next_offset < L:
                        with _torch.cuda.stream(copy_stream):
                            nxt_end = min(next_offset + chunk, L)
                            nxt_buf = buf1 if use_buf0 else buf0
                            nxt_buf[:nxt_end - next_offset].copy_(host_pinned_t[next_offset:nxt_end], non_blocking=True)
                    use_buf0 = not use_buf0
                    offset = next_offset

                copy_stream.synchronize()
                comp_stream.synchronize()

                # 성공
                self._ooc["chunk_rows"] = chunk
                return out_np.astype(np.float32)
            except Exception:
                torch.cuda.synchronize()
                # 반감 재시도
                new_chunk = max(1024, chunk // 2)
                if new_chunk == chunk:
                    raise RuntimeError("OOC matvec failed")
                chunk = new_chunk

    # ---------- Cache & ANN ----------
    def _build_soa(self) -> None:
        if not self.leaves:
            self._soa = {
                "ids": np.zeros(0, dtype=np.int64),
                "V": np.zeros(0, dtype=np.float32),
                "E": np.zeros(0, dtype=np.float32),
                "R": np.zeros(0, dtype=np.float32),
            }
            self._soa_index.clear()
            return
        ids = np.array(self.leaves, dtype=np.int64)
        V = np.array([self.nodes[i].V for i in self.leaves], dtype=np.float32)
        E = np.array([self.nodes[i].E for i in self.leaves], dtype=np.float32)
        R = np.array([self.nodes[i].R for i in self.leaves], dtype=np.float32)
        self._soa = {"ids": ids, "V": V, "E": E, "R": R}
        self._soa_index = {nid: idx for idx, nid in enumerate(self.leaves)}

    def _flush_soa_fields(self, fields: Sequence[str]) -> None:
        if not self._soa or self._soa["ids"].size == 0:
            return
        ids = self._soa["ids"]
        for field_name in fields:
            arr = self._soa.get(field_name)
            if arr is None:
                continue
            for i, nid in enumerate(ids):
                if nid in self.nodes:
                    setattr(self.nodes[nid], field_name, float(arr[i]))

    def _refresh_leaf_cache(self, *, mark_ann_dirty: bool = False, reuse_pinned: bool = True) -> None:
        leaf_count = len(self.leaves)
        if self._leaf_mat_host_t is not None and ((not reuse_pinned) or leaf_count == 0):
            try:
                free_pinned_host_tensor(self._leaf_mat_host_t)
            except Exception:
                pass
            self._leaf_mat_host_t = None

        self._leaf_mat = None
        self._leaf_ids = None
        if mark_ann_dirty:
            self._ann_dirty = True
        self._soa = None
        self._soa_index.clear()
        if leaf_count == 0:
            self._build_soa()
            return

        embs = np.stack([self.nodes[i].emb for i in self.leaves], axis=0)
        normalized = _normalize(embs, self.num.eps_norm)

        if self._use_torch and _TORCH_CUDA:
            reused = False
            pinned = self._leaf_mat_host_t if reuse_pinned else None
            if pinned is not None and tuple(pinned.shape) == tuple(normalized.shape):
                try:
                    src = torch.from_numpy(np.ascontiguousarray(normalized))
                    pinned.copy_(src, non_blocking=False)
                    self._leaf_mat = pinned.numpy()
                    reused = True
                except Exception as e:
                    logger.warning(f"Pinned reuse failed; allocating new: {e}")
                    try:
                        free_pinned_host_tensor(pinned)
                    except Exception:
                        pass
                    self._leaf_mat_host_t = None
                    pinned = None

            if not reused:
                t = allocate_pinned_host_tensor(normalized.shape)
                if t is not None:
                    try:
                        src = torch.from_numpy(np.ascontiguousarray(normalized))
                        t.copy_(src, non_blocking=False)
                        self._leaf_mat_host_t = t
                        self._leaf_mat = t.numpy()
                        reused = True
                    except Exception as e:
                        logger.warning(f"Pinned host tensor path failed, fallback to NumPy: {e}")
                        try:
                            free_pinned_host_tensor(t)
                        except Exception:
                            pass
                        self._leaf_mat_host_t = None

            if not reused:
                self._leaf_mat = normalized
                self._leaf_mat_host_t = None
        else:
            self._leaf_mat = normalized
            self._leaf_mat_host_t = None

        self._leaf_ids = list(self.leaves)
        self._build_soa()

    def _homeostasis_vectorized(self) -> None:
        if not self._soa or self._soa["ids"].size < 4:
            return
        V = self._soa["V"].astype(np.float32, copy=False)
        E = self._soa["E"].astype(np.float32, copy=False)
        # 보호항 E는 감쇠에서만 쓰고, 홈오스타시스 자체에는 사용하지 않음
        V_int_all = np.maximum(0.0, V - E)
        mask = np.isfinite(V_int_all)
        if not np.any(mask):
            return
        V_int = V_int_all[mask].astype(np.float32, copy=False)
        if V_int.size < 4:
            return
        q1, q3 = np.percentile(V_int, [25, 75])
        iqr = max(1e-6, float(q3 - q1))
        lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        filled = np.where(mask, V_int_all, q1).astype(np.float32)
        W = np.clip(filled, lo, hi)
        mu = float(np.mean(W))
        sd = float(np.std(W) + 1e-6)
        z = (W - mu) / sd
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        newV = np.clip(mu + sd * sigmoid, 0.0, 2.0).astype(np.float32)
        self._soa["V"] = newV

    def _ensure_ann_index(self) -> None:
        if self.hyper.ann_backend == "none": self._ann_index = None; return
        if self.hyper.ann_backend == "exact": self._ann_index = None; return
        if self.hyper.ann_backend in ("auto", "hnsw") and not _HAS_HNSW:
            self._ann_index = None; return
        if len(self.leaves) < self.hyper.ann_min_leaves:
            self._ann_index = None; return
        if self._leaf_mat is None or self._leaf_ids is None:
            return

        # lazy 삭제/추가 누적 시 강제 리빌드
        need_rebuild = False
        current_size = len(self.leaves)
        if (self._ann_added_since_build + self._ann_deleted_since_build) > max(32, 0.3 * current_size):
            need_rebuild = True

        needed_capacity = int(len(self.leaves) * self.hyper.ann_capacity_slack + self.hyper.ann_capacity_extra)
        if (self._ann_index is None) or self._ann_dirty or need_rebuild:
            try:
                self._ann_index = HNSWIndex(
                    self.dim, space="cosine", M=self.hyper.ann_M, efC=self.hyper.ann_efC,
                    efS=self.hyper.ann_efS, capacity=needed_capacity,
                )
                labels = np.array(self._leaf_ids, dtype=np.int64)
                self._ann_index.add_items(self._leaf_mat, labels)
                self._ann_index.set_ef(self.hyper.ann_efS)
                self._ann_dirty = False
                self._ann_added_since_build = 0
                self._ann_deleted_since_build = 0
                self.log.write({"t": "ann_build", "L": len(self.leaves)})
            except Exception as e:
                logger.warning(f"HNSW build failed: {e}")
                self._ann_index = None; self._ann_dirty = True
        else:
            try: self._ann_index.resize_if_needed(needed_capacity)
            except Exception: pass

    def _ann_mark_dirty_for_ids(self, added: List[int], removed: List[int]) -> None:
        if not added and not removed: return
        if self.hyper.ann_backend in ("none", "exact"):
            self._ann_dirty = True; self._ann_index = None; return
        if not _HAS_HNSW or len(self.leaves) < self.hyper.ann_min_leaves:
            self._ann_dirty = True; return
        with self._lock:
            if self._ann_index is None:
                self._ann_dirty = True; return
            try:
                for nid in removed:
                    try: self._ann_index.mark_deleted(int(nid))
                    except Exception:
                        self._ann_dirty = True; return
                if added:
                    labels = np.array(added, dtype=np.int64)
                    embs = np.stack([self.nodes[n].emb for n in added], axis=0)
                    embs = _normalize(embs, self.num.eps_norm)
                    self._ann_index.add_items(embs, labels)
                    self._ann_added_since_build += len(added)
                self._ann_deleted_since_build += len(removed)
                self._ann_dirty = False
            except Exception:
                self._ann_dirty = True

    # ---------- LSH planes (optional) ----------
    def _ensure_lsh_planes(self) -> None:
        if self._lsh_H is None and self._x.lsh_density_enable and self._x.lsh_bits > 0:
            self._lsh_H = self._rng_local.normal(size=(self.dim, int(self._x.lsh_bits))).astype(np.float32)
        if self._lsh_H_comp is None and self._x.proactive_compress_enable and self._x.proactive_compress_method == "lsh":
            b = max(4, int(self._x.proactive_lsh_bits))
            self._lsh_H_comp = self._rng_local.normal(size=(self.dim, b)).astype(np.float32)

    # ---------- Context update ----------
    def _update_contexts(self, emb: np.ndarray) -> Tuple[float, float]:
        sims = self.M_heads @ emb
        temp = 1.0 + 5.0 * self.smoothed_lambda
        ex = np.exp(sims / max(1e-6, temp))
        gates = ex / (np.sum(ex) + 1e-8)
        PE = 1.0 - _fast_dot(emb, self.M_agg)
        alpha = self.auto.update_alpha(PE)
        self.M_heads = alpha * self.M_heads + (1 - alpha) * (gates[:, None] * emb)
        self.M_heads = _normalize(self.M_heads, self.num.eps_norm)
        prev = self.M_agg.copy()
        self.M_weights = 0.95 * self.M_weights + 0.05 * gates
        self.M_weights /= (np.sum(self.M_weights) + 1e-8)
        self.M_agg = _normalize(np.sum(self.M_heads * self.M_weights[:, None], axis=0), self.num.eps_norm)
        self.M_long = _normalize(0.99 * self.M_long + 0.01 * self.M_agg, self.num.eps_norm)
        D_birth = 1.0 - _fast_dot(self.M_agg, prev)
        lam = max(0.005, min(0.5 * D_birth, 0.3))
        self.smoothed_lambda = 0.1 * lam + 0.9 * self.smoothed_lambda
        return PE, D_birth

    # ---------- Sent embedding helpers ----------
    def _encode_cached_sentences(self, sents: List[str]) -> np.ndarray:
        if not sents:
            return np.zeros((0, self.dim), dtype=np.float32)
        misses = []
        for s in sents:
            if self._sent_cache.get(f"D:{self._sent_cache_tag}:{s}") is None:
                misses.append(s)
        if misses:
            if hasattr(self.model, "encode_document"):
                embs = self.model.encode_document(misses)
            else:
                embs = self.model.encode(misses)
            for s, e in zip(misses, embs):
                self._sent_cache.put(f"D:{self._sent_cache_tag}:{s}", e)
        out = []
        for s in sents:
            v = self._sent_cache.get(f"D:{self._sent_cache_tag}:{s}")
            if v is None:
                v = np.zeros(self.dim, dtype=np.float32)
            out.append(v)
        return np.stack(out, axis=0)

    def _ensure_node_sent_embs(self, node: Optional[HTNRNode]) -> None:
        if node is None:
            return
        if not node.sentences:
            node.sentences = sent_tokenize(node.content or "")
        if not node.sentences:
            node.sent_embs = None
            return
        needs_refresh = (
            node.sent_embs is None or node.sent_embs.shape[0] != len(node.sentences)
        )
        if not needs_refresh:
            return
        try:
            embs = self._encode_cached_sentences(node.sentences)
            node.sent_embs = _normalize(np.asarray(embs, dtype=np.float32), 1e-8)
        except Exception:
            node.sent_embs = None

    # ---------- Add / Buffer ----------
    def process_and_add(self, text: str, *, source: str = "Unknown",
                        chapter: Optional[int] = None, salience_boost: float = 0.0) -> None:
        if not text:
            return
        if hasattr(self.model, "encode_document"):
            vec = self.model.encode_document([text])[0]
        else:
            vec = self.model.encode([text])[0]
        vec = _normalize(vec, self.num.eps_norm)
        item = BufferItem(text, vec, max(0.0, float(salience_boost)), source, chapter)
        with self._lock:
            self.buffer.append(item)
            self._replay_if_ready()

    def flush_buffer(self) -> None:
        with self._lock:
            self._replay(force=True)

    def _replay_if_ready(self) -> None:
        if self.hyper.batch_freeze_enable:
            if len(self.buffer) >= min(self.buffer.maxlen, self.hyper.batch_freeze_size):
                self._replay(force=False)
        else:
            if len(self.buffer) >= self.buffer.maxlen or self.auto.ph_alpha.stable_run > 20:
                self._replay(force=False)

    def _dynamic_batch_size(self) -> int:
        if not self._x.dynamic_batch_enable:
            return int(self.hyper.batch_freeze_size)
        sr = max(1, int(self.auto.ph_alpha.stable_run))
        t = float(np.tanh(sr / 50.0))
        B = int(self._x.dynamic_batch_min + t * (self._x.dynamic_batch_max - self._x.dynamic_batch_min))
        return max(self._x.dynamic_batch_min, min(self._x.dynamic_batch_max, B))

    def _compress_batch(self, batch: List[BufferItem]) -> List[BufferItem]:
        if not self._x.proactive_compress_enable or len(batch) < 4:
            return batch
        E = np.stack([it.emb for it in batch], axis=0)
        tau = float(np.clip(self.auto.tau_w * self._x.proactive_compress_tau_scale, 0.50, 0.98))

        def _mk_item(idxs: List[int]) -> BufferItem:
            sub = [batch[i] for i in idxs]
            emb = _normalize(np.mean(np.stack([s.emb for s in sub], axis=0), axis=0), 1e-8)
            content = " / ".join((s.content or "")[:240] for s in sub[:3])
            E_init = float(np.mean([s.E_init for s in sub]))
            src = sub[0].source if all((s.source == sub[0].source) for s in sub) else "Compressed"
            ch = sub[0].chapter_hint if all((s.chapter_hint == sub[0].chapter_hint) for s in sub) else None
            return BufferItem(content=content, emb=emb, E_init=E_init, source=src, chapter_hint=ch)

        if self._x.proactive_compress_method == "lsh":
            self._ensure_lsh_planes()
            H = self._lsh_H_comp
            if H is None:
                return batch
            sig_bits = (E @ H >= 0.0).astype(np.uint8)
            weights = (1 << np.arange(sig_bits.shape[1], dtype=np.uint64))
            sig = (sig_bits.astype(np.uint64) @ weights)
            _, inv = np.unique(sig, return_inverse=True)
            out: List[BufferItem] = []
            for bucket_id in np.unique(inv):
                idxs = np.where(inv == bucket_id)[0].tolist()
                if len(idxs) == 1:
                    out.append(batch[idxs[0]]); continue
                centroid = _normalize(np.mean(E[idxs], axis=0), 1e-8)
                sims = (E[idxs] @ centroid).astype(np.float32)
                if float(np.mean(sims)) >= tau:
                    out.append(_mk_item(idxs))
                else:
                    for i in idxs:
                        out.append(batch[i])
            if len(out) > self._x.proactive_compress_max_clusters:
                out = out[: self._x.proactive_compress_max_clusters]
            return out

        # greedy
        centroids: List[np.ndarray] = []
        groups: List[List[int]] = []
        for i in range(E.shape[0]):
            e = E[i]
            if not centroids:
                centroids.append(e); groups.append([i]); continue
            C = np.stack(centroids, axis=0)
            sims = (C @ e).astype(np.float32)
            j = int(np.argmax(sims))
            if float(sims[j]) >= tau:
                groups[j].append(i)
                new_c = _normalize((C[j] * (len(groups[j]) - 1) + e) / len(groups[j]), 1e-8)
                centroids[j] = new_c
            else:
                centroids.append(e); groups.append([i])
        out = [_mk_item(g) if len(g) > 1 else batch[g[0]] for g in groups]
        if len(out) > self._x.proactive_compress_max_clusters:
            out = out[: self._x.proactive_compress_max_clusters]
        return out

    def _ingest_batch_freeze(self, batch: List[BufferItem]) -> None:
        if not batch:
            return
        added_ids: List[int] = []
        if self.step == 0 and float(np.linalg.norm(self.M_agg)) <= self.num.eps_norm:
            first = batch[0]
            self.M_heads[:] = first.emb
            self.M_agg = first.emb.copy()
            self.M_long = first.emb.copy()
            nid0 = self._add_node_from_item(first, PE=0.0, D_birth=0.0, chapter_hint=first.chapter_hint)
            added_ids.append(nid0)
            batch = batch[1:]
            if not batch:
                self._refresh_leaf_cache()
                self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
                self._maintenance()
                return

        M_heads_0 = self.M_heads.copy()
        M_weights_0 = self.M_weights.copy()
        M_agg_0 = self.M_agg.copy()

        E = np.stack([it.emb for it in batch], axis=0)

        sims_to_ctx = (E @ M_agg_0).astype(np.float32)
        sims_to_ctx = np.clip(sims_to_ctx, -1.0, 1.0)
        PE_vec = (1.0 - sims_to_ctx).astype(np.float32)
        PE_mean = float(np.mean(PE_vec)) if PE_vec.size else 0.0

        temp = 1.0 + 5.0 * self.smoothed_lambda
        sims = (M_heads_0 @ E.T) / max(1e-6, temp)
        ex = np.exp(sims - np.max(sims, axis=0, keepdims=True))
        gates = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-8)

        num = gates @ E
        den = np.sum(gates, axis=1, keepdims=True) + 1e-8
        avg_emb = num / den

        alpha_t = self.auto.update_alpha(PE_mean)
        M_heads_1 = alpha_t * M_heads_0 + (1 - alpha_t) * avg_emb
        M_heads_1 = _normalize(M_heads_1, self.num.eps_norm)

        gate_mean = np.mean(gates, axis=1).astype(np.float32)
        M_weights_1 = 0.95 * M_weights_0 + 0.05 * gate_mean
        M_weights_1 = M_weights_1 / (np.sum(M_weights_1) + 1e-8)

        M_agg_1 = _normalize(np.sum(M_heads_1 * M_weights_1[:, None], axis=0), self.num.eps_norm)
        self.M_long = _normalize(0.99 * self.M_long + 0.01 * M_agg_1, self.num.eps_norm)

        D_birth_batch = 1.0 - _fast_dot(M_agg_1, M_agg_0)
        lam = max(0.005, min(0.5 * D_birth_batch, 0.3))
        self.smoothed_lambda = 0.1 * lam + 0.9 * self.smoothed_lambda

        self.M_heads = M_heads_1
        self.M_weights = M_weights_1
        self.M_agg = M_agg_1

        for it, pe in zip(batch, PE_vec.tolist()):
            nid = self._add_node_from_item(it, PE=pe, D_birth=D_birth_batch, chapter_hint=it.chapter_hint)
            added_ids.append(nid)

        self._refresh_leaf_cache()
        self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
        self._maintenance()

    def _replay(self, *, force: bool) -> None:
        if not self.buffer:
            return
        processed = 0
        if self.hyper.batch_freeze_enable:
            while self.buffer and (processed < max(4, self.hyper.buffer_size) or force):
                B = min(len(self.buffer), self._dynamic_batch_size())
                batch = [self.buffer.popleft() for _ in range(B)]
                if self._x.proactive_compress_enable:
                    batch = self._compress_batch(batch)
                self._ingest_batch_freeze(batch)
                processed += B
        else:
            added_ids: List[int] = []
            max_per_tick = max(4, self.hyper.buffer_size)
            while self.buffer and processed < max_per_tick:
                item = self.buffer.popleft()
                if self.step == 0 and float(np.linalg.norm(self.M_agg)) <= self.num.eps_norm:
                    self.M_heads[:] = item.emb
                    self.M_agg = item.emb.copy()
                    self.M_long = item.emb.copy()
                    PE, D_birth = 0.0, 0.0
                else:
                    PE, D_birth = self._update_contexts(item.emb)
                nid = self._add_node_from_item(item, PE=PE, D_birth=D_birth, chapter_hint=item.chapter_hint)
                added_ids.append(nid)
                processed += 1

            self._refresh_leaf_cache()
            self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
            self._maintenance()

    def _add_node_from_item(self, item: BufferItem, *, PE: float, D_birth: float, chapter_hint: Optional[int]) -> int:
        sig = 0.4 * D_birth + 0.4 * PE + 0.2 * (1.0 - _fast_dot(item.emb, self.M_long))
        boundary = self.ph_chapter.update(sig)
        if boundary and (self.step - self.last_chapter_step >= self.hyper.homeo_interval) and (self.ph_chapter.stable_run == 0):
            self.current_chapter += 1
            self.last_chapter_step = self.step

        node_ch = self.current_chapter if chapter_hint is None else int(chapter_hint)
        R_init = _fast_dot(item.emb, self.M_agg)
        V_init = float(np.clip(R_init + item.E_init + 0.7 * PE, 0.0, 2.0))

        sents = sent_tokenize(item.content)
        sent_embs = None
        if sents:
            try:
                embs = self._encode_cached_sentences(sents)
                sent_embs = _normalize(np.asarray(embs, dtype=np.float32), 1e-8)
            except Exception:
                sent_embs = None

        nid = self.next_id; self.next_id += 1
        node = HTNRNode(
            id=nid, content=item.content, emb=item.emb, V=V_init, D=D_birth, S=1.0, R=R_init,
            E=item.E_init, Chi=0.0, parent=None, children=[], birth_step=self.step,
            chapter=node_ch, source=item.source, sentences=sents, sent_embs=sent_embs
        )
        self.nodes[nid] = node
        self.leaves.append(nid)
        self._root_add(nid)
        self.step += 1
        self.log.write({"t": "add", "id": nid, "V": V_init, "R": R_init, "E": item.E_init, "ch": node_ch})
        return nid

    # ---------- Merge ----------
    def _dendritic_gating(self, na: HTNRNode, nb: HTNRNode) -> np.ndarray:
        Ri = _fast_dot(na.emb, self.M_agg); Rj = _fast_dot(nb.emb, self.M_agg)
        diff = 3.0 * (Ri - Rj)
        gate_i = 1.0 / (1.0 + math.exp(-diff))
        gate_j = 1.0 - gate_i
        gated = gate_i * na.emb + gate_j * nb.emb
        s = max(1e-6, na.V + nb.V)
        vw = (na.V / s) * na.emb + (nb.V / s) * nb.emb
        new = 0.5 * gated + 0.5 * vw
        new = (1 - self.hyper.shrink_lambda) * new + self.hyper.shrink_lambda * self.M_agg
        return _normalize(new, self.num.eps_norm)

    def _try_build_parent(self, ida: int, idb: int) -> bool:
        na, nb = self.nodes[ida], self.nodes[idb]
        self._ensure_node_sent_embs(na)
        self._ensure_node_sent_embs(nb)
        def _extract_doc_id(source: Any) -> Optional[str]:
            if isinstance(source, str) and source.startswith("doc:"):
                return source.split(":", 1)[1]
            return None
        doc_a = _extract_doc_id(na.source)
        doc_b = _extract_doc_id(nb.source)
        if not self.hyper.merge_cross_doc and doc_a and doc_b and doc_a != doc_b:
            return False
        merged_emb = self._dendritic_gating(na, nb)
        R_align = _fast_dot(merged_emb, self.M_agg)
        if R_align < self.auto.tau_R:
            return False

        sents_a = na.sentences or []
        sents_b = nb.sentences or []
        sents = sents_a + sents_b
        precomputed_embs = None
        if sents:
            is_a_valid = bool(sents_a) and na.sent_embs is not None and na.sent_embs.shape[0] == len(sents_a)
            is_b_valid = bool(sents_b) and nb.sent_embs is not None and nb.sent_embs.shape[0] == len(sents_b)
            try:
                if is_a_valid and is_b_valid:
                    precomputed_embs = np.vstack([na.sent_embs, nb.sent_embs])
                elif is_a_valid and not sents_b:
                    precomputed_embs = na.sent_embs
                elif is_b_valid and not sents_a:
                    precomputed_embs = nb.sent_embs
            except Exception as e:
                logger.warning(f"Error combining embeddings during merge: {e}")
                precomputed_embs = None

        if self.hyper.mmr_topk_parent_dynamic:
            tk = int(np.interp(len(sents), [0, 8, 20, 50], [2, 3, 5, 8]))
            tk = max(2, min(tk, max(2, self.hyper.mmr_topk_parent)))
        else:
            tk = max(2, self.hyper.mmr_topk_parent)

        try:
            lines = mmr_select(merged_emb, sents, precomputed_embs=precomputed_embs,
                               encode_fn=self._encode_cached_sentences,
                               topk=tk, lambda_mmr=0.7, max_sentences=48)
        except Exception:
            lines = []
        parent_content = " / ".join(lines) if lines else ((na.content or "")[:80] + " / " + (nb.content or "")[:80])

        new_E = 0.5 * (na.E + nb.E)
        new_V = min(2.0, 0.35 * (na.V + nb.V) + 0.65 * (R_align + new_E))
        new_D = max(na.D, nb.D)
        new_S = 1.0 + math.log1p(len(na.children) + len(nb.children))
        new_Chi = 0.5 * (na.Chi + nb.Chi) + 0.5 * R_align

        pid = self.next_id; self.next_id += 1
        parent_source = f"doc:{doc_a}" if doc_a else (f"doc:{doc_b}" if doc_b else "Merge")
        parent_sents = sent_tokenize(parent_content)
        parent_sent_embs = None
        if parent_sents:
            try:
                parent_sent_embs = _normalize(self._encode_cached_sentences(parent_sents), 1e-8)
            except Exception:
                parent_sent_embs = None

        node = HTNRNode(
            id=pid, content=parent_content, emb=merged_emb, V=new_V, D=new_D, S=new_S, R=R_align,
            E=new_E, Chi=new_Chi, parent=None, children=[ida, idb], birth_step=self.step,
            chapter=max(na.chapter, nb.chapter), source=parent_source,
            sentences=parent_sents, sent_embs=parent_sent_embs
        )
        self.nodes[pid] = node
        self.nodes[ida].parent = pid
        self.nodes[idb].parent = pid
        removed = []
        if ida in self.leaves:
            self.leaves.remove(ida); removed.append(ida); self._root_remove(ida)
        if idb in self.leaves:
            self.leaves.remove(idb); removed.append(idb); self._root_remove(idb)
        self.leaves.append(pid)
        self._root_add(pid)
        self._ann_mark_dirty_for_ids(added=[pid], removed=removed)
        self.log.write({"t": "merge", "p": pid, "a": ida, "b": idb, "R": R_align, "V": new_V})
        return True

    def _eligible_leaves(self) -> List[int]:
        L = len(self.leaves)
        if L <= self.hyper.anchor_recent:
            return list(self.leaves)[:-1] if L > 1 else list(self.leaves)
        # 방금 dissolve된 노드 제외
        now = self.step
        cand = [nid for nid in self.leaves[:-self.hyper.anchor_recent]
                if (nid not in self._merge_excl_until) or (self._merge_excl_until[nid] < now)]
        return cand

    def _merge_once(self) -> bool:
        cand_ids = self._eligible_leaves()
        if len(cand_ids) < 2 or self._leaf_mat is None:
            return False
        self._ensure_ann_index()
        pairs: List[Tuple[float, int, int]] = []
        used_set = set()

        if self._ann_index is not None:
            try:
                idx_map = {nid: i for i, nid in enumerate(self.leaves)}
                rows = [idx_map[i] for i in cand_ids if i in idx_map]
                if rows:
                    Q = self._leaf_mat[rows]
                    K = self.hyper.knn_topk_pairs + 8
                    labels, dists = self._ann_index.knn_query(Q, k=min(K, len(self.leaves)))
                    for row, a in enumerate(cand_ids):
                        for lab, dist in zip(labels[row], dists[row]):
                            b = int(lab)
                            if a == b or (b not in cand_ids):
                                continue
                            key = (min(a, b), max(a, b))
                            if key in used_set:
                                continue
                            w = float(1.0 - float(dist))
                            if w >= self.auto.tau_w:
                                pairs.append((w, a, b)); used_set.add(key)
            except Exception as e:
                logger.warning(f"ANN merge candidate failed, will fallback to sampled exact: {e}")

        if not pairs:
            # 정확 경로: 메모리 예산 기반 동적 제한
            scored = [(max(0.0, self.nodes[i].V) * max(0.0, self.nodes[i].R), i) for i in cand_ids]
            scored.sort(key=lambda z: z[0], reverse=True)
            # S = (sub @ sub.T) 의 메모리 (float32) = m*m*4 bytes
            max_bytes = max(16 * 1024 * 1024, int(self.hyper.exact_max_sim_matrix_mb * (1024 ** 2)))
            max_m = int(max(64, math.sqrt(max_bytes / 4)))
            cap = min(self.hyper.cand_prefilter, max_m, len(scored))
            cand_ids = [i for _, i in scored[:cap]]
            idx_map = {nid: k for k, nid in enumerate(self.leaves)}
            rows = [idx_map[i] for i in cand_ids if i in idx_map]
            sub = np.ascontiguousarray(self._leaf_mat[rows])
            S = (sub @ sub.T).astype(np.float32)
            m = S.shape[0]
            k = min(self.hyper.knn_topk_pairs + 1, m)
            for i in range(m):
                idx = _topk_indices(S[i], k)
                for j in idx:
                    if j == i:
                        continue
                    a, b = cand_ids[min(i, j)], cand_ids[max(i, j)]
                    if a == b:
                        continue
                    key = (a, b)
                    if key in used_set:
                        continue
                    w = float(S[i, j])
                    if w >= self.auto.tau_w:
                        pairs.append((w, a, b))
                        used_set.add(key)

        if not pairs:
            return False

        scores = []
        for w, a, b in pairs:
            na, nb = self.nodes[a], self.nodes[b]
            w_sig = 1.0 / (1.0 + math.exp(-3.0 * w))
            den = 1e-3 + math.sqrt(max(0.0, na.S + nb.S))
            scores.append((w_sig / den, a, b))
        scores.sort(key=lambda z: -z[0])

        merged = False
        used_leaf: set[int] = set()
        took = 0
        for _, a, b in scores:
            if took >= self.hyper.max_merges_per_cycle:
                break
            if a in used_leaf or b in used_leaf:
                continue
            if self._try_build_parent(a, b):
                used_leaf.add(a); used_leaf.add(b)
                merged = True
                took += 1

        if merged:
            self._refresh_leaf_cache()
            self._ensure_ann_index()
        return merged

    # ---------- Interference via ANN ----------
    def _interference_ann_density(self) -> None:
        if not self.leaves or self._leaf_mat is None:
            return
        if not self.hyper.approx_interference_enable:
            return
        L = len(self.leaves)
        self._ensure_ann_index()
        k = max(8, int(self.hyper.approx_knn_k))
        sims_all = []

        if self._ann_index is not None:
            try:
                labels, dists = self._ann_index.knn_query(self._leaf_mat, k=min(k + 1, L))
                for r in range(L):
                    row_labs = labels[r].tolist()
                    row_dst = dists[r].tolist()
                    neigh = []
                    for lab, dist in zip(row_labs, row_dst):
                        nid = int(lab)
                        if nid == self.leaves[r]:
                            continue
                        neigh.append(float(1.0 - float(dist)))
                    if neigh:
                        sims_all.extend(neigh)
                sims_all = np.array(sims_all, dtype=np.float32)
            except Exception as e:
                logger.warning(f"ANN interference failed; fallback to sampled exact: {e}")
                sims_all = np.empty((0,), dtype=np.float32)
        else:
            sims_all = np.empty((0,), dtype=np.float32)

        if sims_all.size == 0:
            # 슬라이딩 샘플 풀 유지 (분산 완화)
            max_pairs = int(self.hyper.approx_sample_pairs_when_no_ann)
            total_pairs = L * (L - 1) // 2
            if total_pairs <= 0:
                return
            # 풀 초기화 또는 리사이즈
            if (self._pair_pool is None) or (self._pair_pool[0].size > total_pairs) or (self._pair_pool[0].size == 0):
                iu = np.triu_indices(L, k=1)
                m = iu[0].size
                take = min(max_pairs, m)
                sel = np.random.choice(m, size=take, replace=False)
                self._pair_pool = (iu[0][sel], iu[1][sel])
            else:
                # 일부(20%)만 새로 뽑아 교체
                i0, i1 = self._pair_pool
                replace = max(1, int(0.2 * i0.size))
                iu = np.triu_indices(L, k=1)
                m = iu[0].size
                sel = np.random.choice(m, size=replace, replace=False)
                i0[:replace] = iu[0][sel]
                i1[:replace] = iu[1][sel]
                self._pair_pool = (i0, i1)
            idx0, idx1 = self._pair_pool
            X = self._leaf_mat
            sims_all = np.sum(X[idx0] * X[idx1], axis=1).astype(np.float32)

        q = float(np.quantile(sims_all, self.hyper.interference_q)) if sims_all.size else 0.0
        strong_counts = np.zeros(L, dtype=np.int32)
        strong_avgs = np.zeros(L, dtype=np.float32)

        if self._ann_index is not None:
            try:
                labels, dists = self._ann_index.knn_query(self._leaf_mat, k=min(k + 1, L))
                for r in range(L):
                    self_id = self.leaves[r]
                    vals = []
                    for lab, dist in zip(labels[r], dists[r]):
                        nid = int(lab)
                        if nid == self_id:
                            continue
                        s = float(1.0 - float(dist))
                        if s >= q:
                            vals.append(s)
                    if vals:
                        strong_counts[r] = len(vals)
                        strong_avgs[r] = float(np.mean(vals))
            except Exception as e:
                logger.warning(f"ANN per-node neighbors failed: {e}")
        else:
            block = 2048
            X = self._leaf_mat
            for start in range(0, L, block):
                end = min(start + block, L)
                S = (X[start:end] @ X.T).astype(np.float32)
                for i in range(S.shape[0]):
                    row = S[i]
                    row[start + i] = -1.0
                    idx = _topk_indices(row, min(k, L - 1))
                    vals = [float(row[j]) for j in idx if row[j] >= q]
                    if vals:
                        strong_counts[start + i] = len(vals)
                        strong_avgs[start + i] = float(np.mean(vals))

        # 꼭: 함수명 오타 수정 — _build_oa() → _build_soa()
        if self._soa is None or self._soa["ids"].size != L:
            self._build_soa()

        if self._soa and self._soa["ids"].size == L:
            E = self._soa["E"]
            density = 0.5 * np.log1p(np.maximum(0.0, strong_counts.astype(np.float32))) + 0.5 * strong_avgs
            protection = np.tanh(E)
            dec = np.maximum(0.0, 0.05 * density - 0.01 * protection).astype(np.float32)
            if dec.size:
                self._soa["V"] = np.maximum(0.0, self._soa["V"] - dec)

        self.auto.add_tail_samples(sims_all)

    # ---------- Optional: LSH density ----------
    def _ensure_soa(self):
        if self._soa is None or self._soa.get("ids", np.zeros(0)).size != len(self.leaves):
            self._build_soa()

    def _lsh_density_decay(self) -> None:
        if not (self._x.lsh_density_enable and self._leaf_mat is not None):
            return
        if self.hyper.approx_interference_enable:
            return
        L = len(self.leaves)
        if L < 8:
            return
        self._ensure_lsh_planes()
        H = self._lsh_H
        if H is None:
            return
        proj = self._leaf_mat @ H
        bits = (proj >= 0.0).astype(np.uint8)
        weights = (1 << np.arange(bits.shape[1], dtype=np.uint64))
        sig = (bits.astype(np.uint64) @ weights)
        uniq, inv, counts = np.unique(sig, return_inverse=True, return_counts=True)
        bucket_size = counts[inv].astype(np.float32)
        expected = max(1.0, float(L) / float(2 ** bits.shape[1]))
        density = np.log1p(np.maximum(0.0, bucket_size - expected))
        self._ensure_soa()
        if self._soa and self._soa["ids"].size == L:
            E = self._soa["E"]
            dec = np.maximum(0.0, 0.03 * density.astype(np.float32) - 0.01 * np.tanh(E)).astype(np.float32)
            if dec.size:
                self._soa["V"] = np.maximum(0.0, self._soa["V"] - dec)

    # ---------- Maintenance ----------
    def _maintenance(self) -> None:
        if not self.leaves:
            return
        self._refresh_leaf_cache()

        if self.step % self.hyper.homeo_interval == 0:
            self._homeostasis_vectorized()

        self._interference_ann_density()
        self._lsh_density_decay()

        self._flush_soa_fields(["V"])

        occupancy = float(len(self.leaves) / max(1, self.hyper.max_leaves))
        self.auto.update_thresholds(rl=np.clip(occupancy, 0.0, 2.0))

        merges = 0
        while len(self.leaves) > self.hyper.max_leaves and merges < self.hyper.max_merges_per_cycle:
            if not self._merge_once():
                break
            merges += 1

        self._phase_reorg()

        if self.hyper.opportunistic_merge and len(self.leaves) >= self.hyper.anchor_recent + 2:
            self._merge_once()

        parents_in_leaves = sum(1 for nid in self.leaves if self.nodes[nid].children)
        self.log.write({"t": "metrics", "leaves": len(self.leaves),
                        "parents_in_leaves": parents_in_leaves,
                        "tau_w": self.auto.tau_w, "tau_R": self.auto.tau_R})

    def _phase_reorg(self) -> None:
        if not self.leaves:
            return
        dissolved = 0
        budget = self.hyper.max_phaseR_per_cycle
        removed_ids: List[int] = []
        added_ids: List[int] = []
        for nid in list(self.leaves):
            if dissolved >= budget:
                break
            node = self.nodes.get(nid)
            if not node or not node.children:
                continue
            child_scores = [_fast_dot(self.nodes[c].emb, self.M_agg) for c in node.children]
            if not child_scores:
                continue
            best_child = max(child_scores)
            var = float(np.var(child_scores))
            margin = 0.05 + 0.10 * min(1.0, math.sqrt(var))
            if (node.R + 1e-6) < (best_child - margin) and (self.step - node.birth_step) > 20:
                # 부모 해체 → 자식 복귀(한 사이클 병합 제외)
                for c in node.children:
                    self.nodes[c].parent = None
                    self.nodes[c].V = max(0.0, self.nodes[c].V * 0.97)
                    self._merge_excl_until[c] = self.step + 1  # 한 사이클 제외
                    if c not in self.leaves:
                        self.leaves.append(c)
                        added_ids.append(c)
                    self._root_add(c)
                if nid in self.leaves:
                    self.leaves.remove(nid)
                removed_ids.append(nid)
                self._root_remove(nid)
                self.nodes.pop(nid, None)
                dissolved += 1

        if dissolved > 0:
            self._ann_mark_dirty_for_ids(added=added_ids, removed=removed_ids)
            self.log.write({"t": "phaseR", "count": dissolved})
            while len(self.leaves) > self.hyper.max_leaves and self._merge_once():
                pass
            self._refresh_leaf_cache()
            self._ensure_ann_index()

    # ---------- Retrieval (+ RAG) ----------
    def _uncertainty(self, scores: np.ndarray, k: int) -> Tuple[float, float]:
        if scores.size == 0:
            return float("inf"), 0.0
        idx = _topk_indices(scores, min(k, scores.size))
        s = scores[idx]
        tau = self.hyper.rag_entropy_tau
        ex = np.exp(s / max(1e-6, tau))
        Z = float(np.sum(ex))
        if not np.isfinite(Z) or Z <= 1e-12:
            return 0.0, float(np.max(s) if s.size else 0.0)
        p = ex / (Z + 1e-8)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        return ent, float(np.max(s))

    def retrieve_for_query(self, query: str, *, K_cap: int = 8, mutate: bool = False,
                           flush: bool = False, return_meta: bool = False, use_rag: bool = True) -> List[Any]:
        if flush:
            self.flush_buffer()

        with self._lock:
            leaf_mat = None if self._leaf_mat is None else self._leaf_mat.copy()
            leaf_ids = None if self._leaf_ids is None else list(self._leaf_ids)
            M_agg = self.M_agg.copy()

        if leaf_mat is None or not leaf_ids:
            return self._do_rag(query, K_cap, return_meta) if (use_rag and self.external_rag) else []

        # ---- 쿼리 임베딩: encode_query 우선 ----
        if hasattr(self.model, "encode_query"):
            q = self.model.encode_query([query])[0]
        else:
            q = self.model.encode([query])[0]
        q = _normalize(q, self.num.eps_norm)

        # Query blend (auto: 역관계 — cos 높을수록 블렌딩 적게)
        # vMF posterior mean direction ≈ Norm(κ_q·q + κ_m·M_agg)  ← 이론적 정당화(κ는 농도; 구현은 w로 근사)
        comp = q
        mode = (self.hyper.query_blend or "none").lower()
        if mode != "none":
            cos_qm = _fast_dot(q, M_agg)
            if mode == "fixed":
                w = float(np.clip(self.hyper.query_blend_fixed, 0.0, 0.95))
            else:
                lo, hi = self.hyper.query_blend_cos_low, self.hyper.query_blend_cos_high
                t = (cos_qm - lo) / max(1e-6, (hi - lo))
                t = float(np.clip(t, 0.0, 1.0))
                w = (1.0 - t) * float(np.clip(self.hyper.query_blend_fixed, 0.0, 0.95))
            comp = _normalize((1.0 - w) * q + w * M_agg, self.num.eps_norm)

        # score computation
        if self._use_torch and self._device and self._device.type == "cuda":
            comp_t = torch.from_numpy(np.ascontiguousarray(comp))
            comp_t = comp_t.pin_memory().to(self._device, non_blocking=True)
            if self._leaf_mat_host_t is not None:
                try:
                    scores = self._chunked_gpu_matvec_cached(self._leaf_mat_host_t, comp_t)
                except Exception as e:
                    logger.warning(f"OOC streaming failed, fallback to CPU: {e}")
                    self.stats['ooc_fallback'] += 1
                    scores = (np.ascontiguousarray(leaf_mat) @ np.ascontiguousarray(comp)).astype(np.float32)
            else:
                scores = (np.ascontiguousarray(leaf_mat) @ np.ascontiguousarray(comp)).astype(np.float32)
        else:
            scores = (np.ascontiguousarray(leaf_mat) @ np.ascontiguousarray(comp)).astype(np.float32)

        k = min(K_cap, len(scores))
        idx = _topk_indices(scores, k)
        chosen = [(leaf_ids[ii], float(scores[ii])) for ii in idx]

        outs = []
        query_lower = (query or "").lower()
        needs_detail = any(token in query_lower for token in ("why", "how", "explain", "because", "reason", "왜", "어떻게"))
        for nid, base_score in chosen:
            node = self.nodes.get(nid)
            if not node:
                continue
            # 문장 임베딩 사전 계산/재사용
            self._ensure_node_sent_embs(node)
            sents = node.sentences
            text = node.content or ""
            if sents and len(sents) > 4:
                try:
                    topk_sents = 6 if needs_detail else 4
                    precomputed = None
                    if node.sent_embs is not None and node.sent_embs.shape[0] == len(sents):
                        precomputed = node.sent_embs
                    selected = mmr_select(
                        comp, sents, precomputed_embs=precomputed,
                        encode_fn=self._encode_cached_sentences,
                        topk=topk_sents, lambda_mmr=0.75, max_sentences=48,
                    )
                    if selected:
                        text = " ".join(selected)
                except Exception:
                    pass
            vr = max(-1.0, min(1.0, node.V * node.R / 2.0))
            final_score = base_score * (1.0 + 0.25 * vr)
            raw_source = node.source if isinstance(node.source, str) else ""
            doc_id: Optional[str] = None
            chunk_anchor: Optional[str] = None
            if raw_source.startswith("doc:"):
                tail = raw_source.split(":", 1)[1]
                if tail:
                    chunk_anchor = tail
                    doc_id = tail.split("#", 1)[0]
            meta = {"node_id": nid, "tier": "L1", "chapter": node.chapter}
            if doc_id:
                meta["doc_id"] = doc_id
            if chunk_anchor:
                meta["chunk_id"] = chunk_anchor
            outs.append({
                "content": text, "score": final_score,
                "node_id": nid, "chapter": node.chapter,
                "source": node.source, "tier": "L1",
                "doc_id": doc_id, "chunk_id": chunk_anchor,
                "meta": meta,
            })

        outs.sort(key=lambda x: x["score"], reverse=True)

        if use_rag and self.external_rag:
            ent, top1 = self._uncertainty(scores, k=self.hyper.rag_topk)
            if (top1 < self.hyper.rag_top1_thr) or (ent > self.hyper.rag_entropy_thr):
                rag = self._do_rag(query, K_cap, return_meta=True)
                seen = set()
                final = []
                for r in outs + rag:
                    h = _stable_hash(r["content"][:200])
                    if h in seen:
                        continue
                    seen.add(h)
                    final.append(r)
                    if len(final) >= K_cap:
                        break
                if mutate:
                    with self._lock:
                        for r in outs:
                            nid = r["node_id"]
                            if nid in self.nodes:
                                self.nodes[nid].V = min(2.0, self.nodes[nid].V + 0.05 * float(r["score"]))
                return final if return_meta else [r["content"] for r in final]

        if mutate:
            with self._lock:
                for r in outs:
                    nid = r["node_id"]
                    if nid in self.nodes:
                        self.nodes[nid].V = min(2.0, self.nodes[nid].V + 0.05 * float(r["score"]))
        return outs if return_meta else [r["content"] for r in outs]

    def _do_rag(self, query: str, K_cap: int, return_meta: bool) -> List[Any]:
        try:
            rag_res = self.external_rag.search(query, top_k=min(self.hyper.rag_topk, K_cap))
        except Exception as e:
            logger.warning(f"RAG error: {e}")
            rag_res = []
        rag = [{
            "content": r.content, "score": float(r.score),
            "node_id": -1, "chapter": -1, "source": r.metadata.get("source", "RAG"),
            "url": r.metadata.get("url"), "tier": "L2"
        } for r in rag_res]
        return rag if return_meta else [r["content"] for r in rag]

    # ---------- Snapshot ----------
    def snapshot_save(self, path: str) -> None:
        with self._lock:
            data: Dict[str, Any] = {
                "version": "v56-ann+",
                "dim": self.dim, "step": self.step, "next_id": self.next_id,
                "ctx": {
                    "M_heads": self.M_heads.tolist(), "M_weights": self.M_weights.tolist(),
                    "M_agg": self.M_agg.tolist(), "M_long": self.M_long.tolist(),
                    "smoothed_lambda": self.smoothed_lambda,
                    "current_chapter": self.current_chapter, "last_chapter_step": self.last_chapter_step,
                },
                "auto": {
                    "tau_w": self.auto.tau_w, "tau_R": self.auto.tau_R, "alpha_t": self.auto.alpha_t,
                    "tail_buf": list(self.auto.tail_buf), "ph_alpha_sr": self.auto.ph_alpha.stable_run,
                },
                "hyper": asdict(self.hyper), "numerics": asdict(self.num),
                "nodes": [],
                "leaves": list(self.leaves),
            }
            use_f16 = bool(self.hyper.snapshot_float16)
            for n in self.nodes.values():
                emb = n.emb.astype(np.float16 if use_f16 else np.float32).tolist()
                row = {
                    "id": n.id, "content": n.content, "emb": emb,
                    "V": n.V, "D": n.D, "S": n.S, "R": n.R, "E": n.E, "Chi": n.Chi,
                    "parent": n.parent, "children": n.children, "birth_step": n.birth_step,
                    "chapter": n.chapter, "source": n.source, "sentences": n.sentences,
                }
                if self.hyper.snapshot_store_sent_embs and n.sent_embs is not None:
                    row["sent_embs"] = n.sent_embs.astype(np.float16 if use_f16 else np.float32).tolist()
                data["nodes"].append(row)

            parent = os.path.dirname(os.path.abspath(path))
            if parent and parent != ".":
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    @classmethod
    def snapshot_load(cls, embedder: "EmbedderProtocol", path: str,
                      external_rag: Optional[ExternalRAGProtocol] = None,
                      xopts: Optional[XOpts] = None) -> "HTNRMemory":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dim = embedder.get_sentence_embedding_dimension()
        if dim != data.get("dim"):
            raise ValueError(f"Embedder dim {dim} != snapshot dim {data.get('dim')}")
        inst = cls(embedder, external_rag=external_rag,
                   hyper=Hyper(**data.get("hyper", {})),
                   numerics=Numerics(**data.get("numerics", {})),
                   xopts=xopts)
        inst.step = int(data.get("step", 0))
        inst.next_id = int(data.get("next_id", 0))
        ctx = data.get("ctx", {})
        H = inst.hyper.K_contexts

        def _fit_heads(value: Any) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 2:
                arr = arr.reshape(-1, dim) if arr.size else np.zeros((0, dim), dtype=np.float32)
            out = np.zeros((H, dim), dtype=np.float32)
            h = min(H, arr.shape[0]); d = min(dim, arr.shape[1]) if arr.ndim == 2 else 0
            if h and d:
                out[:h, :d] = arr[:h, :d]
            return out

        def _fit_vec(value: Any) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float32)
            out = np.zeros(dim, dtype=np.float32)
            length = min(dim, arr.size)
            if length:
                out[:length] = arr.flat[:length]
            return out

        inst.M_heads = _fit_heads(ctx.get("M_heads", np.zeros((H, dim), dtype=np.float32)))
        weights = np.asarray(ctx.get("M_weights", np.ones(H, dtype=np.float32) / max(1, H)), dtype=np.float32)
        if weights.size < H:
            weights = np.pad(weights, (0, H - weights.size), constant_values=0.0)
        else:
            weights = weights[:H]
        if not np.any(np.isfinite(weights)) or float(np.sum(np.abs(weights))) <= 1e-8:
            weights = np.ones(H, dtype=np.float32)
        inst.M_weights = weights.astype(np.float32)
        inst.M_weights = inst.M_weights / (np.sum(inst.M_weights) + 1e-8)
        inst.M_agg = _fit_vec(ctx.get("M_agg", np.zeros(dim, dtype=np.float32)))
        inst.M_long = _fit_vec(ctx.get("M_long", np.zeros(dim, dtype=np.float32)))
        inst.smoothed_lambda = float(ctx.get("smoothed_lambda", 0.05))
        inst.current_chapter = int(ctx.get("current_chapter", 0))
        inst.last_chapter_step = int(ctx.get("last_chapter_step", -10**9))
        auto = data.get("auto", {})
        inst.auto.tau_w = float(auto.get("tau_w", 0.2))
        inst.auto.tau_R = float(auto.get("tau_R", math.sqrt((1.0 + inst.auto.tau_w) / 2.0)))
        inst.auto.alpha_t = float(auto.get("alpha_t", 0.9))
        inst.auto.tail_buf.clear()
        for v in auto.get("tail_buf", []):
            inst.auto.tail_buf.append(float(v))
        inst.nodes.clear()
        inst.leaves.clear()
        for row in data.get("nodes", []):
            emb_arr = np.array(row["emb"], dtype=np.float32)  # float16 저장 → float32 복원
            node = HTNRNode(
                id=int(row["id"]), content=row.get("content", ""),
                emb=emb_arr,
                V=float(row["V"]), D=float(row["D"]), S=float(row["S"]),
                R=float(row["R"]), E=float(row["E"]), Chi=float(row["Chi"]),
                parent=(int(row["parent"]) if row["parent"] is not None else None),
                children=list(row.get("children", [])),
                birth_step=int(row.get("birth_step", 0)),
                chapter=int(row.get("chapter", 0)),
                source=row.get("source", "Unknown"),
                sentences=list(row.get("sentences", [])),
                sent_embs=(np.array(row["sent_embs"], dtype=np.float32)
                           if "sent_embs" in row else None),
            )
            if node.sentences and node.sent_embs is None:
                try:
                    embs = inst._encode_cached_sentences(node.sentences)
                    node.sent_embs = _normalize(np.asarray(embs, dtype=np.float32), 1e-8)
                except Exception:
                    node.sent_embs = None
            inst.nodes[node.id] = node
        inst.leaves = [int(x) for x in data.get("leaves", [])]
        for nid in inst.leaves:
            if inst.nodes.get(nid, None) and inst.nodes[nid].parent is None:
                inst._root_add(nid)
        inst._refresh_leaf_cache()
        inst._ensure_ann_index()
        logger.info(f"Loaded snapshot: {path} (nodes={len(inst.nodes)}, leaves={len(inst.leaves)})")
        return inst

    # ---------- Root cache helpers ----------
    def _root_add(self, nid: int) -> None:
        self._roots.add(nid)
    def _root_remove(self, nid: int) -> None:
        self._roots.discard(nid)

    # ---------- Close ----------
    def close(self) -> None:
        self.log.close()
