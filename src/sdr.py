
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
from .xopts import XOpts

class SDRIndexer:
    """Multi-hash SDR (Hamming) prefilter with optional uint64 packing + np.bit_count."""
    def __init__(self, mem, x: XOpts):
        self.mem = mem
        self.x = x
        self._planes: Optional[List[np.ndarray]] = None   # list of [D, bits] float32
        self._codes: Optional[List[np.ndarray]] = None    # per-hash: uint8 packed [L, B/8] or uint64 [L, W]
        self._bitorder = "little"
        self._bits_eff = int(x.sdr_bits)
        self._last_build_step = -10**9
        self._rows_built = -1

        # uint8 LUT fallback
        self._popcnt_u8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # ---------- planes ----------
    def _rng(self, seed_offset: int) -> np.random.Generator:
        seed = int(getattr(self.mem, "_seed", 42))
        return np.random.default_rng(np.uint64(abs(hash((seed, seed_offset, int(getattr(self.mem, "dim", 0)))))))

    def _ensure_planes(self, bits: int, hashes: int) -> List[np.ndarray]:
        dim = int(getattr(self.mem, "dim", 0) or (self.mem.M_agg.size if getattr(self.mem, "M_agg", None) is not None else 0))
        if dim <= 0:
            raise RuntimeError("Cannot infer embedding dimension for SDR planes.")
        if self._planes is None or len(self._planes) != hashes:
            self._planes = []
            for h in range(hashes):
                rng = self._rng(h)
                self._planes.append(rng.normal(size=(dim, bits)).astype(np.float32))
            return self._planes
        # dimension mismatch rebuild
        for i, H in enumerate(self._planes):
            if H.shape[0] != dim:
                rng = self._rng(i)
                self._planes[i] = rng.normal(size=(dim, bits)).astype(np.float32)
            elif H.shape[1] < bits:
                need = bits - H.shape[1]
                rng = self._rng(i + 11)
                extra = rng.normal(size=(dim, need)).astype(np.float32)
                self._planes[i] = np.concatenate([H, extra], axis=1)
            else:
                self._planes[i] = H[:, :bits]
        return self._planes

    # ---------- packing ----------
    def _pack_uint64(self, bits_bin: np.ndarray) -> np.ndarray:
        # bits_bin: [L, B] uint8 {0,1}; pack little-endian into words of 64 bits
        L, B = bits_bin.shape
        pad = (-B) % 64
        if pad:
            bits_bin = np.pad(bits_bin, ((0,0),(0,pad)), constant_values=0)
            B += pad
        # pack into bytes (little bit order), then view as uint64
        try:
            packed = np.packbits(bits_bin, axis=1, bitorder="little")  # [L, B/8] uint8
            self._bitorder = "little"
        except TypeError:
            packed = np.packbits(bits_bin, axis=1)  # default bitorder
            self._bitorder = "big"
        # ensure contiguous groups of 8 bytes per word
        W = B // 64
        packed = packed[:, :W*8]  # truncate to full words
        words = packed.view(np.uint64)
        return words  # [L, W]

    # ---------- build ----------
    def rebuild(self) -> None:
        if not self.x.sdr_filter_enable:
            self._codes = None; return
        leaf = getattr(self.mem, "_leaf_mat", None)
        if leaf is None or leaf.size == 0:
            self._codes = None; return
        bits = int(self.x.sdr_bits)
        hashes = max(1, int(self.x.sdr_multi_hash))
        planes = self._ensure_planes(bits, hashes)

        codes: List[np.ndarray] = []
        for h in range(hashes):
            proj = (leaf @ planes[h]).astype(np.float32)   # [L, bits]
            bits_bin = (proj >= 0.0).astype(np.uint8)
            if self.x.sdr_pack_uint64 and bits % 64 == 0:
                code = self._pack_uint64(bits_bin)         # [L, W] uint64
            else:
                try:
                    code = np.packbits(bits_bin, axis=1, bitorder="little")  # [L, ceil(bits/8)] uint8
                    self._bitorder = "little"
                except TypeError:
                    code = np.packbits(bits_bin, axis=1)   # [L, ...] uint8
                    self._bitorder = "big"
            codes.append(code)

        self._codes = codes
        self._bits_eff = bits
        self._last_build_step = int(getattr(self.mem, "step", 0))
        self._rows_built = leaf.shape[0]

    def maybe_rebuild(self) -> None:
        if not self.x.sdr_filter_enable: 
            return
        step = int(getattr(self.mem, "step", 0))
        leaf = getattr(self.mem, "_leaf_mat", None)
        need = (
            self._codes is None or
            (step - self._last_build_step) >= int(self.x.sdr_rebuild_every) or
            (leaf is not None and leaf.shape[0] != self._rows_built)
        )
        if need:
            self.rebuild()

    # ---------- filter ----------
    def _hamming_uint64(self, codes: np.ndarray, q_code: np.ndarray) -> np.ndarray:
        # codes: [L, W] uint64, q_code: [W] uint64
        xor = np.bitwise_xor(codes, q_code[None, :])
        try:
            # NumPy 1.24+
            ham = np.bit_count(xor).sum(axis=1).astype(np.int32)
        except Exception:
            # Fallback: view as uint8 and LUT
            u8 = xor.view(np.uint8)
            ham = self._popcnt_u8[u8].sum(axis=1).astype(np.int32)
        return ham

    def _hamming_u8(self, codes: np.ndarray, q_code: np.ndarray) -> np.ndarray:
        # codes: [L, B] uint8, q_code: [B] uint8
        xor = np.bitwise_xor(codes, q_code[None, :])
        ham = self._popcnt_u8[xor].sum(axis=1).astype(np.int32)
        return ham

    def filter_rows(self, comp: np.ndarray, desired_M: int) -> np.ndarray:
        """Return candidate row indices after multi-hash SDR prefilter."""
        self.maybe_rebuild()
        if self._codes is None or len(self._codes) == 0:
            # no filter
            leaf = getattr(self.mem, "_leaf_mat", None)
            L = 0 if leaf is None else leaf.shape[0]
            return np.arange(L, dtype=np.int64)

        leaf = getattr(self.mem, "_leaf_mat", None)
        L = leaf.shape[0] if leaf is not None else 0
        if L == 0:
            return np.zeros((0,), dtype=np.int64)

        # dynamic M
        M = int(desired_M)
        if self.x.sdr_dynamic_prefilter:
            dyn = int(max(self.x.sdr_min_prefilter, self.x.sdr_dynamic_alpha * np.sqrt(max(1.0, float(L)))))
            M = max(M, dyn)

        H = self._ensure_planes(self._bits_eff, len(self._codes))
        # query bit code per hash
        cand_sets: List[np.ndarray] = []
        per_hash = int(self.x.sdr_prefilter_per_hash or max(1, M // max(1, len(self._codes))))
        for h, code in enumerate(self._codes):
            q_proj = (comp @ H[h]).astype(np.float32)
            q_bits = (q_proj >= 0.0).astype(np.uint8)
            if self.x.sdr_pack_uint64 and self._bits_eff % 64 == 0 and code.dtype == np.uint64:
                q_code = self._pack_uint64(q_bits[None, :]).reshape(-1)  # [W]
                ham = self._hamming_uint64(code, q_code)
            else:
                try:
                    q_code = np.packbits(q_bits, axis=0, bitorder=self._bitorder)  # [B]
                except TypeError:
                    q_code = np.packbits(q_bits, axis=0)  # default
                ham = self._hamming_u8(code, q_code)
            # take top per_hash
            m = min(per_hash, L)
            idx = np.argpartition(ham, m - 1)[:m]
            cand_sets.append(idx.astype(np.int64))

        # union & cap
        if len(cand_sets) == 1:
            rows = cand_sets[0]
        else:
            rows = np.unique(np.concatenate(cand_sets, axis=0))
        cap = int(self.x.sdr_union_cap)
        if rows.size > cap:
            # keep ones with smallest ham using first hash as tiebreaker
            code0 = self._codes[0]
            if code0.dtype == np.uint64:
                q_proj = (comp @ H[0]).astype(np.float32)
                q_bits = (q_proj >= 0.0).astype(np.uint8)
                q_code = self._pack_uint64(q_bits[None, :]).reshape(-1)
                ham0 = self._hamming_uint64(code0[rows], q_code)
            else:
                try:
                    q_code = np.packbits((comp @ H[0] >= 0.0).astype(np.uint8), axis=0, bitorder=self._bitorder)
                except TypeError:
                    q_code = np.packbits((comp @ H[0] >= 0.0).astype(np.uint8), axis=0)
                ham0 = self._hamming_u8(code0[rows], q_code)
            take = np.argpartition(ham0, cap - 1)[:cap]
            rows = rows[take]
        return rows
