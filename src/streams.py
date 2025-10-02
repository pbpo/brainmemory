
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    _HAS_TORCH = True
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    _HAS_TORCH = False
    _TORCH_CUDA = False

from .xopts import XOpts

class GpuStreamer:
    """Out-of-core streaming matvec with pinned safeguards + cosine normalization."""
    def __init__(self, mem, x: XOpts):
        self.mem = mem
        self.x = x

    def _chunked(self, host_t, dev_vec, *, chunk_rows: Optional[int], cover_ratio: float, use_fp16: bool) -> "torch.Tensor":
        """Double-buffered H2D copy + matvec overlap. Returns torch.Tensor on CPU (pinned)."""
        assert _HAS_TORCH and _TORCH_CUDA
        device = getattr(self.mem, "_device", torch.device("cuda"))
        L, D = host_t.shape
        # chunk size
        if not chunk_rows or chunk_rows <= 0:
            try:
                free_b, _ = torch.cuda.mem_get_info(device)
            except Exception:
                free_b = 256 * 1024 * 1024
            bytes_per_row = D * (2 if use_fp16 else 4)
            rows = int((free_b * cover_ratio) // (bytes_per_row * 2 + 1))
            chunk_rows = max(1024, min(rows, 65536))
        # buffers/streams
        copy_stream = torch.cuda.Stream(device=device)
        comp_stream = torch.cuda.Stream(device=device)
        dtype = torch.float16 if use_fp16 else torch.float32
        buf0 = torch.empty((min(chunk_rows, L), D), dtype=dtype, device=device)
        buf1 = torch.empty_like(buf0)
        out_host = torch.empty(L, dtype=torch.float32, pin_memory=True)
        use0, off = True, 0

        with torch.cuda.stream(copy_stream):
            end = min(off + chunk_rows, L)
            if end > off:
                src = host_t[off:end]
                if use_fp16: src = src.half()
                (buf0 if use0 else buf1)[:end-off].copy_(src, non_blocking=True)

        while off < L:
            with torch.cuda.stream(comp_stream):
                comp_stream.wait_stream(copy_stream)
                cur = buf0 if use0 else buf1
                rows = min(chunk_rows, L - off)
                res = torch.matmul(cur[:rows], dev_vec if dev_vec.dtype == cur.dtype else dev_vec.to(cur.dtype))
                out_host[off:off+rows].copy_(res.float(), non_blocking=True)
            nxt = off + rows
            if nxt < L:
                with torch.cuda.stream(copy_stream):
                    end = min(nxt + chunk_rows, L)
                    nxtbuf = buf1 if use0 else buf0
                    src = host_t[nxt:end]
                    if use_fp16: src = src.half()
                    nxtbuf[:end-nxt].copy_(src, non_blocking=True)
            use0 = not use0
            off = nxt

        copy_stream.synchronize(); comp_stream.synchronize()
        return out_host

    def dot_rows(self, rows: np.ndarray, comp: np.ndarray, *, return_cosine: bool = True) -> np.ndarray:
        """Compute dot(leaf[rows], comp) and optionally divide by row norms for cosine."""
        leaf = getattr(self.mem, "_leaf_mat", None)
        if leaf is None or leaf.size == 0 or rows.size == 0:
            return np.zeros((0,), dtype=np.float32)
        sub = leaf[rows]  # [M, D]

        use_gpu = bool(getattr(self.mem, "_use_torch", False) and _HAS_TORCH and _TORCH_CUDA)
        if use_gpu:
            try:
                # pinned safeguard by size
                M, D = sub.shape
                bytes_est = M * D * 4
                use_pinned = bytes_est <= int(getattr(self.mem.hyper, "pinned_bytes_max", self.x.pinned_bytes_max))

                if use_pinned:
                    subset_t = torch.empty((M, D), dtype=torch.float32, pin_memory=True)
                    subset_t.copy_(torch.from_numpy(np.ascontiguousarray(sub)), non_blocking=False)
                else:
                    subset_t = torch.from_numpy(np.ascontiguousarray(sub))

                comp_t = torch.from_numpy(np.ascontiguousarray(comp)).pin_memory().to(getattr(self.mem, "_device", torch.device("cuda")), non_blocking=True)

                # use class streamer in mem if present; else internal
                if hasattr(self.mem, "_chunked_gpu_matvec"):
                    scores = self.mem._chunked_gpu_matvec(
                        subset_t, comp_t,
                        chunk_rows=(getattr(self.mem.hyper, "stream_chunk_rows", 0) or self.x.stream_chunk_rows),
                        cover_ratio=float(getattr(self.mem.hyper, "stream_cover_ratio", self.x.stream_cover_ratio)),
                        use_fp16=bool(getattr(self.mem.hyper, "stream_use_fp16", self.x.stream_use_fp16))
                    )
                    if _HAS_TORCH and isinstance(scores, torch.Tensor):
                        scores = scores.detach().cpu().float().numpy()
                    else:
                        scores = np.asarray(scores, dtype=np.float32)
                else:
                    scores_t = self._chunked(
                        subset_t,
                        comp_t,
                        chunk_rows=(getattr(self.mem.hyper, "stream_chunk_rows", 0) or self.x.stream_chunk_rows),
                        cover_ratio=float(getattr(self.mem.hyper, "stream_cover_ratio", self.x.stream_cover_ratio)),
                        use_fp16=bool(getattr(self.mem.hyper, "stream_use_fp16", self.x.stream_use_fp16))
                    )
                    scores = scores_t.detach().cpu().float().numpy()
            except Exception:
                scores = (sub @ comp).astype(np.float32)
        else:
            scores = (sub @ comp).astype(np.float32)

        if not return_cosine:
            return scores.astype(np.float32)

        # cosine normalization (comp is assumed normalized already)
        norms = getattr(self.mem, "_leaf_row_norms", None)
        if norms is not None and norms.shape[0] >= np.max(rows) + 1:
            denom = norms[rows, 0]
        else:
            denom = np.linalg.norm(sub, axis=1).astype(np.float32)
        denom = np.maximum(denom, 1e-8)
        return (scores / denom).astype(np.float32)
