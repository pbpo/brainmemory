
from __future__ import annotations
from .xopts import XOpts
from .sdr import SDRIndexer
from .streams import GpuStreamer
from .inhibition import LateralInhibitor
from .dendrites import wrap_update_contexts, wrap_ingest_batch_freeze, wrap_dendritic_gating
from .retrieve import wrap_retrieve
from .maintenance import wrap_maintenance


def _configure_embedder(mem, *, model_name: str = "google/embeddinggemma-300m") -> None:
    """Ensure the memory exposes a SentenceTransformer encoder for embeddings."""
    existing = getattr(mem, "_brain_embedder_name", None)
    if existing == model_name and getattr(mem, "model", None) is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "sentence-transformers is required to use google/embeddinggemma-300m"
        ) from exc

    kwargs = {"trust_remote_code": True}

    device_hint = None
    try:
        import torch

        if getattr(mem, "_use_torch", False) and torch.cuda.is_available():
            dev = getattr(mem, "_device", None)
            device_hint = dev if dev is not None else torch.device("cuda")
    except Exception:
        device_hint = None

    if device_hint is not None:
        kwargs["device"] = device_hint if isinstance(device_hint, str) else str(device_hint)

    model = SentenceTransformer(model_name, **kwargs)

    try:
        dim = int(model.get_sentence_embedding_dimension())
    except Exception:
        dim = None

    mem.model = model
    mem._brain_embedder_name = model_name

    if dim and dim > 0:
        setattr(mem, "dim", dim)

def install(mem, xopts: XOpts = None) -> None:
    """Install brain-inspired patches on a HTNRMemoryV55X instance."""
    x = xopts if xopts is not None else XOpts()
    mem._x = x  # attach options

    _configure_embedder(mem)

    # Core state
    if not hasattr(mem, "_delta_M"):
        D = int(getattr(mem, "dim", 0) or (mem.M_agg.size if getattr(mem, "M_agg", None) is not None else 0))
        mem._delta_M = (mem.M_agg.copy() if getattr(mem, "M_agg", None) is not None else None)
        if mem._delta_M is None or (hasattr(mem._delta_M, "size") and mem._delta_M.size == 0):
            import numpy as _np
            mem._delta_M = _np.zeros(D, dtype=_np.float32) if D > 0 else _np.zeros((0,), dtype=_np.float32)

    # Components
    mem._brain_sdr = SDRIndexer(mem, x)
    mem._brain_streamer = GpuStreamer(mem, x)
    mem._brain_inhibitor = LateralInhibitor(mem, x.inhibit_topN, x.inhibit_gamma, x.inhibit_iters)

    # Monkey patches
    wrap_update_contexts(mem, x)
    wrap_ingest_batch_freeze(mem, x)
    wrap_dendritic_gating(mem, x)
    wrap_retrieve(mem, x, mem._brain_sdr, mem._brain_streamer, mem._brain_inhibitor)
    wrap_maintenance(mem, x)

    # Initial SDR build if enabled
    try:
        mem._brain_sdr.maybe_rebuild()
    except Exception:
        pass

