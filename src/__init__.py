
"""
htnr_brain: Brain-inspired runtime augmentations for HTNRMemoryV55X

Modules:
 - xopts: Options dataclass
 - utils: Shared helpers (normalize, cosine, signature-safe call, etc.)
 - sdr:   SDRIndexer (multi-hash SDR prefilter)
 - streams: GpuStreamer (OOC streaming matvec with pinned safeguards)
 - inhibition: LateralInhibitor (soft WTA on top-N)
 - dendrites: Predictive dendrite gating + neuromodulated plasticity wrappers
 - retrieve: Retrieve wrapper that integrates SDR→cosine→inhibition
 - maintenance: Tiled Top-K interference pass (OOC, approximate O(n^2))
 - installer: install(mem, xopts) entry point

Usage:
    from htnr_brain.installer import install
    from htnr_brain.xopts import XOpts
    install(mem, XOpts(...))
"""
from .xopts import XOpts
from .installer import install
__all__ = ["XOpts", "install"]
