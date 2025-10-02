
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import inspect

def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x)); 
        return x if n <= eps else (x / n)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, (n + eps), out=np.zeros_like(x), where=(n > eps))

def cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps: return 0.0
    v = float(np.dot(a, b) / (na * nb))
    return float(np.clip(v, -1.0, 1.0))

def fast_dot(a: np.ndarray, b: np.ndarray) -> float:
    v = float(np.dot(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)))
    return max(-1.0, min(1.0, v))

def safe_call(orig_fn, query, **kwargs):
    """Call original retrieve_for_query honoring its signature; robust to variants."""
    try:
        sig = inspect.signature(orig_fn)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return orig_fn(query, **accepted)
    except Exception:
        try:
            if "K_cap" in kwargs: 
                return orig_fn(query, kwargs["K_cap"])
        except Exception:
            pass
        return orig_fn(query)

class DummyLock:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
