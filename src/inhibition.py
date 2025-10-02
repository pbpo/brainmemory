
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .utils import normalize

class LateralInhibitor:
    """Soft-WTA style lateral inhibition over top-N chosen nodes."""
    def __init__(self, mem, topN: int, gamma: float, iters: int):
        self.mem = mem
        self.topN = int(topN)
        self.gamma = float(gamma)
        self.iters = int(iters)

    def apply(self, chosen: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not chosen:
            return chosen
        N = min(max(3, self.topN), len(chosen))
        # preserve original order within topN; drop missing nodes
        headN = [(int(nid), float(sc)) for (nid, sc) in chosen[:N] if int(nid) in self.mem.nodes]
        if len(headN) < 3:
            return chosen
        idsN = [nid for nid, _ in headN]
        id2score = {int(nid): float(sc) for nid, sc in chosen}
        embs = np.stack([self.mem.nodes[i].emb for i in idsN], axis=0).astype(np.float32)
        embs = normalize(embs)
        S = (embs @ embs.T).astype(np.float32)
        s = np.array([id2score[i] for i in idsN], dtype=np.float32)
        for _ in range(max(1, self.iters)):
            for i in range(1, len(idsN)):
                s[i] = s[i] - self.gamma * float(np.sum(S[i, :i] * s[:i]))
        s = np.maximum(s, -1.0)
        head = list(zip(idsN, s.astype(np.float32).tolist()))
        tail = chosen[N:]
        return sorted(head + tail, key=lambda z: -z[1])
