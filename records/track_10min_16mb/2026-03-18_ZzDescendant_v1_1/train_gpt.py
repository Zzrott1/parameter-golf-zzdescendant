"""
Development wrapper for the zzmodel-descendant Parameter Golf entry.

This wrapper sets the v1.1 prototype defaults and delegates to the repo-root
`train_gpt.py`, which contains the actual architecture implementation for A0-A3.

Before a leaderboard PR, snapshot the exact root script used for the run into
this folder so `bytes_code` reflects the real submission artifact.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

DEFAULTS = {
    "MODEL_VARIANT": "graph",
    "NUM_LAYERS": "12",
    "NUM_BLOCK_TYPES": "2",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "2",
    "GRAPH_NODES": "4",
    "VOCAB_SIZE": "1024",
    "TIE_EMBEDDINGS": "1",
}

for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)

runpy.run_path(str(ROOT / "train_gpt.py"), run_name="__main__")
