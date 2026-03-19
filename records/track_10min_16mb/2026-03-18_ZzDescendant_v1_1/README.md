# zzmodel Descendant v1.1

This record folder points at the current compact prototype implementation for the Parameter Golf adaptation.

What is implemented:
- `A0`: official baseline path via `MODEL_VARIANT=baseline NUM_LAYERS=9`
- `A1`: tied-depth control via `MODEL_VARIANT=tied NUM_LAYERS=12 NUM_BLOCK_TYPES=2`
- `A2`: tied-depth + GraphFFN via `MODEL_VARIANT=graph NUM_LAYERS=12 NUM_BLOCK_TYPES=2 GRAPH_NODES=4`
- `A3`: projection-light attention on the A2 shell via `MODEL_VARIANT=graph_proj NUM_LAYERS=12 NUM_BLOCK_TYPES=2 GRAPH_NODES=4`

What is intentionally deferred:
- side-state / carried latent
- hybrid tokenizer experiments
- width increases beyond `MODEL_DIM=512`
- richer patterned schedules beyond alternating tied blocks

Implementation notes:
- The actual code currently lives in the repo-root `train_gpt.py`.
- This folder's `train_gpt.py` is a development wrapper that sets the v1.1 defaults and then delegates to the root script.
- Before an official submission run, snapshot the exact root script into this folder and use that frozen copy for training and byte counting.

Suggested commands:
```bash
# A0: official baseline control
MODEL_VARIANT=baseline NUM_LAYERS=9 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-18_ZzDescendant_v1_1/train_gpt.py

# A1: tied-depth only
MODEL_VARIANT=tied NUM_LAYERS=12 NUM_BLOCK_TYPES=2 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-18_ZzDescendant_v1_1/train_gpt.py

# A2: tied-depth + GraphFFN
MODEL_VARIANT=graph NUM_LAYERS=12 NUM_BLOCK_TYPES=2 GRAPH_NODES=4 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-18_ZzDescendant_v1_1/train_gpt.py

# A3: projection-light attention ablation
MODEL_VARIANT=graph_proj NUM_LAYERS=12 NUM_BLOCK_TYPES=2 GRAPH_NODES=4 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-18_ZzDescendant_v1_1/train_gpt.py
```

For local non-H100 smoke runs, set `ALLOW_DEV_SDP_FALLBACK=1` so PyTorch can use math / mem-efficient attention backends when flash attention is unavailable.
If your local PyTorch install does not have Triton / Inductor available, also set `DISABLE_TORCH_COMPILE=1`.
If you only want quick train-loop validation on a desktop GPU, set `SKIP_FINAL_EVAL=1` to skip the full validation/export pass.

Smoke validation completed locally on CPU for all four variants:
- model construction
- forward pass
- int8 quantize/dequantize roundtrip
- tied/shared block schedules

No leaderboard run or `submission.json` is included yet.
