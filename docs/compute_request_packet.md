# Compute Request Packet

Official compute request form:
- [OpenAI Parameter Golf compute credit form](https://openai.com/index/parameter-golf/)

Form fields currently shown on the official page:
- First name
- Last name
- Email
- GitHub username
- Country of residence
- Current role
- Compute support level

Recommended support level for this project right now:
- `Development grant (~$500 / ~160 compute hours)`

Why this is the right level now:
- The A0-A3 shell is implemented and smoke-checked.
- The project has a concrete architecture thesis, not just a vague idea.
- We are not yet near the top of the leaderboard, so `Advanced competitor grant` would be premature.
- `Quick-start credits` would likely be too small for the planned A0-A3 comparisons plus at least one 8xH100 finalist run.

Current repo state:
- Core implementation: `train_gpt.py`
- Record wrapper and run notes: `records/track_10min_16mb/2026-03-18_ZzDescendant_v1_1/README.md`
- Local ablation runner: `scripts/run_a0_a3_local.ps1`

Current technical status:
- `sp1024` smoke dataset is downloaded locally.
- Local tokenizer path exists at `data/tokenizers/fineweb_1024_bpe.model`.
- Implemented variants:
  - `A0`: baseline
  - `A1`: tied-depth only
  - `A2`: tied-depth + GraphFFN
  - `A3`: projection-light attention on the A2 shell
- Verified locally:
  - Python syntax check
  - model construction
  - forward pass
  - int8 quantize/dequantize roundtrip
  - A2 desktop smoke reached warmup and the first train step on the local RTX 4070 Ti
- Available local GPU for iteration:
  - `NVIDIA GeForce RTX 4070 Ti`
- Local dev runs can enable `ALLOW_DEV_SDP_FALLBACK=1` to use non-flash SDP backends without changing the challenge defaults.
- Local dev runs can enable `DISABLE_TORCH_COMPILE=1` if the local machine lacks the Triton / Inductor stack expected by the challenge script.
- Local dev smokes can enable `SKIP_FINAL_EVAL=1` to validate the train path quickly without paying for the full validation/export loop on desktop hardware.

Suggested post-grant run order:
1. Run `A0-A3` on `1xH100` for short comparisons.
2. Rank by post-export `val_bpb`, compressed bytes, and step time.
3. Kill losing branches immediately.
4. Freeze the exact winning `train_gpt.py` into the record folder.
5. Run the finalist on `8xH100` under the 10-minute cap.
6. Only then consider side-state, tokenizer edits, or richer patterned schedules.

Checklist before submitting the form:
- Confirm the GitHub username you want attached to the challenge work.
- Choose `Development grant`.
- Be ready to point at the repo, the implemented A0-A3 plan, and the fact that the local smoke dataset is already in place.

Local caveat:
- On this Windows desktop stack, local single-GPU smokes are useful for partial validation but are not the authoritative environment for challenge timing or end-to-end reproducibility. Use a Linux CUDA box for the first serious A0-A3 comparison run.
