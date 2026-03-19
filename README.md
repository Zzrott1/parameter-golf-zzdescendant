# parameter-golf-zzdescendant

Work-in-progress compact language model experiments for OpenAI Model Craft: Parameter Golf.

## Current direction
This project adapts ideas from an older personal LM research effort into a submission-oriented Parameter Golf architecture focused on held-out performance per compressed byte under the official 16 MB artifact cap and 10-minute training budget.

Current experiments center on:
- tied-depth decoder-only language models
- graph-shaped FFN blocks using reusable structured micro-experts
- projection-light attention ablations
- contiguous SP-1024 tokenization
- int8 export and compression-aware evaluation

## Current ablation ladder
- A0: official baseline shell
- A1: tied-depth control
- A2: tied-depth + GraphFFN
- A3: projection-light attention ablation

## Status
Local preparation and smoke testing are complete, including model construction, forward-pass validation, int8 roundtrip checks, and setup against the official SP-1024 smoke dataset.

Next step: H100 benchmarking, post-export BPB comparison, and pruning losing variants before submission work.
