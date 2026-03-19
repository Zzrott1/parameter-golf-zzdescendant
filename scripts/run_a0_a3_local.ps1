param(
    [string]$PythonExe = "python",
    [string]$RunScript = "records\track_10min_16mb\2026-03-18_ZzDescendant_v1_1\train_gpt.py",
    [string]$DataPath = "data\datasets\fineweb10B_sp1024",
    [string]$TokenizerPath = "data\tokenizers\fineweb_1024_bpe.model",
    [int]$Iterations = 200,
    [int]$TrainBatchTokens = 65536,
    [int]$ValBatchSize = 65536,
    [int]$NProcPerNode = 1
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $RunScript)) {
    throw "Run script not found: $RunScript"
}
if (-not (Test-Path $DataPath)) {
    throw "Dataset path not found: $DataPath"
}
if (-not (Test-Path $TokenizerPath)) {
    throw "Tokenizer path not found: $TokenizerPath"
}

$variants = @(
    @{ RunId = "A0_baseline_smoke"; Variant = "baseline"; Layers = "9";  BlockTypes = "2"; GraphNodes = "4" },
    @{ RunId = "A1_tied_smoke";     Variant = "tied";     Layers = "12"; BlockTypes = "2"; GraphNodes = "4" },
    @{ RunId = "A2_graph_smoke";    Variant = "graph";    Layers = "12"; BlockTypes = "2"; GraphNodes = "4" },
    @{ RunId = "A3_proj_smoke";     Variant = "graph_proj"; Layers = "12"; BlockTypes = "2"; GraphNodes = "4" }
)

foreach ($spec in $variants) {
    Write-Host "=== Running $($spec.RunId) ==="
    $env:RUN_ID = $spec.RunId
    $env:MODEL_VARIANT = $spec.Variant
    $env:NUM_LAYERS = $spec.Layers
    $env:NUM_BLOCK_TYPES = $spec.BlockTypes
    $env:GRAPH_NODES = $spec.GraphNodes
    $env:MODEL_DIM = "512"
    $env:NUM_HEADS = "8"
    $env:NUM_KV_HEADS = "4"
    $env:MLP_MULT = "2"
    $env:VOCAB_SIZE = "1024"
    $env:TIE_EMBEDDINGS = "1"
    $env:ALLOW_DEV_SDP_FALLBACK = "1"
    $env:DISABLE_TORCH_COMPILE = "1"
    $env:SKIP_FINAL_EVAL = "1"
    $env:DATA_PATH = $DataPath
    $env:TOKENIZER_PATH = $TokenizerPath
    $env:ITERATIONS = "$Iterations"
    $env:TRAIN_BATCH_TOKENS = "$TrainBatchTokens"
    $env:VAL_BATCH_SIZE = "$ValBatchSize"
    $env:VAL_LOSS_EVERY = "0"
    $env:TRAIN_LOG_EVERY = "10"
    $env:MAX_WALLCLOCK_SECONDS = "0"

    & $PythonExe -m torch.distributed.run --standalone --nproc_per_node=$NProcPerNode $RunScript
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($spec.RunId)"
    }
}

Write-Host "A0-A3 local smoke sequence completed."
