# Transformer Sampling Industrial Validation Runbook

## Goal

Run isolated sampling-bias experiments for interview narrative and mechanism validation.
Do not modify canonical outputs, README, resume numbers, or `CLAUDE.md` canonical numbers.

## Local Workflow

Use the project PyTorch interpreter:

```powershell
D:\tool\spyder\envs\pytorch\python.exe
```

Push code only from the local Windows repository. The server is a read-only code mirror.

## Server Queue

Launch from `/workspace/amazon-two-tower` with an external log:

```bash
mkdir -p /workspace/server-logs
tmux new-session -d -s transformer_sampling_overnight \
  "cd /workspace/amazon-two-tower && \
   python scripts/run_transformer_sampling_overnight.py \
     --generated_config_dir /workspace/server-logs/transformer_sampling_generated_configs \
   2>&1 | tee /workspace/server-logs/transformer_sampling_overnight_queue.log"
```

Smoke order:

```text
baseline-infonce
empirical-oldlogq-alpha025
uber-batchq-alpha025
uber-batchq-alpha100
refined-logq
mns5050
mns5050-refined-logq
```

The queue then selects:

```text
balanced candidate
Recall upper-bound candidate
```

Each unique candidate runs:

```text
seed42 full train + full eval
full-test exposure audit
4ch valid-selected weighted RRF + frozen test once
Faiss FlatIP alignment
IVF nprobe=16/32/64
HNSW efSearch=64/128
```

## Outputs

```text
outputs/transformer_sampling_industrial_smoke/
outputs/transformer_sampling_industrial_smoke_audit/
outputs/transformer_sampling_industrial_selection/
outputs/text_timeaware_transformer_sampling_full/
outputs/transformer_sampling_full_audit/
outputs/multichannel_transformer_sampling/
outputs/faiss_transformer_sampling/
```

Generated machine-specific configs stay outside the repository:

```text
/workspace/server-logs/transformer_sampling_generated_configs/
```

## Acceptance Boundary

- Smoke baseline must align with historical `0.124460` within `0.0005`.
- Balanced candidates must pass Recall, coverage, head-share, and non-head bucket gates.
- Recall upper-bound candidates remain `diagnostic-only` unless they also pass health gates.
- Faiss reports offline ANN retrieval consistency and latency only.
- Do not update canonical decisions until Eddy manually reviews the completed artifacts.
