# GRPO Training Setup Guide

Complete guide for training English-Hindi translation models using GRPO with multiple reward functions.

## Overview

This project implements GRPO (Group Relative Policy Optimization) reinforcement learning for machine translation with configurable reward functions:

- **Format Matching**: Exact and approximate format rewards
- **BLEU**: Traditional MT metric with Hindi tokenization
- **ROUGE**: Recall-oriented metric
- **chrF++**: Character-level F-score
- **COMET**: Neural reference-based metric
- **Semantic Similarity**: BGE-M3 embedding-based metric (NEW!)

## Quick Start

### 1. Setup BGE-M3 Service (Optional but Recommended)

```bash
# Navigate to service directory
cd script/bge_service

# Start the service with Docker Compose
docker-compose up -d

# Wait for model to load (~30 seconds)
# Check if service is ready
curl http://localhost:8000/

# Test the service
python test_service.py
```

### 2. Run GRPO Training

```bash
# Default config (all rewards)
uv run python script/grpo.py

# Or use a specific config
GRPO_CONFIG=conf/grpo_config_semantic_only.yaml uv run python script/grpo.py
```

## Available Configurations

### 1. All Rewards (Default)
**File**: `conf/grpo_config.yaml`

Uses all reward functions. Best for comprehensive training.

```bash
# Start BGE-M3 service first
cd script/bge_service && docker-compose up -d && cd ../..

# Run training
uv run python script/grpo.py
```

### 2. Semantic Similarity Only
**File**: `conf/grpo_config_semantic_only.yaml`

Uses only BGE-M3 semantic similarity. Fast and language-agnostic.

```bash
# Start BGE-M3 service
cd script/bge_service && docker-compose up -d && cd ../..

# Run training
GRPO_CONFIG=conf/grpo_config_semantic_only.yaml uv run python script/grpo.py
```

### 3. COMET Only
**File**: `conf/grpo_config_comet_only.yaml`

Uses only COMET neural metric. Most reliable reference-based metric.

```bash
GRPO_CONFIG=conf/grpo_config_comet_only.yaml uv run python script/grpo.py
```

### 4. Format Only
**File**: `conf/grpo_config_format_only.yaml`

Uses only format matching. Good for initial training stage.

```bash
GRPO_CONFIG=conf/grpo_config_format_only.yaml uv run python script/grpo.py
```

## Configuration Guide

### Basic Structure

```yaml
# Model settings
model:
  model_name: "unsloth/Qwen3-4B-Base"
  max_seq_length: 2048
  lora_rank: 32
  random_state: 44

# Data settings
data:
  train_file: "translation/data/english-hindi-train.xlsx"
  source_column: "Source"
  translation_column: "Translation"

# GRPO training
grpo:
  learning_rate: 5.0e-6
  max_steps: 500
  save_steps: 100
  num_generations: 4

# Rewards (enable/disable as needed)
rewards:
  bleu:
    enabled: true
    weight: 2.0

  semantic_similarity:
    enabled: true
    weight: 2.5
    service_url: "http://localhost:8000"
```

### Reward Configuration

Each reward can be enabled/disabled independently:

```yaml
rewards:
  # Format rewards
  format_exact_match:
    enabled: true
    weight: 3.0

  format_approximate:
    enabled: true
    reasoning_end: 0.5
    solution_start: 0.5
    solution_end: 0.5
    penalty: -1.0

  # Translation quality metrics
  bleu:
    enabled: true
    weight: 1.0

  rouge:
    enabled: true
    weight: 1.0

  chrf:
    enabled: true
    weight: 1.0

  comet:
    enabled: false
    weight: 1.0
    model: "Unbabel/wmt22-comet-da"

  # Semantic similarity (requires BGE-M3 service)
  semantic_similarity:
    enabled: false
    weight: 1.0
    service_url: "http://localhost:8000"
    max_passage_length: 128
    weights: [0.4, 0.2, 0.4]  # [dense, sparse, colbert]
```

## BGE-M3 Service

### Starting the Service

```bash
cd script/bge_service
docker-compose up -d
```

### Testing the Service

```bash
# Check health
curl http://localhost:8000/

# Run test script
python test_service.py
```

### Stopping the Service

```bash
cd script/bge_service
docker-compose down
```

### Service Configuration

The semantic similarity reward uses three scoring modes:
- **Dense**: Dense embedding similarity (weight: 0.4)
- **Sparse**: Lexical matching (weight: 0.2)
- **ColBERT**: Multi-vector interaction (weight: 0.4)

Final score = 0.4×dense + 0.2×sparse + 0.4×colbert

Adjust in config:
```yaml
semantic_similarity:
  weights: [0.4, 0.2, 0.4]  # [dense, sparse, colbert]
```

## Experiment Workflows

### Progressive Training

Train in stages with increasing complexity:

```bash
# Stage 1: Learn format (100 steps)
GRPO_CONFIG=conf/grpo_config_format_only.yaml uv run python script/grpo.py

# Stage 2: Learn translation quality with semantic similarity (500 steps)
GRPO_CONFIG=conf/grpo_config_semantic_only.yaml uv run python script/grpo.py

# Stage 3: Fine-tune with all metrics (500 steps)
GRPO_CONFIG=conf/grpo_config.yaml uv run python script/grpo.py
```

### Ablation Studies

Test individual reward contributions:

```bash
# 1. COMET only
GRPO_CONFIG=conf/grpo_config_comet_only.yaml uv run python script/grpo.py

# 2. Semantic similarity only
GRPO_CONFIG=conf/grpo_config_semantic_only.yaml uv run python script/grpo.py

# 3. All rewards
GRPO_CONFIG=conf/grpo_config.yaml uv run python script/grpo.py

# Compare results!
```

### Custom Configurations

Create your own config for specific experiments:

```bash
# Copy and modify
cp conf/grpo_config.yaml conf/grpo_config_custom.yaml
# Edit conf/grpo_config_custom.yaml
# Run
GRPO_CONFIG=conf/grpo_config_custom.yaml uv run python script/grpo.py
```

## Output

Training outputs are saved to:

```
outputs/
├── grpo_translation/         # Training checkpoints
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
└── grpo_lora/               # Final LoRA weights
```

## Monitoring

Training progress is logged to console:

```
Enabled reward functions: ['match_format_exactly', 'match_format_approximately',
                           'calculate_bleu', 'calculate_rouge', 'calculate_chrf',
                           'calculate_comet', 'calculate_semantic_similarity',
                           'print_examples']

Loading model and tokenizer...
Loading and preparing data...
Dataset size: 50000
Max prompt length in dataset: 512
Using all 50000 samples for GRPO training
Starting GRPO training...
```

## Troubleshooting

### BGE-M3 Service Not Starting

```bash
# Check logs
cd script/bge_service
docker logs bge-m3-service

# Restart
docker-compose restart
```

### Connection Refused to BGE-M3

Training will continue without semantic similarity if service is unavailable. Check:

```bash
curl http://localhost:8000/
```

### Out of Memory

Reduce batch size or number of generations:

```yaml
grpo:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  num_generations: 2  # Reduce from 4
```

### Stanza Download Issues

If Stanza fails to download Hindi models:

```yaml
rewards:
  stanza_download: false  # Disable auto-download
```

Then manually download:
```python
import stanza
stanza.download('hi')
```

## Performance Notes

- **GPU Memory**: ~16GB for full model + LoRA + GRPO
- **BGE-M3 Service**: ~4GB GPU memory
- **Training Speed**: ~2-5 iterations/second (depends on rewards enabled)
- **Fastest config**: `grpo_config_format_only.yaml`
- **Best quality**: `grpo_config.yaml` (all rewards)

## References
- Unsloth: [unslothai/unsloth](https://github.com/unslothai/unsloth)

## Support

For issues or questions:
1. Check the logs
2. Review configuration files in `conf/`
3. Test BGE-M3 service with `script/bge_service/test_service.py`
4. See detailed docs in `conf/README.md` and `script/bge_service/README.md`
