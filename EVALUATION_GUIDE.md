# Translation Model Evaluation Guide

This guide explains how to use the evaluation script to assess translation model performance.

## Overview

The evaluation script (`script/eval.py`) allows you to:
- Load a base model with or without LoRA adapters
- Generate translations on a test dataset
- Calculate standard MT metrics (BLEU, chrF++, ROUGE)
- Save detailed results and metrics

## Usage

All parameters are configured in the YAML config file. Simply run:

```bash
python script/eval.py --config conf/eval_config_sft.yaml
```

### Evaluate with LoRA Adapter

Set `lora_path` to your trained LoRA directory in the config:

```yaml
model:
  model_name: "unsloth/Qwen3-32b-unsloth-bnb-4bit"
  lora_path: "grpo_trainer_lora_model"  # Path to LoRA weights
```

Then run:
```bash
python script/eval.py --config conf/eval_config_sft.yaml
```

### Specify Output Path

Set `output_path` in the evaluation section:

```yaml
evaluation:
  output_path: "results/my_eval_results.json"
```

The script will automatically:
- Create the `results/` directory if it doesn't exist
- Save JSON metrics to `results/my_eval_results.json`
- Save CSV with translations to `results/my_eval_results.csv`

## Configuration

The evaluation script uses the same config format as training scripts but focuses on specific sections:

### Key Configuration Sections

1. **Model Config**: Defines the base model and architecture
   ```yaml
   model:
     model_name: "unsloth/Qwen3-32b-unsloth-bnb-4bit"
     max_seq_length: 2048
     load_in_4bit: true
   ```

2. **Template Config**: Defines the chat template and output format
   ```yaml
   template:
     system_prompt: "You are an expert translator..."
     reasoning_start: "<start_working_out>"  # For GRPO models
     reasoning_end: "<end_working_out>"
     solution_start: "<SOLUTION>"
     solution_end: "</SOLUTION>"
   ```

   For SFT models without reasoning, leave reasoning fields empty:
   ```yaml
   template:
     system_prompt: "You are an expert translator..."
     reasoning_start: ""
     reasoning_end: ""
     solution_start: ""
     solution_end: ""
   ```

3. **Data Config**: Specifies the evaluation dataset
   ```yaml
   data:
     dataset_name: "VISAI-AI/JustNLP-MT"
     split: "test"
     max_samples: 100  # Limit for quick testing
     source_column: "en"
     translation_column: "hi"
   ```

4. **Evaluation Config**: Controls generation parameters and optional metrics
   ```yaml
   evaluation:
     temperature: 0.7
     top_p: 0.9
     top_k: 50
     max_new_tokens: 512

     # Optional: COMET evaluation (requires service)
     comet_enabled: false
     comet_service_url: "http://localhost:44002"

     # Optional: Semantic similarity (requires service)
     semantic_similarity_enabled: false
     bge_service_url: "http://localhost:8000"
   ```

## Output Files

The script generates two output files:

1. **JSON file** (specified by `--output`): Contains aggregate metrics and configuration
   ```json
   {
     "config": {...},
     "aggregate_metrics": {
       "bleu_mean": 0.45,
       "bleu_std": 0.12,
       "chrf_mean": 0.52,
       "rouge1_mean": 0.48,
       "comet_mean": 0.75,
       "semantic_similarity_mean": 0.82,
       ...
     },
     "num_samples": 100
   }
   ```

2. **CSV file** (same path as JSON, with `.csv` extension): Contains all translations with per-sample metrics
   - Source text (original column from dataset)
   - Reference translation (gold answer)
   - Generated output (full model output with reasoning)
   - Extracted translation (parsed translation only)
   - **Per-sample metrics**:
     - `metric_bleu`: BLEU score for this sample
     - `metric_chrf`: chrF++ score for this sample
     - `metric_rouge1`: ROUGE-1 score for this sample
     - `metric_rouge2`: ROUGE-2 score for this sample
     - `metric_rougeL`: ROUGE-L score for this sample
     - `metric_comet`: COMET score (if enabled)
     - `metric_semantic_similarity`: Semantic similarity score (if enabled)

   **Example CSV structure:**
   ```
   en,hi,generated_output,extracted_translation,metric_bleu,metric_chrf,metric_rouge1,...
   "source text","reference","<reasoning>...<SOLUTION>translation</SOLUTION>","translation",0.45,0.52,0.48,...
   ```

## Metrics Explained

### Standard Metrics (Always Calculated)

- **BLEU**: Measures n-gram overlap (0-1, higher is better)
- **chrF++**: Character n-gram F-score with word ordering (0-1, higher is better)
- **ROUGE-1/2/L**: Recall-oriented metrics for unigrams, bigrams, and longest common subsequence

### Optional Metrics (Require External Services)

- **COMET**: Neural MT evaluation metric using cross-lingual embeddings (requires COMET service)
- **Semantic Similarity**: BGE-M3 multilingual embedding similarity (requires BGE-M3 service)

Each metric includes:
- `*_mean`: Average score across all samples
- `*_std`: Standard deviation showing consistency

### Enabling Optional Metrics

To use COMET and Semantic Similarity, you need to run their respective services:

1. **COMET Service** (see GRPO_SETUP_GUIDE.md for setup):
   ```yaml
   evaluation:
     comet_enabled: true
     comet_service_url: "http://localhost:44002"
   ```

2. **BGE-M3 Service** (see GRPO_SETUP_GUIDE.md for setup):
   ```yaml
   evaluation:
     semantic_similarity_enabled: true
     bge_service_url: "http://localhost:8000"
   ```

## Examples

### Quick Test on 10 Samples

```bash
# Edit config to set max_samples: 10
python script/eval.py \
    --config conf/eval_config.yaml \
    --lora-path grpo_trainer_lora_model \
    --output results/quick_test.json
```

### Full Evaluation

```bash
# Edit config to set max_samples: null (all samples)
python script/eval.py \
    --config conf/eval_config.yaml \
    --lora-path grpo_trainer_lora_model \
    --output results/full_eval.json
```

### Compare Base vs Fine-tuned

```bash
# Evaluate base model
python script/eval.py \
    --config conf/eval_config.yaml \
    --output results/base_model.json

# Evaluate fine-tuned model
python script/eval.py \
    --config conf/eval_config.yaml \
    --lora-path grpo_trainer_lora_model \
    --output results/finetuned_model.json

# Compare the metrics in both JSON files
```

## Tips

1. **Memory Management**: Use `load_in_4bit: true` for large models
2. **Quick Testing**: Set `max_samples: 10-100` for rapid iteration
3. **Temperature**: Use lower values (0.1-0.3) for more deterministic outputs
4. **LoRA Path**: Can be a local path or a Hugging Face Hub repo

## Troubleshooting

### Out of Memory
- Reduce `gpu_memory_utilization` in config
- Enable `load_in_4bit: true`
- Reduce `max_samples`

### Slow Generation
- Reduce `max_new_tokens`
- Increase `temperature` (paradoxically can be faster)
- Reduce `max_samples` for testing

### Stanza Download Issues
- Set `stanza_download: false` if already downloaded
- Or disable Stanza: `use_stanza_tokenizer: false` (uses simple split)
