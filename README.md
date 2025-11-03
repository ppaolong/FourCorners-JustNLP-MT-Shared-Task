# For JustNLP-MT shared task paper double blind review

## Anonymized data and model weights
For Anonymous Git repo, please see https://anonymous.4open.science/r/JustNLP-MT-Shared-Task-406A/

For data curricilum version of official training dataset, please see https://huggingface.co/datasets/anon-justnlp-mt-406A/JustNLP-MT-data-curriculum

For model weights, please see https://huggingface.co/anon-justnlp-mt-406A/JustNLP-MT-weights/tree/main , then download each model and save it in ```output/```.

## Reproducability

### Prerequisites
- This repo is using uv as python package manager (https://github.com/astral-sh/uv).
- Make sure to download model weights first to produce evaluation results.
- Or train new models using our prepared config for each settings.

### Training

Please check each training config in ```conf/``` first.

#### SFT
For SFT training, you can try to run:
```python
uv run python script/sft_qwen3.py --config sft_qwen3_32b_config.yaml # for cold start model setting
uv run python script/sft_qwen3.py --config sft_qwen3_32b_full_config.yaml # for sft baseline setting
```
Noted that in SFT training phase, we use NVIDIA A100 80gb GPU for training. Training estimation time is ~6 hours.

#### RLVR
For RLVR training, please read ```GRPO_SETUP_GUIDE.md``` and see ```script/``` for training and evaluation codes. You can try to run:
```python
uv run python script/grpo_qwen3.py --config grpo_qwen3_32b_config_bleu.yaml # for BLEU rewards only setting
```
Noted that in RLVR training phase, we use NVIDIA A100 80gb GPU for training. Training estimation time is ~110 hours.


### Evaluation
For evaluation on test set, please read ```EVALUATION_GUIDE.md``` and see ```script/``` for training and evaluation codes.

Example:
After download model weights e.g. ```grpo_qwen3_32b_bleu_merged``` and save at ```output/grpo_qwen3_32b_bleu_merged/```

```python
uv run python script/eval_qwen3_test.py --model grpo_qwen3_32b_bleu_merged
```

The script will save the codabench submit-ready csv file in ```results``` folder.