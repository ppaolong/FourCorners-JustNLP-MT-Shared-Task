#!/usr/bin/env python3
"""
SFT Training Script for English-Hindi Translation
Based on Unsloth implementation with configurable chat templates
"""
from dotenv import load_dotenv
load_dotenv()

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig
import os
import re
import yaml
import torch
import pandas as pd
import numpy as np
import html
import string
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset

def contains_disallowed_character(input_string: str) -> bool:
    """
    Checks if a string contains any character that is NOT one of the following:
    - English alphabet (a-z, A-Z)
    - Hindi (Devanagari script)
    - Punctuation
    - Whitespace (space, tab, newline)

    Args:
        input_string: The string to be checked.

    Returns:
        True if a disallowed character is found, False otherwise.
    """
    # Create a set of allowed ASCII characters for fast lookup.
    # This includes English letters, standard punctuation, and whitespace.
    allowed_ascii_chars = set(
        string.ascii_letters +      # a-z, A-Z
        string.punctuation +        # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        string.whitespace +          # space, tab, newline, etc.
        '0123456789' +
        "–—" + # En dash (U+2013) and Em dash (U+2014)
        "“”’‘"
    )

    for char in input_string:
        # 1. Check if the character is in our set of allowed ASCII characters
        if char in allowed_ascii_chars:
            continue  # This character is allowed, move to the next one

        # 2. Check if the character is within the Hindi (Devanagari) Unicode range
        # The Devanagari script block is from U+0900 to U+097F.
        # ord(char) gives the Unicode code point of the character.
        is_hindi = 0x0900 <= ord(char) <= 0x097F
        if is_hindi:
            continue  # This character is allowed, move to the next one

        # 3. If the character is neither allowed ASCII nor Hindi, it's disallowed.
        # We found a disallowed character, so we can stop and return True.
        print(f"Disallowed character found: '{char}' (Unicode: U+{ord(char):04X})")
        return True
    
    # check ending char
    ending_allow_format = "।॥" + string.punctuation + "0123456789"
    if input_string[-1] not in ending_allow_format:
        return True

    # If the loop completes without finding any disallowed characters,
    # it means the entire string is valid.
    return False

def clean_pipeline(text):
    """
    A comprehensive cleaning pipeline for legal text.
    Applies a series of targeted regex and standard library functions.
    """
    # 1. Decode HTML entities (&amp;, &apos;, etc.)
    text = html.unescape(text)

    # 2. Normalize various quote characters to a standard double quote
    text = re.sub(r'[`''\""“”’‘]', '"', text)

    # 3. Standardize leading list markers (e.g., "3-" -> "3 - ")
    text = re.sub(r'^(\d+)\s*-\s*', r'\1 - ', text)

    # 4. Fix inconsistent hyphenation (e.g., "extra - ordinary" -> "extraordinary")
    text = re.sub(r'([a-zA-Z])\s*-\s*([a-zA-Z])', r'\1\2', text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Fix punctuation spacing
    text = re.sub(r'\s+([.,!?;:|।॥])', r'\1', text)
    text = re.sub(r'([.,!?;:|।॥])\s*', r'\1 ', text)

    # 7. Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # ﬁ ﬂ
    text = re.sub("ﬁ", "fi", text).strip()
    text = re.sub("ﬂ", "fl", text).strip()
    
    text = text[0].upper() + text[1:]
    return text.strip()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model and tokenizer with configurable chat template"""
    model_config = config['model']

    # Load model using FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config['model_name'],
        max_seq_length=model_config['max_seq_length'],
        load_in_4bit=model_config['load_in_4bit'],
        load_in_8bit=model_config.get('load_in_8bit', False),
        full_finetuning=model_config.get('full_finetuning', False),
    )

    # Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text
        finetune_language_layers=True,
        finetune_attention_modules=model_config.get('finetune_attention_modules', True),
        finetune_mlp_modules=model_config.get('finetune_mlp_modules', True),
        r=model_config['lora_rank'],
        lora_alpha=model_config['lora_alpha'],
        lora_dropout=model_config.get('lora_dropout', 0),
        bias=model_config.get('bias', 'none'),
        random_state=config['seed'],
    )

    # Apply chat template if not already present
    if tokenizer.chat_template is None:
        chat_template_name = config['template'].get('chat_template')
        print(f"Applying chat template: {chat_template_name}")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template_name,
        )
    else:
        print("Chat template already exists, skipping chat template setup")

    return model, tokenizer


def load_and_prepare_data(config: Dict[str, Any], tokenizer):
    """Load and prepare translation dataset"""
    data_config = config['data']
    template_config = config['template']

    # Load dataset from Hugging Face
    dataset_name = data_config.get('dataset_name', 'VISAI-AI/JustNLP-MT')
    split = data_config.get('split', 'train')

    print(f"Loading dataset {dataset_name} (split: {split})...")
    dataset = load_dataset(dataset_name, split=split)

    # Standardize data formats
    dataset = standardize_data_formats(dataset)

    # Filter by difficulty levels if specified
    difficulty_levels = data_config.get('difficulty_levels')
    if difficulty_levels is not None:
        difficulty_column = data_config.get('difficulty_column', 'Difficulty')
        original_size = len(dataset)

        print(f"Filtering dataset by difficulty levels: {difficulty_levels}")
        dataset = dataset.filter(lambda x: x[difficulty_column] in difficulty_levels)

        print(f"Dataset filtered: {original_size} -> {len(dataset)} samples")

    # Filter out samples with disallowed characters
    source_column = data_config['source_column']
    translation_column = data_config['translation_column']
    original_size = len(dataset)

    print(f"Filtering out samples with disallowed characters...")
    dataset = dataset.filter(
        lambda x: not contains_disallowed_character(clean_pipeline(x[source_column])) and
                  not contains_disallowed_character(clean_pipeline(x[translation_column]))
    )

    print(f"Dataset filtered by character check: {original_size} -> {len(dataset)} samples")

    # Limit samples if specified
    if data_config.get('max_samples'):
        dataset = dataset.select(range(min(data_config['max_samples'], len(dataset))))

    # Format dataset with chat template
    def formatting_prompts_func(examples):
        sources = examples[data_config['source_column']]
        translations = examples[data_config['translation_column']]

        convos = [
            [
                {"role": "system", "content": template_config['system_prompt']},
                {"role": "user", "content": clean_pipeline(source)},
                {"role": "assistant", "content": clean_pipeline(translation)},
            ]
            for source, translation in zip(sources, translations)
        ]

        texts = [
            tokenizer.apply_chat_template(convo,
                                          tokenize=False,
                                          add_generation_prompt=False,
                                          enable_thinking=False)
            for convo in convos
        ]

        return {"text": texts}

    print("Formatting dataset with chat template...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Shuffle dataset
    seed = config.get('seed', 44)
    print(f"Shuffling dataset with seed={seed}...")
    dataset = dataset.shuffle(seed=seed)

    return dataset


def train_sft(model, tokenizer, dataset, config: Dict[str, Any]):
    """Train model with SFT"""
    sft_config = config['sft']

    print(f"Using {len(dataset)} samples for SFT training")

    # Setup SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=sft_config['per_device_train_batch_size'],
            gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
            warmup_ratio=sft_config.get('warmup_ratio', 0.1),
            num_train_epochs=sft_config.get('num_train_epochs', 1),
            learning_rate=sft_config['learning_rate'],
            logging_steps=sft_config['logging_steps'],
            optim=sft_config['optim'],
            weight_decay=sft_config['weight_decay'],
            lr_scheduler_type=sft_config['lr_scheduler_type'],
            seed=config['seed'],
            report_to=config['logging']['report_to'],
            run_name=config['logging']['wandb_run_name'],
            save_strategy="epoch",
        ),
    )

    # Train on responses only (mask instruction part)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Starting SFT training...")
    trainer_stats = trainer.train()

    # Print training stats
    print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")

    return model, trainer_stats


def save_model(model, tokenizer, config: Dict[str, Any]):
    """Save trained model (LoRA adapters only)"""
    output_config = config['output']

    # Save LoRA adapters
    save_path = output_config['save_lora_path']
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving LoRA adapters to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"LoRA adapters and tokenizer saved to {save_path}")
    
    # Optionally save merged model
    if output_config.get('save_merged', False):
        merged_path = save_path + "_merged"
        print(f"\nSaving merged model to {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer)
        print(f"Merged model saved to {merged_path}")
    
    if output_config.get('save_gguf', False):
        if not output_config.get('save_merged', False):
            merged_path = save_path + "_merged"
            print(f"\nSaving merged model to {merged_path}")
            model.save_pretrained_merged(merged_path, tokenizer)
            print(f"Merged model saved to {merged_path}")
            
        print(f"\nSaving GGUF model...")
        model.save_pretrained_gguf(
            merged_path,
            quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
            tokenizer=tokenizer,
        )

def main():
    """Main training function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SFT Training Script for English-Hindi Translation')
    parser.add_argument(
        '--config',
        type=str,
        default='conf/sft_gemma3_config.yaml',
        help='Path to the configuration YAML file (default: conf/sft_gemma3_config.yaml)'
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    config = load_config(config_path)

    print(f"Loaded config from {config_path}")

    os.environ["WANDB_PROJECT"] = config['logging']['wandb_project']
    os.environ["WANDB_RUN_NAME"] = config['logging']['wandb_run_name']
    print(f"\nWANDB_PROJECT: {config['logging']['wandb_project']}")
    print(f"\nWANDB_RUN_NAME: {config['logging']['wandb_run_name']}")

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and prepare data
    print("\nLoading and preparing data...")
    dataset = load_and_prepare_data(config, tokenizer)
    print(f"Dataset size: {len(dataset)}")

    # Print a sample
    print("\nSample formatted text:")
    print(dataset[0]["text"][:500])

    # Train with SFT
    print("\n" + "="*50)
    print("Starting SFT Training")
    print("="*50)
    model, trainer_stats = train_sft(model, tokenizer, dataset, config)

    # Save model
    print("\n" + "="*50)
    print("Saving Model")
    print("="*50)
    save_model(model, tokenizer, config)

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
