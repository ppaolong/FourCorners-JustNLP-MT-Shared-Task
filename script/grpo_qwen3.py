#!/usr/bin/env python3
"""
GRPO Training Script for Gemma-3 English-Hindi Translation
Based on Unsloth Gemma-3 GRPO implementation with MT metrics rewards
"""
from dotenv import load_dotenv
load_dotenv()

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats
from trl import GRPOConfig, GRPOTrainer
import os
import re
import yaml
import torch
import pandas as pd
import numpy as np
import html
import string
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
import stanza
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer

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
    
    # check ending char hindi
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


class TranslationRewardFunctions:
    """Reward functions for translation quality evaluation using MT metrics"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rewards_config = config['rewards']

        # Check which rewards are enabled
        self.check_format_enabled = self.rewards_config.get('check_format', {}).get('enabled', False)
        self.allow_char_enabled = self.rewards_config.get('allow_char', {}).get('enabled', False)
        self.bleu_enabled = self.rewards_config.get('bleu', {}).get('enabled', False)
        self.rouge_enabled = self.rewards_config.get('rouge', {}).get('enabled', False)
        self.chrf_enabled = self.rewards_config.get('chrf', {}).get('enabled', False)
        self.comet_enabled = self.rewards_config.get('comet', {}).get('enabled', False)
        self.semantic_sim_enabled = self.rewards_config.get('semantic_similarity', {}).get('enabled', False)

        # Cache for format check results (used as prerequisite for other rewards)
        self.format_check_cache = None

        # Initialize Hindi tokenizer with Stanza only if BLEU or ROUGE is enabled
        self.hindi_nlp = None
        if self.bleu_enabled or self.rouge_enabled:
            if self.rewards_config.get('stanza_download', True):
                try:
                    stanza.download(self.rewards_config['stanza_lang'], verbose=False)
                except:
                    pass

            self.hindi_nlp = stanza.Pipeline(
                self.rewards_config['stanza_lang'],
                processors='tokenize',
                verbose=False,
                download_method=None
            )

        # Initialize metrics based on what's enabled
        self.bleu = BLEU(effective_order=True) if self.bleu_enabled else None
        self.chrf = CHRF(word_order=2) if self.chrf_enabled else None  # chrF++
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        ) if self.rouge_enabled else None

        # Initialize COMET service if enabled
        self.comet_service_url = None
        self.comet_use_service = self.rewards_config.get('comet', {}).get('use_service', True)

        if self.comet_enabled:
            if self.comet_use_service:
                # Use COMET as a service
                self.comet_service_url = self.rewards_config['comet'].get(
                    'service_url', 'http://localhost:44002'
                )
                # Test connection
                try:
                    response = requests.get(f"{self.comet_service_url}/", timeout=5)
                    if response.status_code == 200:
                        print(f"Connected to COMET service at {self.comet_service_url}")
                    else:
                        print(f"Warning: COMET service returned status {response.status_code}")
                        self.comet_enabled = False
                except Exception as e:
                    print(f"Warning: Could not connect to COMET service: {e}")
                    print(f"Make sure the service is running at {self.comet_service_url}")
                    self.comet_enabled = False

        # Initialize BGE-M3 semantic similarity service if enabled
        self.bge_service_url = None
        if self.semantic_sim_enabled:
            self.bge_service_url = self.rewards_config['semantic_similarity'].get(
                'service_url', 'http://localhost:8000'
            )
            # Test connection
            try:
                response = requests.get(f"{self.bge_service_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"Connected to BGE-M3 service at {self.bge_service_url}")
                else:
                    print(f"Warning: BGE-M3 service returned status {response.status_code}")
                    self.semantic_sim_enabled = False
            except Exception as e:
                print(f"Warning: Could not connect to BGE-M3 service: {e}")
                print(f"Make sure the service is running at {self.bge_service_url}")
                self.semantic_sim_enabled = False

        # Print step counter
        self.printed_times = 0
        self.print_every_steps = config['logging'].get('print_every_steps', 10)
        
        self.think_pattern = re.compile(
            r'<think>(.*?)</think>\s*(.*)',
            flags=re.DOTALL
        )

    def tokenize_hindi(self, text: str) -> List[str]:
        """Tokenize Hindi text using Stanza"""
        doc = self.hindi_nlp(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens

    def extract_translation(self, response: str) -> Optional[str]:
        """Extract translation from response with <think>...</think> format"""
        match = self.think_pattern.search(response)
        if match:
            # Extract the answer part (after </think>)
            answer = match.group(2).strip()
            return answer if answer else None
        # If no <think> tags found, return the entire response
        return response.strip() if response.strip() else None

    def extract_thinking(self, response: str) -> Optional[str]:
        """Extract thinking process from response"""
        match = self.think_pattern.search(response)
        if match:
            # Extract the thinking part (inside <think>...</think>)
            thinking = match.group(1).strip()
            return thinking if thinking else None
        return None
    
    def check_format_reward(self, prompts, completions, **kwargs):
        """Check format"""
        ending_allow_format = "।॥" + string.punctuation + "0123456789"
        responses = [completion[0]["content"] for completion in completions]
        format_match = [self.think_pattern.search(response) for response in responses]
        check_ending = [True if line.group(0).strip()[-1] in ending_allow_format else False for line in format_match]
        weight = self.rewards_config['check_format']["weight"]
        scores = [1 * weight if line else 0 for line in check_ending]

        # Cache the boolean format check results (True/False) for use by other rewards
        self.format_check_cache = check_ending

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[CHECK FORMAT] Average: {avg_score:.4f}, Min: {min(scores):.4f}, Max: {max(scores):.4f}")
        return scores
    
    def check_allow_char_reward(self, prompts, completions, **kwargs):
        """Check allow characters"""
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]
        disallowed = [contains_disallowed_character(line) for line in extracted]
        weight = self.rewards_config['allow_char']["weight"]

        # Apply format check prerequisite: if format check failed, set score to 0
        if self.format_check_cache is not None:
            scores = [1 * weight if (line and format_ok) else 0 for line, format_ok in zip(disallowed, self.format_check_cache)]
        else:
            scores = [1 * weight if line else 0 for line in disallowed]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[ALLOW CHARACTERS] Average: {avg_score:.4f}, Min: {min(scores):.4f}, Max: {max(scores):.4f}")
        return scores

    def calculate_bleu(self, prompts, completions, reference_translation, **kwargs):
        """Calculate BLEU score reward"""
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]

        scores = []
        weight = self.rewards_config['bleu']['weight']

        for idx, (prediction, reference) in enumerate(zip(extracted, reference_translation)):
            # Apply format check prerequisite: if format check failed, set score to 0
            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                scores.append(0.0)
                continue

            if prediction is None:
                scores.append(-1.0)
                continue

            # Tokenize Hindi text
            pred_tokens = self.tokenize_hindi(prediction)
            ref_tokens = self.tokenize_hindi(reference)

            # Calculate BLEU
            bleu_score = self.bleu.sentence_score(
                ' '.join(pred_tokens),
                [' '.join(ref_tokens)]
            ).score / 100.0  # Normalize to 0-1

            scores.append(bleu_score * weight)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[BLEU] Average: {avg_score:.4f}, Min: {min(scores):.4f}, Max: {max(scores):.4f}")
        return scores

    def calculate_rouge(self, prompts, completions, reference_translation, **kwargs):
        """Calculate ROUGE score reward"""
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]

        scores = []
        weight = self.rewards_config['rouge']['weight']

        for idx, (prediction, reference) in enumerate(zip(extracted, reference_translation)):
            # Apply format check prerequisite: if format check failed, set score to 0
            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                scores.append(0.0)
                continue

            if prediction is None:
                scores.append(-1.0)
                continue

            # Tokenize Hindi text
            pred_tokens = ' '.join(self.tokenize_hindi(prediction))
            ref_tokens = ' '.join(self.tokenize_hindi(reference))

            # Calculate ROUGE
            rouge_scores = self.rouge_scorer.score(ref_tokens, pred_tokens)
            # Use average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
            # avg_rouge = (
            #     rouge_scores['rouge1'].fmeasure +
            #     rouge_scores['rouge2'].fmeasure +
            #     rouge_scores['rougeL'].fmeasure
            # ) / 3.0
            avg_rouge = rouge_scores['rougeL'].fmeasure

            scores.append(avg_rouge * weight)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[ROUGE] Average: {avg_score:.4f}, Min: {min(scores):.4f}, Max: {max(scores):.4f}")
        return scores

    def calculate_chrf(self, prompts, completions, reference_translation, **kwargs):
        """Calculate chrF++ score reward"""
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]

        scores = []
        weight = self.rewards_config['chrf']['weight']

        for idx, (prediction, reference) in enumerate(zip(extracted, reference_translation)):
            # Apply format check prerequisite: if format check failed, set score to 0
            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                scores.append(0.0)
                continue

            if prediction is None:
                scores.append(-1.0)
                continue

            # Calculate chrF++
            chrf_score = self.chrf.sentence_score(
                prediction,
                [reference]
            ).score / 100.0  # Normalize to 0-1

            scores.append(chrf_score * weight)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[chrF++] Average: {avg_score:.4f}, Min: {min(scores):.4f}, Max: {max(scores):.4f}")
        return scores

    def calculate_comet(self, prompts, completions, reference_translation, **kwargs):
        """Calculate COMET score reward"""
        source_texts = [prompts[0][-1]["content"] for _ in completions]
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]

        weight = self.rewards_config['comet']['weight']
        batch_size = self.rewards_config['comet'].get('batch_size', 8)

        # Prepare COMET input
        comet_samples = []
        valid_indices = []

        for idx, (src, pred, ref) in enumerate(zip(source_texts, extracted, reference_translation)):
            # Apply format check prerequisite: skip if format check failed
            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                continue

            if pred is not None:
                comet_samples.append({
                    "src": src,
                    "mt": pred,
                    "ref": ref
                })
                valid_indices.append(idx)

        # Calculate COMET scores
        if comet_samples:
            if self.comet_use_service and self.comet_service_url:
                # Use COMET service
                try:
                    response = requests.post(
                        f"{self.comet_service_url}/score",
                        json={
                            "samples": comet_samples,
                            "batch_size": batch_size,
                            "gpus": 1
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        comet_scores = result['scores']

                        # Map back to original indices, applying format check prerequisite
                        result_scores = []
                        for idx in range(len(completions)):
                            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                                result_scores.append(0.0)
                            elif idx in valid_indices:
                                score_idx = valid_indices.index(idx)
                                result_scores.append(comet_scores[score_idx] * weight)
                            else:
                                result_scores.append(0.0)

                        avg_score = sum(result_scores) / len(result_scores) if result_scores else 0.0
                        valid_scores = [s for s in result_scores if s > 0.0]
                        min_score = min(valid_scores) if valid_scores else 0.0
                        max_score = max(valid_scores) if valid_scores else 0.0
                        print(f"[COMET] Average: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
                        return result_scores
                    else:
                        print(f"Warning: COMET service returned status {response.status_code}")
                        return [0.0] * len(completions)

                except Exception as e:
                    print(f"Error calling COMET service: {e}")
                    return [0.0] * len(completions)

        return [0.0] * len(completions)

    def calculate_semantic_similarity(self, prompts, completions, reference_translation, **kwargs):
        """Calculate semantic similarity using BGE-M3 model"""
        if not self.semantic_sim_enabled or self.bge_service_url is None:
            return [0.0] * len(completions)

        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_translation(r) for r in responses]

        weight = self.rewards_config['semantic_similarity']['weight']
        max_length = self.rewards_config['semantic_similarity'].get('max_passage_length', 128)
        weights_modes = self.rewards_config['semantic_similarity'].get('weights', [0.4, 0.2, 0.4])

        # Prepare sentence pairs for batch processing
        sentence_pairs = []
        valid_indices = []

        for idx, (pred, ref) in enumerate(zip(extracted, reference_translation)):
            # Apply format check prerequisite: skip if format check failed
            if self.format_check_cache is not None and not self.format_check_cache[idx]:
                continue

            if pred is not None:
                sentence_pairs.append({
                    "sentence_1": ref,  # Reference translation
                    "sentence_2": pred  # Predicted translation
                })
                valid_indices.append(idx)

        # Calculate semantic similarity if we have valid pairs
        if sentence_pairs:
            try:
                response = requests.post(
                    f"{self.bge_service_url}/similarity",
                    json={
                        "sentence_pairs": sentence_pairs,
                        "max_passage_length": max_length,
                        "weights": weights_modes
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    similarity_scores = result['scores']

                    # Map back to original indices, applying format check prerequisite
                    result_scores = []
                    for idx in range(len(completions)):
                        if self.format_check_cache is not None and not self.format_check_cache[idx]:
                            result_scores.append(0.0)
                        elif idx in valid_indices:
                            score_idx = valid_indices.index(idx)
                            result_scores.append(similarity_scores[score_idx] * weight)
                        else:
                            result_scores.append(0.0)

                    avg_score = sum(result_scores) / len(result_scores) if result_scores else 0.0
                    valid_scores = [s for s in result_scores if s > 0.0]
                    min_score = min(valid_scores) if valid_scores else 0.0
                    max_score = max(valid_scores) if valid_scores else 0.0
                    print(f"[Semantic Similarity] Average: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
                    return result_scores
                else:
                    print(f"Warning: BGE-M3 service returned status {response.status_code}")
                    return [0.0] * len(completions)

            except Exception as e:
                print(f"Error calling BGE-M3 service: {e}")
                return [0.0] * len(completions)

        return [0.0] * len(completions)

    def get_reward_functions(self):
        """Return list of enabled reward functions"""
        reward_funcs = []

        # Add translation quality rewards if enabled
        if self.check_format_enabled:
            reward_funcs.append(self.check_format_reward)
        if self.allow_char_enabled:
            reward_funcs.append(self.check_allow_char_reward)
            
        if self.bleu_enabled:
            reward_funcs.append(self.calculate_bleu)

        if self.rouge_enabled:
            reward_funcs.append(self.calculate_rouge)

        if self.chrf_enabled:
            reward_funcs.append(self.calculate_chrf)

        if self.comet_enabled:
            reward_funcs.append(self.calculate_comet)

        if self.semantic_sim_enabled:
            reward_funcs.append(self.calculate_semantic_similarity)

        # Print which rewards are enabled
        enabled_rewards = [func.__name__ for func in reward_funcs]
        print(f"Enabled reward functions: {enabled_rewards}")

        return reward_funcs


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup Gemma-3 model and tokenizer"""
    model_config = config['model']

    # Load model using FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['model_name'],
        max_seq_length=model_config['max_seq_length'],
        load_in_4bit=model_config.get('load_in_4bit', False),
        load_in_8bit=model_config.get('load_in_8bit', False),
        full_finetuning=model_config.get('full_finetuning', False),
        fast_inference=model_config.get('fast_inference', False),
        gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.5),
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        # finetune_vision_layers=False,  # Turn off for just text
        # finetune_language_layers=True,
        # finetune_attention_modules=model_config.get('finetune_attention_modules', True),
        # finetune_mlp_modules=model_config.get('finetune_mlp_modules', True),
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
    """Load and prepare translation dataset for GRPO"""
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

    # Format dataset for GRPO - we need prompts and reference translations
    def formatting_prompts_func(examples):
        sources = examples[data_config['source_column']]
        translations = examples[data_config['translation_column']]

        prompts = [
            [
                {"role": "system", "content": template_config['system_prompt']},
                {"role": "user", "content": clean_pipeline(source)},
            ]
            for source in sources
        ]

        # Store reference translations for reward calculation
        reference_translations = [clean_pipeline(translation) for translation in translations]

        return {
            "prompt": prompts,
            "reference_translation": reference_translations,
        }

    print("Formatting dataset for GRPO training...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Shuffle dataset
    seed = config.get('seed', 44)
    print(f"Shuffling dataset with seed={seed}...")
    dataset = dataset.shuffle(seed=seed)

    return dataset


def train_grpo(model, tokenizer, dataset, config: Dict[str, Any]):
    """Train model with GRPO"""
    grpo_config = config['grpo']
    max_prompt_length = grpo_config.get('max_prompt_length', 256)

    print(f"Using {len(dataset)} samples for GRPO training")

    # Create reward functions
    reward_functions = TranslationRewardFunctions(config)
    reward_funcs = reward_functions.get_reward_functions()

    # Setup GRPO trainer
    training_args = GRPOConfig(
        learning_rate=grpo_config['learning_rate'],
        adam_beta1=grpo_config.get('adam_beta1', 0.9),
        adam_beta2=grpo_config.get('adam_beta2', 0.99),
        weight_decay=grpo_config['weight_decay'],
        warmup_ratio=grpo_config.get('warmup_ratio', 0.1),
        lr_scheduler_type=grpo_config['lr_scheduler_type'],
        optim=grpo_config['optim'],
        logging_steps=grpo_config['logging_steps'],
        per_device_train_batch_size=grpo_config['per_device_train_batch_size'],
        gradient_accumulation_steps=grpo_config['gradient_accumulation_steps'],
        num_generations=grpo_config.get('num_generations', 4),
        max_prompt_length=max_prompt_length,
        max_completion_length=config['model']['max_seq_length'] - max_prompt_length,
        num_train_epochs=grpo_config.get('num_train_epochs', 1),
        save_strategy=grpo_config.get('save_strategy', 'epoch'),
        max_grad_norm=grpo_config.get('max_grad_norm', 0.1),
        report_to=config['logging']['report_to'],
        run_name=config['logging']['wandb_run_name'],
        output_dir=config['output']['save_lora_path'],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    print(f"Watch the 'reward' column - it should increase over time!")

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
    parser = argparse.ArgumentParser(description='GRPO Training Script for Gemma-3 English-Hindi Translation')
    parser.add_argument(
        '--config',
        type=str,
        default='conf/grpo_gemma3_config.yaml',
        help='Path to the configuration YAML file (default: conf/grpo_gemma3_config.yaml)'
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    config = load_config(config_path)

    print(f"Loaded config from {config_path}")

    os.environ["WANDB_PROJECT"] = config['logging']['wandb_project']
    os.environ["WANDB_RUN_NAME"] = config['logging']['wandb_run_name']
    print(f"\nWANDB_PROJECT: {config['logging']['wandb_project']}")
    print(f"WANDB_RUN_NAME: {config['logging']['wandb_run_name']}")

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup model and tokenizer
    print("\nLoading Gemma-3 model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and prepare data
    print("\nLoading and preparing data...")
    dataset = load_and_prepare_data(config, tokenizer)
    print(f"Dataset size: {len(dataset)}")

    # Print a sample
    print("\nSample prompt:")
    sample_prompt = tokenizer.apply_chat_template(
        dataset[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True
    )
    print(sample_prompt[:500] + "...")
    print(f"\nReference translation: {dataset[0]['reference_translation']}")

    # Train with GRPO
    print("\n" + "="*50)
    print("Starting GRPO Training")
    print("="*50)
    model, trainer_stats = train_grpo(model, tokenizer, dataset, config)

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
