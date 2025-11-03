import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m")

args = parser.parse_args()
import re
import html
import os
import string

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

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"outputs/{args.model}",
        max_seq_length=2048,
        # load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )

import pandas as pd
val = pd.read_excel("./translation/data/english-hindi-test-eng.xlsx")
val_source = [clean_pipeline(line) for line in val["Source"]]

from tqdm import tqdm
from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    max_tokens = 2048,
)

messages = [[
    {"role": "system", "content": "You are an expert English to Hindi legal translator. Translate the given Legal English text to Legal Hindi accurately and naturally. The structure of the sentence in Hindi should maintain the legal nuance."},
    {"role": "user", "content": line},
] for line in val_source]

prompts = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = model.fast_generate(
    prompts,
    sampling_params = sampling_params,
)

answers = [line.outputs[0].text.split("</think>")[-1].strip() for line in outputs]
output_df = pd.DataFrame()
output_df["ID"] = val["ID"]
output_df["Translation"] = answers

os.makedirs("./results/", exist_ok=True)

output_df.to_csv(f"./results/{args.model}.csv", index=False, encoding="utf8")

