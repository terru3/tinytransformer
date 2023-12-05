## Archived——Tokenized data has now been uploaded to Hugging Face and is loaded from there

## Imports
from datasets import load_dataset
from transformers import AutoTokenizer

from utils import set_seed, device

## Setup
set_seed()

## Data Loading
data = load_dataset("roneneldan/TinyStories")

## Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# vocab size 50257, model_max_length 2048

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess(data):
  inputs = tokenizer(data['text'],
        padding='max_length',
        truncation=True,
        max_length=384 # after tokenization, truncating to 384 affects < 10% of data
    )
  return inputs

data_tokenized = data.map(preprocess,
                        batched=True)

data_tokenized.save_to_disk('/tokenized_data_384.hf')