## Imports
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from constants import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if multi-GPU
    torch.backends.cudnn.deterministic = (
        True  # only applies to CUDA convolution operations
    )
    torch.backends.cudnn.benchmark = False
    # usually CuDNN has heuristics as to which algorithm to pick.
    # cudnn.benchmark benchmarks several algorithms and picks the fastest, which is often helpful
    # if your input shapes are fixed and not changing a lot during training. However, this means it
    # may pick a different algorithm even when the deterministic flag is set.
    # As such it is good practice to turn off cudnn.benchmark when turning on cudnn.deterministic


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Training setup

### Create shifted inputs (teacher forcing)

# In teacher forcing, regardless of the model's output at each timestep,
# it receives the true value as input for the next timestep. This is efficient because you don't need to run the
# model sequentially, the outputs at the different sequence locations can be computed in parallel.


def load_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # cut off last token
    shifted_input_ids[:, 0] = decoder_start_token_id  # add the start token
    return shifted_input_ids


tokenizer = load_tokenizer

# define custom collate function for dataloader
def collate_wrapper(batch):

    # batch = list (of len batch_size) of dictionaries (text, input_ids, attention_mask). We ignore attention_mask
    targets = [i["input_ids"] for i in batch]
    targets = torch.stack(targets)  # batch_size * seq_len of input_ids as expected

    # because GPT tokenizer does not add EOS id, we manually add here
    first_pad_idx = torch.sum(targets != tokenizer.pad_token_id, dim=-1).unsqueeze(
        -1
    )  # (batch_size, 1)
    # by summing how many in that sequence are not padding, we arrive at the index of the first padding token
    # if sequence has no padding, first_pad_idx = seq_length so we must subtract 1 so we can use it to index the last token
    first_pad_idx[first_pad_idx == SEQ_LENGTH] = SEQ_LENGTH - 1

    # Replace first padding token with EOS in-place
    # scatter_ is like gather except you can assign values to the retrieved elements
    targets.scatter_(
        index=first_pad_idx, dim=-1, value=tokenizer.eos_token_id
    )  # EOS for GPT-2 is the same as BOS (50256)

    # generate shifted model inputs
    inputs = shift_tokens_right(targets, tokenizer.bos_token_id)
    # begin each input sequence to the model with the id of the BOS token: <|endoftext|>
    return inputs, targets
