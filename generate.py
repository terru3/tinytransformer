## Imports
import re

import numpy as np
import torch

from constants import *
from utils import set_seed, device, collate_wrapper

## Setup
set_seed()


def generate_inference(
    model,
    tokenizer,
    device,
    method=None,
    k=None,
    p_nucleus=None,
    temp=None,
    max_new_tokens=None,
    cond="",
    deterministic=None,
):
    """
    Wrapper for generating text using the specified model. Generates unconditionally if cond=None.

    Inputs:
      -model: Decoder model to be used for text generation
      -tokenizer: Compatible tokenizer
      -device: Device of model (CPU/CUDA)
      -method (str): Decoding method for text generation ('multinomial', 'temperature', 'greedy', 'nucleus', or 'top-k')
      -k (int): Positive integer for top-k logits to sample if top-k decoding
      -p_nucleus (float/int): Cumulative probability cutoff if nucleus/top-p decoding
      -temp (float/int): Temperature if temperature decoding
      -max_new_tokens (int): Maximum number of tokens to generate
      -cond (str=''): If provided, will serve as conditional prompt for text generation
      -deterministic (int): If deterministic, uses the specified seed for model generation
    Returns:
      -res (str): Generated text string
    """

    assert method in [
        "multinomial",
        "temperature",
        "greedy",
        "nucleus",
        "top-k",
    ], "method must be 'multinomial', 'temperature', 'greedy', 'nucleus', or 'top-k'"

    if method == "temperature":
        assert (
            (temp is not None)
            and isinstance(temp, (int, float))
            and (0 < temp)
            and (temp <= 1)
        ), "temp must be defined as a number between (0, 1]"
    if method == "nucleus":
        assert (
            (p_nucleus is not None)
            and isinstance(p_nucleus, (int, float))
            and (0 < p_nucleus)
            and (p_nucleus <= 1)
        ), "p_nucleus must be defined as a number between (0, 1]"
    # if method == 'num_beams':
    #     assert isinstance(num_beams, int) and (num_beams) > 0 and (num_beams) < 100
    if method == "top-k":
        assert (
            (k is not None) and isinstance(k, int) and (k > 0) and (k < SEQ_LENGTH)
        ), "k must be defined as an integer greater than 0 and less than the model sequence length"

    if max_new_tokens is None:
        print("No max_new_tokens provided, using a default value of 250\n")
        max_new_tokens = 250

    assert (
        (max_new_tokens is not None)
        and isinstance(max_new_tokens, int)
        and (max_new_tokens) > 0
        and (max_new_tokens) <= 1000
    ), "max_new_tokens must be an integer between (0, 1000]"

    if deterministic is not None:
        set_seed(deterministic)

    if cond != "":

        cond_tokens = tokenizer(cond).input_ids

        gen_tokens = model.generate(
            torch.tensor(cond_tokens).unsqueeze(0).long().to(device),
            method=method,
            k=k,
            p_nucleus=p_nucleus,
            temp=temp,
            max_new_tokens=max_new_tokens,
        )[0]

        # Insert delimiter to indicate where prompt ends
        gen_prep = torch.zeros(
            len(gen_tokens) + 2
        ).long()  # make space for two more tokens for delimiter
        gen_prep -= 1
        gen_prep[: len(cond_tokens)] = gen_tokens[: len(cond_tokens)]
        gen_prep[-(len(gen_tokens) - len(cond_tokens)) :] = gen_tokens[
            -(len(gen_tokens) - len(cond_tokens)) :
        ]
        gen_prep[gen_prep == -1] = torch.tensor(
            tokenizer.encode(" || ")
        )  # insert tokens for || in between

        res = tokenizer.decode(gen_prep)

    else:
        empty_tokens = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long).to(
            device
        )

        res = tokenizer.batch_decode(
            model.generate(
                empty_tokens,
                method=method,
                k=k,
                p_nucleus=p_nucleus,
                temp=temp,
                max_new_tokens=max_new_tokens,
            )
        )[0]

        res = re.sub(re.escape(tokenizer.bos_token), "", res, count=2) # Remove start and end tokens

    # Clean up Unicode character issues
    # 'â€œ' then 'â€' = opening and closing double quotes
    # 'â€™' = apostrophe
    res = re.sub(r"â€œ", '"', res)
    res = re.sub(r"â€™", "'", res)
    res = re.sub(r"â€", '"', res)
    res = res + " <|endoftext|>" ## better end token with space

    return res
