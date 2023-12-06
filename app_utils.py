## Imports
import numpy as np
import torch
from transformers import AutoTokenizer

from constants import *
from model import BetterTransformer
from utils import set_seed


def load_big_model(tokenizer, device):
    ## Model architecture
    set_seed(42)
    N_HEAD = 16
    N_LAYER = 8
    N_EMBD = 768
    VOCAB_SIZE = 50258
    SEQ_LENGTH = 384

    MODEL_FILE = f"bt_{N_LAYER}_LAYERs_100_DATA_PCT_{N_EMBD}_EMBD_DIM_epoch_10.pt"

    model = BetterTransformer(
        VOCAB_SIZE,
        SEQ_LENGTH,
        N_EMBD,
        N_HEAD,
        N_LAYER,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        device=device,
    )
    model.init_params()

    model.load_state_dict(
        torch.load(f"{PATH}/model/{MODEL_FILE}", map_location=device)[
            "model_state_dict"
        ]
    )

    return model
