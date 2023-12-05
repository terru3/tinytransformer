## TO MODULARIZE

## Model architecture
N_HEAD = 4
N_LAYER = 4
N_EMBD = 256
VOCAB_SIZE = 50258
SEQ_LENGTH = 384

## Data and training
BATCH_SIZE = 32 # increase if you have more VRAM! double as much as is possible
DATA_PCT = 0.20 # percent of training data
# ROC_DATA_PCT = 0.40 # percent of ROC benchmark data to validate on (40% is roughly same size as TinyStories val)
MAX_LR = 1e-3 # crazy
# PCT_WARMUP = 0.02

## Epoch-level hyperparameters
EPOCHS = 100
SAVE_EVERY = int(EPOCHS*0.05) # save model every x epochs
GENERATE_EVERY = int(EPOCHS*0.05) # generate text from model every x epochs
# PERIOD_NUM_EPOCH = 10 # period of cosine annealing scheduler, in number of epochs

## Step-level hyperparameters——how often to compute train and validation losses——appear later!
## Those require the data to be loaded in first to compute number of steps in a batch

## Model loading
# If loading, set CHECKPOINT = True and specify LOAD_EPOCH
# If training from scratch, set CHECKPOINT = False and specify LOAD_EPOCH=None
CHECKPOINT = False
LOAD_EPOCH = None
# REMINDER: Set LOAD_EPOCH = None if training from scratch

START_EPOCH = LOAD_EPOCH if LOAD_EPOCH is not None else 0

path = '/content/drive/Shareddrives/DSU Better Transformer'

MODEL_NAME = f"bt_{N_LAYER}_LAYERs_{int(DATA_PCT*100)}_DATA_PCT_{N_EMBD}_EMBD_DIM"
print("Model Name:", MODEL_NAME)
print(f'Model will be saved every {SAVE_EVERY} epochs, and will generate text every {GENERATE_EVERY} epochs')

# Imports

import json
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
# from torchinfo import summary
from tqdm import tqdm

from transformers import AutoTokenizer

from utils import set_seed, device

"""# Setup"""
set_seed()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

"""# Data"""

data = load_dataset('terru3/tokenized_tinystories384')
data = data.with_format('torch') # format using tensors rather than native Python; otherwise batches give lists not tensors

data = data.shuffle(seed=42)
train_data = data['train'].select(range(int(DATA_PCT*len(data['train']))))
val_data = data['validation']

# ## ROCStories benchmark
# ROC_data = load_from_disk(f'{path}/ROCStories/tokenized_ROC.hf')
# ROC_data = ROC_data.with_format('torch')
# ROC_data = ROC_data.shuffle(seed=42)
# ROC_data = ROC_data.select(range(int(ROC_DATA_PCT*len(ROC_data))))
# ROC_data

"""# Model"""

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # replaced ReLU
            nn.Dropout(p=dropout),
            nn.Linear(4 * n_embd, n_embd),

            # inverted bottleneck. original design choice has multiplier = 4. The larger the model, the larger this multiplier can be
            # if small like 50M, maybe use smaller like 1.5-2?
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_length, dropout=0.1):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head # Dimension of each head's key, query, and value
        assert self.head_dim * n_head == self.n_embd, "n_embd must be divisible by n_head"
        self.seq_length = seq_length
        self.drop = nn.Dropout(p=dropout)

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False) # multi-head combining weight matrix

    def split_heads(self, x):
        B, S, D = x.size()
        # split dimension into n_head * head_dim, then transpose the sequence length w/ n_head
        # output: [B, n_head, S, head_dim]
        return x.view(B, S, self.n_head, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        # use permute or transpose to reverse
        # taking a view earlier may produce a non-contiguous tensor, so we convert back because view needs a contiguous input
        B, _, S, head_dim = x.size() # _ is n_head which we will merge
        # output: [B, S, n_embd]
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    def scaled_dot_product(self, q, k, v, dropout, mask=None):
        # q,k,v are [B, n_head, S, head_dim]
        # the key transpose sets up batch multiplication s.t. wei = [B, n_head, S, S]
        wei = q @ k.transpose(-2,-1) / np.sqrt(self.head_dim)
        # mask is [B, 1, S, S], so simply broadcasted across each head and works as expected
        if mask is not None:
          wei = wei.masked_fill(mask, float('-inf'))
        wei = dropout(F.softmax(wei, dim=-1))
        out = wei @ v
        return out

    def forward(self, x, mask=None):
        # x: (B, S, n_embd)
        # Step 1 and 2: Project full query, key, value, then split via reshaping
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))

        # Step 3: Compute scaled dot-product attention with causal mask
        # not done. should use generate_mask
        attn = self.scaled_dot_product(q, k, v, self.drop, mask)

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn)) # (B, S, n_embd)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_length, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, seq_length, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # experimentally, apply layer norm before attention/MLP
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # residual connection (stream)
        x = x + self.drop(self.sa(self.ln1(x), mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class PositionalEncoding(nn.Module):
  """
  Formula taken from the original Transformer paper:
  PE(pos, 2i (even)) = sin(pos/(10000^{2i/d_model}))
  PE(pos, 2i+1 (odd)) = cos(pos/(10000^{2i/d_model}))

  See reference for more details:
  https://kikaben.com/transformers-positional-encoding/
  """
  def __init__(self, d_model, max_len):
      # just set d_model = n_embd and max_len = seq_len
      super().__init__()

      position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
      divisor = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model / 2, half for each of sin and cos]
      pe = torch.zeros(max_len, d_model)
      pe[:, 0::2] = torch.sin(position * divisor) # 0 for second dim or :?
      pe[:, 1::2] = torch.cos(position * divisor)
      self.register_buffer('pe', pe) # result: self.pe = [max_len, d_model], mapping each token index to a vector of length d_model as desired

  def forward(self, x):
      # x = torch.arange(seq_length) has shape [seq_length], so x.size(0) extracts it, then we index self.pe for the first seq_length mappings
      # note we do not add the positional embeddings to x itself yet, we simply return them
      # output = (seq_length, d_model=n_embd)
      return self.pe[:x.size(0)]

class BetterTransformer(nn.Module):
    def __init__(self, vocab_size, seq_length,
                 n_embd, n_head, n_layer,
                 pad_idx, eos_idx,
                 device, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)
        # we need to make sure the embedding ignores the padding token right?
        self.position_embedding = PositionalEncoding(n_embd, seq_length)
        self.blocks = nn.Sequential(*[Block(n_embd,
                                            n_head,
                                            seq_length,
                                            dropout) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.seq_length = seq_length
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.device = device
        self.init_params()

    # optional weight initialization (Xavier uniform)
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # Remark: Xavier normal is not supported at this time.

    def get_causal_mask(self, x):
        """
        Generates causal mask for decoding
        """
        seq_len = x.size(-1) # x = (batch_size x seq_len)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # k = 1 shifts the diagonal, so that the main diagonal gets 0's
        return (torch.from_numpy(subsequent_mask) == 0).to(self.device) # (1, seq_len x seq_len)
        # True along main diagonal + below, False elsewhere

    def get_pad_mask(self, x, pad_idx):
        """
        Generates padding mask
        """
        return (x != pad_idx).unsqueeze(1).unsqueeze(-2).to(self.device)
        # (batch_size x 1 x 1 x seq_len)

    def forward(self, x, targets=None):

        # should alr be int64 tokens but explicit cast in case
        x = x.to(torch.int64)
        B, S = x.shape

        # get mask
        mask = self.get_pad_mask(x, self.pad_idx) & self.get_causal_mask(x).to(self.device)
        # mask = (batch_size x 1 x seq_len x seq_len)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(S))
        x = self.drop(tok_emb + pos_emb)
        # (B, S, n_embd)
        for block in self.blocks:
            x = block(x, ~mask) # (batch_size, seq_length, n_embd)
        # negate mask to fill originally False values with -inf later
        logits = self.lm_head(x) # (batch_size, seq_length, vocab_size)

        # this code assumes teacher forcing——for each text of seq length S we have S autoregressive predictions,
        # thus we have B*S logits and B*S targets
        if targets is None:
            loss = None
        else:
            B, S, C = logits.shape
            logits = logits.view(B*S, C)
            targets = targets.view(B*S)
            loss = F.cross_entropy(logits, targets, ignore_index=self.pad_idx)

        return logits, loss


    def generate(self, input_ids, method='multinomial',
                 max_new_tokens=1000, temp=None,
                 num_beams=None, p_nucleus=None, k=None):

        # References:
        # https://huggingface.co/transformers/v3.4.0/_modules/transformers/generation_utils.html

        # Assertions, other complex logic, etc., are built into the generate_inference function.
        # When model.generate is called with generate_train, arguments are fixed.

        # input_ids begins as (batch_size, seq_length)

        self.eval()

        for _ in range(max_new_tokens):
            # for future compatibility, if method == beam, may take a different approach
            if method in ['multinomial', 'temperature', 'greedy', 'nucleus', 'top-k']:
                # i) Truncate to the most recent `max length` tokens
                text_cond = input_ids[:, -self.seq_length:]
                # ii) Retrieve predictions
                with torch.no_grad():
                    logits, _ = self(text_cond)
                # model output: (batch_size, seq_length, vocab_size)
                # iii) Find last token logits of each
                logits = logits[:, -1, :] # (batch_size, vocab_size)

                # aside: if temperature sampling, divide logits by temp before applying softmax
                if method == 'temperature':
                    logits = logits / temp

                # iv) Take softmax along each
                probs = F.softmax(logits, dim=-1)

                # v) Sample next token depending on method
                if method == 'greedy':
                    next_idx = probs.argmax(dim=-1).unsqueeze(-1)

                elif method in ['multinomial', 'temperature', 'nucleus', 'top-k']:
                    if method == 'nucleus':
                        assert p_nucleus is not None and (0 < p_nucleus) and (p_nucleus <= 1)

                        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                        prob_cumsum = sorted_probs.cumsum(dim=-1)
                        idx_remove = prob_cumsum > p_nucleus
                        # shift one right to ensure the first token is above the threshold
                        idx_remove[..., 1:] = idx_remove[..., :-1].clone()
                        idx_remove[..., 0] = False
                        # retrieve original indices by reverse-sorting
                        remove_mask = idx_remove.gather(dim=-1,
                                          index=sorted_idx.argsort(dim=-1))
                        # ^ specifically, we do this by first argsorting the indices which were returned from argsort
                        # this returns indices that when used to subset a sorted array, returns the original array in unsorted order
                        # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
                        # torch.gather is how we apply a multi-dimensional index
                        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
                        probs[remove_mask] = 0

                    if method == 'top-k':
                        remove_mask = probs < torch.topk(probs, k).values[..., -1, None] # the topk returns (B, 1), leaving only the
                        # kth largest probs (i.e. the cutoff value for each). Then mask is same size as probs (B, vocab_size)
                        probs[remove_mask] = 0

                    # Sample probabilistically via scores
                    next_idx = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

                # vi) Autoregressively append to input_text
                input_ids = torch.cat((input_ids, next_idx), dim=-1)
                # end prematurely if <EOS> generated
                if next_idx == self.eos_idx:
                  break
                # now input_text = (batch_size, seq_length + 1)

        return input_ids

"""# Training

### Create shifted inputs (teacher forcing)
^ In teacher forcing, regardless of the model's output at each timestep, it receives the true value as input for the next timestep. This is efficient because you don't need to run the model sequentially, the outputs at the different sequence locations can be computed in parallel.
"""

def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int):
  shifted_input_ids = input_ids.new_zeros(input_ids.shape)
  shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() # cut off last token
  shifted_input_ids[:, 0] = decoder_start_token_id # add the start token
  return shifted_input_ids

# re-defining how the dataloader will load batches, using our fn above
def collate_wrapper(batch):

  # batch = list (of len batch_size) of dictionaries (text, input_ids, attention_mask). We ignore attention_mask
  targets = [i['input_ids'] for i in batch]
  targets = torch.stack(targets) # batch_size * seq_len of input_ids as expected

  # because GPT tokenizer does not add EOS id, we manually add here
  first_pad_idx = torch.sum(targets != tokenizer.pad_token_id, dim=-1).unsqueeze(-1) # (batch_size, 1)
  # by summing how many in that sequence are not padding, we arrive at the index of the first padding token
  # if sequence has no padding, first_pad_idx = seq_length so we must subtract 1 so we can use it to index the last token
  first_pad_idx[first_pad_idx == SEQ_LENGTH] = (SEQ_LENGTH-1)

  # Replace first padding token with EOS in-place
  # scatter_ is like gather except you can assign values to the retrieved elements
  targets.scatter_(index=first_pad_idx, dim=-1, value=tokenizer.eos_token_id) # EOS for GPT-2 is the same as BOS (50256)

  # generate shifted model inputs
  inputs = shift_tokens_right(targets, tokenizer.bos_token_id) # tokenizer.bos_token_id gives integer index of our BOS token, <|endoftext|>
  # these shifted tokens become our inputs to the model. it always starts with the BOS token (50256)
  return inputs, targets

train_dataloader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=BATCH_SIZE,
                              collate_fn=collate_wrapper)

val_dataloader = DataLoader(val_data,
                            shuffle=True,
                            batch_size=BATCH_SIZE,
                            collate_fn=collate_wrapper)

# ROC_dataloader = DataLoader(ROC_data,
#                               shuffle=True,
#                               batch_size=BATCH_SIZE,
#                               collate_fn=collate_wrapper)

model = BetterTransformer(VOCAB_SIZE, SEQ_LENGTH, N_EMBD, N_HEAD, N_LAYER,
                          tokenizer.pad_token_id, tokenizer.eos_token_id, device).to(device)
model.init_params()

print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

# num_steps = len(train_dataloader) * EPOCHS

# warmup = lambda step: (step+1) / (PCT_WARMUP*num_steps)

# warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
# train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, PERIOD_NUM_EPOCH*len(train_dataloader))
# # second arg indicates num iterations between warm restarts.
# # set as some function of num_steps, maybe reset every 1 epoch? idk what's standard
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler],
#                                                                 [PCT_WARMUP*num_steps]) # last argument indicates when to switch scheduler from warmup to train

## Step-level hyperparameters——CHANGE as needed

# Compute training loss roughly 10 times per epoch and val loss once an epoch

TRAIN_EVERY = round(int(len(train_dataloader) / 10),
                    -2) # nearest hundred

print(f'There are {len(train_dataloader)} batches in an epoch; train loss is computed every {TRAIN_EVERY} steps and',
      f'validation loss is computed every one epoch.')

## Load model as needed

def load(model, optimizer):
    model.load_state_dict(torch.load(f'{path}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt',
                                  map_location=device)["model_state_dict"])
    print("Model loaded")

    optimizer.load_state_dict(torch.load(f'{path}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt',
                                      map_location=device)["optimizer_state_dict"])
    print("Optimizer loaded")

    # scheduler.load_state_dict(torch.load(f'{path}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt',
    #                                      map_location=device)["scheduler_state_dict"])
    # print("Scheduler loaded")

    with open(f'{path}/train_logs/{MODEL_NAME}_train_losses.json', 'r') as f3:
      train_losses = json.load(f3)

    with open(f'{path}/train_logs/{MODEL_NAME}_val_losses.json', 'r') as f4:
      val_losses = json.load(f4)

    print("Losses loaded")

## Train

def train(model,
          train_dataloader, val_dataloader,
          device, optimizer,
          train_loss_list=None, val_loss_list=None):

    train_losses = train_loss_list if train_loss_list is not None else []
    val_losses = val_loss_list if val_loss_list is not None else []

    model.train()
    model.to(device)

    # Set up prompt generation
    generation_file_path = f'{path}/model/OUTPUT_{MODEL_NAME}.txt'
    empty_tokens = torch.full((1,1), tokenizer.bos_token_id, dtype=torch.long).to(device)
    cond_prompts = ["Once there was a strong girl named Alyssa. She loved to lift weights. She",
                    "One day, Casey was driving his car. He wanted to race with the police. He",
                    "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she",
                    "Once upon a time, there was a cat who got lost in the forest. One day,",
                    "Terry saw a big red dog in the alley. He",
                    "One day, Jake kissed Harry in bed. They made love in the night.",
                    "Poppy was extremely tired. Her mom told her to wash the dishes, but she just wanted to",
                    "One day, Daniel went to the beach. He brought a",
                    "Once there was a tiny cat named Bob. He wanted to eat the cookies on the counter, but"]

    cond_token_list = tokenizer(cond_prompts).input_ids

    for epoch in range(START_EPOCH, EPOCHS):
        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

        # Compute validation loss each epoch (separate function)
        evaluate(model,
                 val_dataloader,
                 device,
                 val_losses,
                 epoch)

        train_times = [] # running average of train batch times, resets every epoch

        for step, batch in enumerate(train_dataloader):

            start = time.perf_counter()

            optimizer.zero_grad()

            dec_input = batch[0].to(device)
            targets = batch[1].to(device)

            logits, loss = model(dec_input, targets)
            loss.backward()

            # Monitoring gradient norm
            grads = [
                    param.grad.detach().flatten()
                    for param in model.parameters()
                    if param.grad is not None
                ]
            norm = torch.cat(grads).norm()

            optimizer.step()
            # scheduler.step() # step every batch, not epoch

            train_times.append(time.perf_counter()-start)

            # remove time of last batch later. temporarily there for estimation of training time
            if step % TRAIN_EVERY == 0:
                if step != 0: # avoid clashing with validation print statement
                    print(f"Epoch: {epoch+1}/{EPOCHS} | Step: {step}/{len(train_dataloader)} | Train Loss: {loss.item():.5f} |",
                              f"Grad Norm: {norm:.5f} | Train Batch Time: {np.mean(train_times):.3f}")

                    # if using scheduler, print learning rate here, and also use scientific notation
                    # same for epoch-level print statement
                    # {scheduler.get_last_lr()[0]:.3g}

                train_losses.append(loss.item())

        # save model every few epochs, save losses and generated text every epoch
        if epoch % SAVE_EVERY == 0:
            print("Saving model at current epoch")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                # "scheduler_state_dict": scheduler.state_dict()},
                f'{path}/model/{MODEL_NAME}_epoch_{epoch+1}.pt')

        with open(f'{path}/train_logs/{MODEL_NAME}_train_losses.json', 'w') as f:
          json.dump(train_losses, f)

        with open(f'{path}/train_logs/{MODEL_NAME}_val_losses.json', 'w') as f2:
          json.dump(val_losses, f2)

        # Generate and store unconditional and conditional prompts
        if epoch % GENERATE_EVERY == 0:
          generate_train(model, tokenizer,
                        generation_file_path,
                        empty_tokens, cond_token_list,
                        epoch)

        print(f"Epoch {epoch+1} finished, model + data saved \n")

def evaluate(model,
             val_dataloader,
             device,
             val_losses,
             epoch):
    """
    Evaluates model on validation dataset
    """

    start = time.perf_counter()
    model.eval()

    # Evaluate on validation dataset
    with torch.no_grad():
        val_losses_temp = []
        for batch in val_dataloader:
            dec_input = batch[0].to(device)
            targets = batch[1].to(device)
            logits, loss = model(dec_input, targets)
            val_losses_temp.append(loss.item())

    avg_val_loss = sum(val_losses_temp) / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch: {epoch+1}/{EPOCHS} | Full Val Loss: {avg_val_loss:.5f} |",
          f"Val Batch Time: {time.perf_counter()-start:.3f}")

    model.train()

def generate_train(model, tokenizer,
                   generation_file_path,
                   empty_tokens, cond_token_list, epoch):
    """
  Generates model output to unconditional and a bed of conditional prompts via top-k (5) (default), writes output to file
    """

    set_seed(42)

    uncond_res1 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='top-k', k=5,
                                                       max_new_tokens=150))[0]
    uncond_res2 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='greedy',
                                                       max_new_tokens=150))[0]
    uncond_res3 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='nucleus', p_nucleus=0.5,
                                                       max_new_tokens=150))[0]
    uncond_res4 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='multinomial',
                                                       max_new_tokens=150))[0]
    uncond_res5 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='top-k', k=5,
                                                       max_new_tokens=250))[0]
    uncond_res6 = tokenizer.batch_decode(model.generate(empty_tokens,
                                                       method='nucleus', p_nucleus=0.5,
                                                       max_new_tokens=250))[0]

    cond_res_list = []
    for prompt in cond_token_list:
        gen_tokens = model.generate(torch.tensor(prompt).unsqueeze(0).long().to(device),
                      method='top-k', k=5,
                      max_new_tokens=250)[0]

        # Insert delimiter to indicate where prompt ends
        gen_prep = torch.zeros(len(gen_tokens)+2).long() # make space for two more tokens for delimiter
        gen_prep -= 1
        gen_prep[:len(prompt)] = gen_tokens[:len(prompt)]
        gen_prep[-(len(gen_tokens)-len(prompt)):] = gen_tokens[-(len(gen_tokens)-len(prompt)):]
        gen_prep[gen_prep == -1] = torch.tensor(tokenizer.encode(' || ')) # insert tokens for || in between

        cond_res = tokenizer.decode(gen_prep)
        cond_res_list.append(cond_res)

    cond_res_list = '\n\n'.join(cond_res_list)

    generation_text = f"""{MODEL_NAME} Output @Epoch {epoch+1}
    UNCONDITIONAL GENERATION:

    Top-k (5) (150 max_tokens):
    {uncond_res1}

    Greedy (150 max_tokens):
    {uncond_res2}

    Nucleus (0.5) (150 max_tokens):
    {uncond_res3}

    Multinomial (150 max_tokens):
    {uncond_res4}

    Top-k (5) (250 max_tokens):
    {uncond_res5}

    Nucleus (0.5) (250 max_tokens):
    {uncond_res6}

    #####################################################
    CONDITIONAL GENERATION (Top-k (5), 250 max_tokens):
    {cond_res_list}
    -----------------------------------------------------
    """
    with open(generation_file_path, 'a') as file:
      file.write(generation_text)
    print(generation_text)

"""# Driver code"""

# TRAIN:
if CHECKPOINT:
  load(model, optimizer)
  train(model, train_dataloader, val_dataloader, device, optimizer,
        train_losses, val_losses)
else:
  train(model, train_dataloader, val_dataloader, device, optimizer)

## Loss curve

##Generation

# # if loading in already trained model just for generation, can re-run this
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model = BetterTransformer(VOCAB_SIZE, SEQ_LENGTH, N_EMBD, N_HEAD, N_LAYER,
#                           tokenizer.pad_token_id, tokenizer.eos_token_id, device).to(device)
# model.load_state_dict(torch.load(f'{path}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt',
#                                   map_location=device)["model_state_dict"])

def generate_inference(model, tokenizer, device,
                       method=None, k=None, p_nucleus=None, temp=None,
                       max_new_tokens=None, cond=None, deterministic=None):
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
      -cond (str=None): If provided, will serve as conditional prompt for text generation
      -deterministic (int): If deterministic, uses the specified seed for model generation
    Returns:
      -res (str): Generated text string
    """

    assert method in ['multinomial', 'temperature', 'greedy', 'nucleus', 'top-k'], \
        "method must be 'multinomial', 'temperature', 'greedy', 'nucleus', or 'top-k'"

    if method == 'temperature':
        assert (temp is not None) and isinstance(temp, (int, float)) and (0 < temp) and (temp <= 1), \
        "temp must be defined as a number between (0, 1]"
    if method == 'nucleus':
        assert (p_nucleus is not None) and isinstance(p_nucleus, (int, float)) and (0 < p_nucleus) and (p_nucleus <= 1), \
        "p_nucleus must be defined as a number between (0, 1]"
    # if method == 'num_beams':
    #     assert isinstance(num_beams, int) and (num_beams) > 0 and (num_beams) < 100
    if method == 'top-k':
        assert (k is not None) and isinstance(k, int) and (k > 0) and (k < SEQ_LENGTH), \
        "k must be defined as an integer greater than 0 and less than the model sequence length"

    if max_new_tokens is None:
        print('No max_new_tokens provided, using a default value of 250\n')
        max_new_tokens = 250

    assert (max_new_tokens is not None) and isinstance(max_new_tokens, int) and (max_new_tokens) > 0 and (max_new_tokens) <= 1000, \
    "max_new_tokens must be an integer between (0, 1000]"

    if deterministic is not None:
        set_seed(deterministic)

    if cond is not None:

        cond_tokens = tokenizer(cond).input_ids

        gen_tokens = model.generate(torch.tensor(cond_tokens).unsqueeze(0).long().to(device),
                                    method=method, k=k, p_nucleus=p_nucleus, temp=temp,
                                    max_new_tokens=max_new_tokens)[0]

        # Insert delimiter to indicate where prompt ends
        gen_prep = torch.zeros(len(gen_tokens)+2).long() # make space for two more tokens for delimiter
        gen_prep -= 1
        gen_prep[:len(cond_tokens)] = gen_tokens[:len(cond_tokens)]
        gen_prep[-(len(gen_tokens)-len(cond_tokens)):] = gen_tokens[-(len(gen_tokens)-len(cond_tokens)):]
        gen_prep[gen_prep == -1] = torch.tensor(tokenizer.encode(' || ')) # insert tokens for || in between

        res = tokenizer.decode(gen_prep)

    else:
        empty_tokens = torch.full((1,1), tokenizer.bos_token_id, dtype=torch.long).to(device)

        res = tokenizer.batch_decode(model.generate(empty_tokens,
                                                    method=method, k=k,
                                                    p_nucleus=p_nucleus, temp=temp,
                                                    max_new_tokens=max_new_tokens))[0]

        res = re.sub(re.escape(tokenizer.bos_token), '', res, count=1)

    # Clean up Unicode character issues
    # 'â€œ' then 'â€' = opening and closing double quotes
    # 'â€™' = apostrophe
    res = re.sub(r'â€œ', '"', res)
    res = re.sub(r'â€™', "'", res)
    res = re.sub(r'â€', '"', res)

    return res

"""#### i) Unconditional"""

generate_inference(model, tokenizer, device, method='multinomial')

generate_inference(model, tokenizer, device, method='top-k', k=5)

"""#### ii) Conditional"""

generate_inference(model, tokenizer, device, method='top-k', k=5, cond='One morning, Jack woke up and got out of')