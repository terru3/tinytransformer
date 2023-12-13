## Imports
import json
import time

import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

# from torchinfo import summary

from transformers import AutoTokenizer

from constants import *
from model import BetterTransformer
from utils import set_seed, device, collate_wrapper

## Setup
set_seed()

## Data
data = load_dataset("terru3/tokenized_tinystories384")
data = data.with_format(
    "torch"
)  # format using tensors rather than native Python; otherwise batches give lists not tensors
data = data.shuffle(seed=42)
train_data = data["train"].select(range(int(DATA_PCT * len(data["train"]))))
val_data = data["validation"]

# ### ROCStories benchmark
# ROC_data = load_from_disk(f'/ROCStories/tokenized_ROC.hf')
# ROC_data = ROC_data.with_format('torch')
# ROC_data = ROC_data.shuffle(seed=42)
# ROC_data = ROC_data.select(range(int(ROC_DATA_PCT*len(ROC_data))))


def get_dataloaders():
    """
    Returns DataLoaders.
    """
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_wrapper
    )

    val_dataloader = DataLoader(
        val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_wrapper
    )

    # ROC_dataloader = DataLoader(ROC_data,
    #                               shuffle=True,
    #                               batch_size=BATCH_SIZE,
    #                               collate_fn=collate_wrapper)
    return train_dataloader, val_dataloader


def prep_train(train_dataloader):
    """
    Returns newly initialized model, tokenizer, and optimizer.
    """
    model = BetterTransformer(
        VOCAB_SIZE,
        SEQ_LENGTH,
        N_EMBD,
        N_HEAD,
        N_LAYER,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        device,
    ).to(device)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

    # num_steps = len(train_dataloader) * EPOCHS
    # warmup = lambda step: (step+1) / (PCT_WARMUP*num_steps)
    # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    # train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, PERIOD_NUM_EPOCH*len(train_dataloader))
    # # second arg indicates num iterations between warm restarts
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler],
    #                                                                 [PCT_WARMUP*num_steps])
    # # last argument indicates when to switch scheduler from warmup to train

    return model, tokenizer, optimizer


def load(model, optimizer):
    model.load_state_dict(
        torch.load(
            f"{PATH}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt", map_location=device
        )["model_state_dict"]
    )
    print("Model loaded")

    optimizer.load_state_dict(
        torch.load(
            f"{PATH}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt", map_location=device
        )["optimizer_state_dict"]
    )
    print("Optimizer loaded")

    # scheduler.load_state_dict(torch.load(f'{PATH}/model/{MODEL_NAME}_epoch_{LOAD_EPOCH}.pt',
    #                                      map_location=device)["scheduler_state_dict"])
    # print("Scheduler loaded")

    with open(f"{PATH}/train_logs/{MODEL_NAME}_train_losses.json", "r") as f3:
        train_losses = json.load(f3)

    with open(f"{PATH}/train_logs/{MODEL_NAME}_val_losses.json", "r") as f4:
        val_losses = json.load(f4)

    print("Losses loaded")
    return model, optimizer, train_losses, val_losses


## Train


def train(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    device,
    optimizer,
    train_loss_list=None,
    val_loss_list=None,
):

    train_losses = train_loss_list if train_loss_list is not None else []
    val_losses = val_loss_list if val_loss_list is not None else []

    model.train()
    model.to(device)

    # Set up prompt generation
    generation_file_path = f"{PATH}/model/OUTPUT_{MODEL_NAME}.txt"
    empty_tokens = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long).to(
        device
    )
    cond_prompts = [
        "Once there was a strong girl named Alyssa. She loved to lift weights. She",
        "One day, Casey was driving his car. He wanted to race with the police. He",
        "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she",
        "Once upon a time, there was a cat who got lost in the forest. One day,",
        "Terry saw a big red dog in the alley. He",
        "Poppy was extremely tired. Her mom told her to wash the dishes, but she just wanted to",
        "One day, Daniel went to the beach. He brought a",
        "Once there was a tiny cat named Bob. He wanted to eat the cookies on the counter, but",
    ]

    cond_token_list = tokenizer(cond_prompts).input_ids

    # Compute frequency to print training statistics
    # Default: Compute training loss roughly 10 times per epoch and val loss once an epoch
    COMPUTE_EVERY = round(
        int(len(train_dataloader) / COMPUTE_PER_EPOCH), -2
    )  # nearest hundred steps

    print(
        f"There are {len(train_dataloader)} batches in an epoch; train loss is computed every {COMPUTE_EVERY} steps and",
        f"validation loss is computed every epoch.",
    )

    for epoch in range(START_EPOCH, EPOCHS):
        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

        # Compute validation loss each epoch
        evaluate(model, val_dataloader, device, val_losses, epoch)

        train_times = []  # running average of train batch times, resets every epoch

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

            train_times.append(time.perf_counter() - start)

            # remove time of last batch later. temporarily there for estimation of training time
            if step % COMPUTE_EVERY == 0:
                if step != 0:  # avoid clashing with validation print statement
                    print(
                        f"Epoch: {epoch+1}/{EPOCHS} | Step: {step}/{len(train_dataloader)} | Train Loss: {loss.item():.5f} |",
                        f"Grad Norm: {norm:.5f} | Train Batch Time: {np.mean(train_times):.3f}",
                    )

                    # if using scheduler, print learning rate here, and also use scientific notation
                    # same for epoch-level print statement
                    # {scheduler.get_last_lr()[0]:.3g}

                train_losses.append(loss.item())

        # save model every few epochs, save losses and generated text every epoch
        if epoch % SAVE_EVERY == 0:
            print("Saving model at current epoch")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                # "scheduler_state_dict": scheduler.state_dict()},
                f"{PATH}/model/{MODEL_NAME}_epoch_{epoch+1}.pt",
            )

        with open(f"{PATH}/train_logs/{MODEL_NAME}_train_losses.json", "w") as f:
            json.dump(train_losses, f)

        with open(f"{PATH}/train_logs/{MODEL_NAME}_val_losses.json", "w") as f2:
            json.dump(val_losses, f2)

        # Generate and store unconditional and conditional prompts
        if epoch % GENERATE_EVERY == 0:
            generate_train(
                model,
                tokenizer,
                generation_file_path,
                empty_tokens,
                cond_token_list,
                epoch,
            )

        print(f"Epoch {epoch+1} finished, model + data saved \n")


def evaluate(model, val_dataloader, device, val_losses, epoch):
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

    print(
        f"Epoch: {epoch+1}/{EPOCHS} | Full Val Loss: {avg_val_loss:.5f} |",
        f"Val Batch Time: {time.perf_counter()-start:.3f}",
    )

    model.train()


def generate_train(
    model, tokenizer, generation_file_path, empty_tokens, cond_token_list, epoch
):
    """
    Generates model output to unconditional and a bed of conditional prompts via top-k (5) (default), writes output to file
    """

    set_seed(42)

    uncond_res1 = tokenizer.batch_decode(
        model.generate(empty_tokens, method="top-k", k=5, max_new_tokens=150)
    )[0]
    uncond_res2 = tokenizer.batch_decode(
        model.generate(empty_tokens, method="greedy", max_new_tokens=150)
    )[0]
    uncond_res3 = tokenizer.batch_decode(
        model.generate(
            empty_tokens, method="nucleus", p_nucleus=0.5, max_new_tokens=150
        )
    )[0]
    uncond_res4 = tokenizer.batch_decode(
        model.generate(empty_tokens, method="multinomial", max_new_tokens=150)
    )[0]
    uncond_res5 = tokenizer.batch_decode(
        model.generate(empty_tokens, method="top-k", k=5, max_new_tokens=250)
    )[0]
    uncond_res6 = tokenizer.batch_decode(
        model.generate(
            empty_tokens, method="nucleus", p_nucleus=0.5, max_new_tokens=250
        )
    )[0]

    cond_res_list = []
    for prompt in cond_token_list:
        gen_tokens = model.generate(
            torch.tensor(prompt).unsqueeze(0).long().to(device),
            method="top-k",
            k=5,
            max_new_tokens=250,
        )[0]

        # Insert delimiter to indicate where prompt ends
        gen_prep = torch.zeros(
            len(gen_tokens) + 2
        ).long()  # make space for two more tokens for delimiter
        gen_prep -= 1  # set all ids to -1 to avoid clashing with token ids
        # fill in prompt and generated tokens
        gen_prep[: len(prompt)] = gen_tokens[: len(prompt)]
        gen_prep[-(len(gen_tokens) - len(prompt)) :] = gen_tokens[
            -(len(gen_tokens) - len(prompt)) :
        ]
        # insert tokens for || in the remaining indices between
        gen_prep[gen_prep == -1] = torch.tensor(tokenizer.encode(" || "))

        cond_res = tokenizer.decode(gen_prep)
        cond_res_list.append(cond_res)

    cond_res_list = "\n\n".join(cond_res_list)

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
    with open(generation_file_path, "a") as file:
        file.write(generation_text)
    print(generation_text)
