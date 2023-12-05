from constants import *
from generate import generate_inference
from model import BetterTransformer
from train import get_dataloaders, prep_train, load, train
from utils import set_seed, device, collate_wrapper

## Step-level hyperparameters——how often to compute train and validation losses——appear later
## Those require the data to be loaded in first to compute number of steps in a batch

def main():
    set_seed()
    print("Model Name:", MODEL_NAME)
    print(f'Model will be saved every {SAVE_EVERY} epochs, and will generate text every {GENERATE_EVERY} epochs')

    train_dataloader, val_dataloader = get_dataloaders()
    model, tokenizer, optimizer, TRAIN_EVERY = prep_train(train_dataloader)
    if CHECKPOINT:
        model, optimizer, train_losses, val_losses = load(model, optimizer)
        train(model, tokenizer, train_dataloader, val_dataloader, device, optimizer, TRAIN_EVERY,
                train_losses, val_losses)
    else:
        train(model, tokenizer, train_dataloader, val_dataloader, device, optimizer, TRAIN_EVERY)
    
    # # Test trained model
    # generate_inference(model, tokenizer, device, method='multinomial')
    # generate_inference(model, tokenizer, device, method='top-k', k=5)
    # generate_inference(model, tokenizer, device, method='top-k', k=5, cond='One morning, Jack woke up and got out of')

if __name__ == "__main__":
   main()