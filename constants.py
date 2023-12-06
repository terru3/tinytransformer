## Model architecture
N_HEAD = 4
N_LAYER = 4
N_EMBD = 256
VOCAB_SIZE = 50258
SEQ_LENGTH = 384

## Data and training
BATCH_SIZE = 32
DATA_PCT = 1 # percent of training data
# ROC_DATA_PCT = 0.40 # percent of ROC benchmark data to validate on (40% is roughly same size as TinyStories val)
MAX_LR = 1e-3
# PCT_WARMUP = 0.02

## Epoch-level hyperparameters
EPOCHS = 100
SAVE_EVERY = int(EPOCHS*0.05) # save model every x epochs
GENERATE_EVERY = int(EPOCHS*0.05) # generate text from model every x epochs
# PERIOD_NUM_EPOCH = 10 # period of cosine annealing scheduler, in number of epochs

## Step-level hyperparameters——how often to compute train and validation losses
COMPUTE_PER_EPOCH = 10 # approx. number of times to print training statistics per epoch

## Model loading
# If loading, set CHECKPOINT = True and specify LOAD_EPOCH
# If training from scratch, set CHECKPOINT = False and specify LOAD_EPOCH=None
CHECKPOINT = False
LOAD_EPOCH = None
START_EPOCH = LOAD_EPOCH if LOAD_EPOCH is not None else 0

PATH = './'

MODEL_NAME = f"bt_{N_LAYER}_LAYERs_{int(DATA_PCT*100)}_DATA_PCT_{N_EMBD}_EMBD_DIM"