import torch


eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = "zinc_gpt"
wandb_run_name = "zinc_gpt"

dataset = "zinc"
gradient_accumulation_steps = 1
batch_size = 8192
block_size = 128

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 288
dropout = 0.0
vocab_size = 2048

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 10_000_000
lr_decay_iters = 1_000_000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
bias = False
warmup_iters = 100  # not super necessary potentially
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# adamw optimizer
weight_decay = 1e-1
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
