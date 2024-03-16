import torch
from heroicml import use_plugin
from model import GPTConfig, GPT
from data import OpenWebTextDataLoader

use_plugin("pytorch")


def model(session):
    hyperparameters = session.hyperparameters
    config = GPTConfig(**hyperparameters)
    return GPT(config)


def data(session):
    batch_size = session.hyperparameters["batch_size"]
    training_set = "data/openwebtext/train.txt"
    validation_set = "data/openwebtext/valid.txt"
    training_data = OpenWebTextDataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data = OpenWebTextDataLoader(validation_set, batch_size=batch_size, shuffle=False)
    return {"training_data": training_data, "validation_data": validation_data}


def optimizer(session):
    model = session.model
    hyperparameters = session.hyperparameters
    weight_decay = hyperparameters["weight_decay"]
    learning_rate = hyperparameters["learning_rate"]
    betas = (hyperparameters["beta1"], hyperparameters["beta2"])
    return model.configure_optimizers(weight_decay, learning_rate, betas)


loss = torch.nn.CrossEntropyLoss()


hyperparameters = {
    "gradient_accumulation_steps": 5*8,
    "batch_size": 12,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "bias": False,
    "learning_rate": 6e-4,
    "max_iters": 600000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmpup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,
}

### plugin configurations ###

# distributed plugin
# backend = "nccl"

# pytorch plugin
# device = "cuda"
# dtype = "float16"
# compile = True

# training_progress plugin
# log_interval = 1

# wandb plugin
# wandb_log = False
# wandb_project = "owt"
# wandb_run_name = "gpt2"

# loss/ eval plugin
# eval_interval = 2000
# eval_iters = 200
# eval_only = False

# checkpoints plugin
# ckpt_dir = "out"
# always_save_checkpoint = True
# init_from = "scratch"
