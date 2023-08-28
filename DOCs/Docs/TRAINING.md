# Andromeda Model Training Standard Operating Procedure

This document provides instructions on how to train the Andromeda model end-to-end using the provided code. The training procedure consists of three main scripts: `build_dataset.py`, `model.py`, and `train_distributed.py`. Follow the steps below to train the Andromeda model.

## Prerequisites

Before starting the training process, ensure that you have the following requirements:

- Python 3.7 or higher
- PyTorch 1.9 or higher
- Transformers library
- Datasets library
- Accelerate library
- Wandb library (optional, for logging)

## Step 1: Building the Dataset

The first step is to build the dataset required for training. The `build_dataset.py` script processes the training data and prepares it for training. Follow the instructions below to build the dataset:

1. Open the `build_dataset.py` script.
2. Set the configuration parameters in the `CFG` class according to your requirements:
   - `HF_ACCOUNT_REPO`: Replace with your Hugging Face API key.
   - `TOKENIZER`: Choose the tokenizer model to use (e.g., "EleutherAI/gpt-neox-20b").
   - `DATASET_NAME`: Choose the dataset to process (e.g., "tiiuae/falcon-refinedweb").
   - `SEQ_LEN`: Set the desired sequence length.
3. Save the changes to the script.
4. Open a terminal or command prompt and navigate to the directory containing the `build_dataset.py` script.
5. Run the following command to execute the script:
   ```
   python build_dataset.py
   ```
6. The script will process the dataset and push it to your Hugging Face account repository specified by `HF_ACCOUNT_REPO`.

## Step 2: Defining the Andromeda Model

The second step is to define the Andromeda model architecture. The `model.py` script contains the model definition and configuration. Follow the instructions below to configure the Andromeda model:

1. Open the `model.py` script.
2. Set the configuration parameters in the `AndromedaTokenizer` and `Andromeda` classes according to your requirements:
   - `tokenizer`: Configure the tokenizer with the desired parameters.
   - `Andromeda`: Configure the Andromeda model with the desired architecture.
3. Save the changes to the script.

## Step 3: Training the Andromeda Model

The final step is to train the Andromeda model using the `train_distributed.py` script. Follow the instructions below to start the training process:

1. Open the `train_distributed.py` script.
2. Set the configuration parameters in the `TrainAndromeda.CFG` class according to your requirements:
   - `BATCH_SIZE`: Set the batch size for training.
   - `GRADIENT_ACCUMULATE_EVERY`: Set the number of gradient accumulation steps.
   - `LEARNING_RATE`: Set the learning rate for the optimizer.
   - `WEIGHT_DECAY`: Set the weight decay for the optimizer.
   - `SEQ_LEN`: Set the desired sequence length.
   - `USE_DEEPSPEED`: Set to `True` if using DeepSpeed for optimization.
   - `USE_FSDP`: Set to `True` if using Fully Sharded Data Parallelism.
   - `USE_PRETOKENIZED`: Set to `True` if using a pre-tokenized dataset.
   - `USE_ACTIVATION_CHECKPOINTING`: Set to `True` if using activation checkpointing.
   - `RESUME_FROM_CHECKPOINT`: Set to the path of a checkpoint to resume training from.
   - `CHECKPOINTING_STEPS`: Set the number of steps between checkpoints.
   - `OUTPUT_DIR`: Set the output directory for saving the model checkpoints and logs.
   - `ENTITY_NAME`: Set the Wandb entity name for logging (optional).
3. Save the changes to the script.
4. Open a terminal or command prompt and navigate to the directory containing the `train_distributed.py` script.
5. Run the following command to start the training:
   ```
   python train_distributed.py
   ```
6. The script will train the Andromeda model using the specified configuration and dataset.
7. During training, the progress will be displayed in the terminal, and logs will be saved to the specified output directory.

# Other Training methods

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py`


