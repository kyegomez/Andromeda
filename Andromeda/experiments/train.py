# from torch.utils.data import DataLoader
# from accelerate import Accelerator
# from lion_pytorch import Lion
# from andromeda import Andromeda

# #set up model, tokenizer, and other configs
# tokenizer = Lion()
# dataset = ... # load alpaca dataset or books dataset


# #init accelerator object
# accelerator = Accelerator()

# #set up model
# model = andromeda()

# #create training and validation data loaders
# train_dataloader = DataLoader(dataset.train_data, batch_size=256, shuffle=True)
# valid_dataloader = DataLoader(dataset.valid_data, batch_size=256, shuffle=False)

# #move the model and data loaders to the approate device
# model, train_dataloader, vali_dataloader = accelerator.prepare(
#     model, train_dataloader, valid_dataloader
# )

# #set up lion with inverse square root 
# optimizer = Lion(model.parameters())

# #set up the training loop and run it for 1 million steps
# num_steps = 100000
# step = 0

# Import required libraries and classes
# import torch
# from torch.utils.data import Dataset, DataLoader
# from accelerate import Accelerator
# from transformers import AutoTokenizer
# from datasets import load_dataset
# from colt5_attention.transformer_block import Andromeda

# #constants
# LEARNING_RATE = 1e-4

# # Load the dataset
# raw_dataset = load_dataset('the_pile_books3')

# # Custom dataset class
# class Books3Dataset(Dataset):
#     def __init__(self, raw_dataset, tokenizer):
#         self.raw_dataset = raw_dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.raw_dataset)

#     def __getitem__(self, idx):
#         text = self.raw_dataset[idx]['text']
#         tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=4096)
#         input_ids = torch.tensor(tokens['input_ids'])
#         return input_ids

# # Set up model, tokenizer, and other configurations
# tokenizer = AutoTokenizer.from_pretrained('t5-base')
# dataset = Books3Dataset(raw_dataset['train'], tokenizer)

# # Initialize the Accelerator object
# accelerator = Accelerator()

# # Set up the model
# model = Andromeda()

# # Create training data loader
# train_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# # Move the model and data loader to the appropriate device
# model, train_dataloader = accelerator.prepare(model, train_dataloader)

# # Set up the Lion optimizer (use Adafactor as a placeholder, replace it with the Lion optimizer)
# optimizer = Lion(model.parameters(), lr=LEARNING_RATE)

# # Set up the training loop and run it for the desired number of steps
# num_steps = 1000000
# step = 0
# while step < num_steps:
#     model.train()
#     for batch in train_dataloader:
#         # Process the input data
#         inputs = accelerator.prepare(batch)

#         # Forward pass
#         outputs = model(inputs)

#         # Calculate loss
#         loss = ...  # Add your loss calculation logic here
#         loss = accelerator.backward(loss)

#         # Optimize the model
#         optimizer.step()
#         optimizer.zero_grad()

#         step += 1
#         if step >= num_steps:
#             break

#     # Validation and other tasks (logging, checkpointing, etc.) can be added here







#==========================================================+> v2

# import json
# import os
# import random
# from accelerate import Accelerator
# from torch.optim import Adam
# from transformers import DataCollatorForLanguageModeling
# from torch.utils.data import DataLoader, Dataset
# from lion_pytorch import Lion
# from andromeda import andromeda
# import torch.nn as nn
# import torch
# from transformers import DataCollatorForLanguageModeling

# class CustomDataCollator(DataCollatorForLanguageModeling):
#     def __init__(self, tokenizer, *args, **kwargs):
#         super().__init__(tokenizer, *args, **kwargs)
#         self.andromeda = andromeda
#         self.tokenizer = tokenizer
    
#     def __call__(self, examples):
#         input_ids = [self.tokenizer.encode(example['input']) for example in examples]
#         collated_data = super().__call__(input_ids)
#         collated_data["original_examples"] = examples
#         return collated_data


# # Load the Alpaca dataset
# class AlpacaDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, "r") as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Data preparation
# def preprocess_data(data):
#     instruction = data.get("instruction", "")
#     input_data = data.get("input", "")
#     output = data.get("output", "")
#     return f"{instruction}\n{input_data}\n{output}\n"


# # Prepare the data
# dataset_path = "alpaca_data.json"
# batch_size = 16
# num_epochs = 10

# # Load the dataset and create a DataLoader
# alpaca_dataset = AlpacaDataset(dataset_path)
# data_collator = CustomDataCollator(tokenizer=andromeda.tokenizer, mlm=False)
# dataloader = DataLoader(alpaca_dataset, batch_size=batch_size, collate_fn=data_collator)

# # Initialize the accelerator
# accelerator = Accelerator()
# andromeda, dataloader = accelerator.prepare(andromeda, dataloader)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss(ignore_index=andromeda.tokenizer.pad_token_id)
# optimizer = Lion(andromeda.parameters())

# # Training loop
# for epoch in range(num_epochs):
#     for step, batch in enumerate(dataloader):
#         optimizer.zero_grad()

        
#        ## Preprocess the data and run it through the model
#         texts = [''.join(preprocess_data(example)) for example in batch["original_examples"]]
#         inputs = andromeda.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
#         input_ids = inputs["input_ids"].to(accelerator.device)
#         attention_mask = inputs["attention_mask"].to(accelerator.device)
#         outputs = andromeda(input_ids, mask=attention_mask)


#         # Calculate the loss and update the model parameters
#         loss = criterion(outputs.view(-1, andromeda.vocab_size), batch["labels"].view(-1))
#         accelerator.backward(loss)
#         optimizer.step()

#         if step % 100 == 0:
#             print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

# # Save the model
# torch.save(andromeda.state_dict(), "andromeda.pth")




#==========================================================+> v3
import json
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from andromeda import Andromeda
import torch.nn as nn
import torch
from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit
from rich.progress import Progress
from andromeda import AndromedaTokenizer
from datasets import load_dataset



def count_number_of_parameters(model, only_trainable: bool = True) -> int:
    if only_trainable:
        num_params: int = sum(p.numel()
                              for p in model.parameters() if p.requires_grad)
    else:
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    return int(num_params)


def prep_sample(sample):
    instruction = sample["instruction"]
    output = sample["output"]
    return {"instruction": instruction, "output": output}

# class WizardDataset(Dataset):
#     def __init__(self, data):
#         self.tokenizer = AndromedaTokenizer()
#         self.data = self.pad_and_tokenize_sequences(data)

#     def pad_and_tokenize_sequences(self, raw_data):
#         tokenized_data = [{"input_tokens": self.tokenizer.encode(sample["instruction"]),
#                            "output_tokens": self.tokenizer.encode(sample["output"])}
#                           for sample in raw_data]
#         max_length = max(len(sample["input_tokens"]) + len(sample["output_tokens"]) for sample in tokenized_data)
#         padded_data = []
#         for sample in tokenized_data:
#             input_tokens = sample["input_tokens"]
#             output_tokens = sample["output_tokens"]
#             padded_sample = {
#                 "input_tokens": self.tokenizer.pad_sequences([input_tokens], max_length).squeeze(0),
#                 "output_tokens": self.tokenizer.pad_sequences([output_tokens], max_length).squeeze(0),
#             }
#             padded_data.append(padded_sample)
#         return padded_data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]




def train(args):
    accelerator = Accelerator(
        mixed_precision="fp16"
    )
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        accelerator.set_seed(args.seed)


    andromeda = Andromeda(dim=512,
                          num_layers=6,
                          max_seq_len=8192,
                          device=accelerator.device,
                          light_dim_head=64,
                          light_heads=8,
                          light_window_size=128,
                          heavy_dim_head=64,
                          heavy_heads=8,
                          light_ff_mult=0.5,
                          heavy_ff_mult=4,
                          num_heavy_ff_tokens=1024,
                          num_heavy_attn_tokens_q=32,
                          num_heavy_attn_tokens_kv=1024,
                          num_routed_kv=2,
                          use_triton=True,
                          use_flash_attn=True,
                          )

    dataset = load_dataset("ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered", split="train")

    dataset = dataset.map(prep_sample, num_proc=8)
    remove_columns = ["instruction", "output"]
    dataset = dataset.map(AndromedaTokenizer.tokenize, batched=True, batch_size=128, remove_columns=remove_columns)


    
    andromeda = accelerator.prepare(andromeda)

    # Load the dataset and create a DataLoader

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    andromeda, dataloader = accelerator.prepare(andromeda, dataloader)

    accelerator.print(
        f"Number of parameters: {count_number_of_parameters(andromeda):,}")
    accelerator.print(
        f"Number of trainable parameters: {count_number_of_parameters(andromeda, only_trainable=True):,}")

    # Log model and optimizer parameters to wandb
    accelerator.init_trackers(project_name="andromeda")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=andromeda.tokenizer.pad_token_id)
    optimizer = AdamW8bit(andromeda.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #learningr rate scheduler
    total_steps = args.max_steps
    warmup_steps = int(total_steps * 0.1) # set the warmup porpotion to 10% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


    # Training loop
    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=args.max_steps)
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = batch["input_tokens"].to(accelerator.device)
            attention_mask = (inputs != andromeda.tokenizer.pad_token_id).to(accelerator.device)
            labels = batch["output_tokens"].to(accelerator.device)
            outputs = andromeda(inputs, mask=attention_mask)
            
            loss = criterion(outputs.view(-1, andromeda.vocab_size), labels.view(-1))
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()  # update learning rate using the scheduler

            if step % 100 == 0:
                progress.update(task, advance=1, description=f"Step Loss: {loss.item():.5f}")
                print(f"Epoch: {step // len(dataloader)}, Step: {step}, Loss: {loss.item()}")

            if step >= args.max_steps - 1:
                break



    # Save the model
    torch.save(andromeda.state_dict(), "andromeda.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(args)