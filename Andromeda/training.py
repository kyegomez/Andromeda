import math
import multiprocessing
import os
from datetime import timedelta
from functools import partial
from itertools import chain

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from accelerate import Accelerator
from accelerate.utils import (DummyOptim, DummyScheduler,
                              InitProcessGroupKwargs)
from datasets import concatenate_datasets, load_dataset
from lion_pytorch import Lion
# from palm_rlhf_pytorch import PaLM
from torch.nn import LayerNorm
# from palm_rlhf_pytorch.palm import LayerNorm, TransformerWrapper

from torch.nn import LayerNorm
from optimus_prime import TransformerWrapper, AutoregressiveWrapper, AndromedaEmbedding, Decoder

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)


from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, default_data_collator,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

# from palm.stable_adamw import StableAdamWUnfused
from utils.stable_adamw import StableAdamWUnfused

from optimus_prime import TransformerWrapper, AutoregressiveWrapper, AndromedaEmbedding, Decoder


class TrainAndromeda:
    class CFG:
        BATCH_SIZE = 3
        GRADIENT_ACCUMULATE_EVERY: int = 1
        SEED: int = 42
        LEARNING_RATE: float = 3e-4
        WEIGHT_DECAY: float = 0.1
        SEQ_LEN: int = 8192
        NUM_CPU: int = multiprocessing.cpu_count()
        USE_DEEPSPEED: bool = True
        USE_FSDP: bool = True
        USE_PRETOKENIZED: bool = True
        USE_ACTIVATION_CHECKPOINTING: bool = True
        RESUME_FROM_CHECKPOINT: str = True
        CHECKPOINTING_STEPS: int = 1000
        OUTPUT_DIR: str = "YOUR_OUTPUT_DIR"
        ENTITY_NAME: str = "YOUR_ENTITY_NAME"

    @staticmethod
    def print_num_params(model, accelerator: Accelerator):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"Number of parameters in model: {n_params}")

    @staticmethod
    def activation_checkpointing(
        model: torch.nn.Module,
        offload_to_cpu: bool = False,
        accelerator: Accelerator = None,
    ):
        if accelerator is not None:
            accelerator.print(f"Using activation checkpointing")
        check_fn = lambda submodule: isinstance(submodule, TransformerWrapper)
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=offload_to_cpu,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    @staticmethod
    def fsdp(
        model: torch.nn.Module,
        auto_wrap: bool = False,
        mp: str = "fp32",
        shard_strat: str = "NO_SHARD",
    ):
        if auto_wrap:
            andromeda_auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    TransformerWrapper,
                },
            )
        else:
            andromeda_auto_wrap_policy = None

        if mp == "bf16":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif mp == "fp16":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif mp == "fp32":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'bf16', 'fp16' or 'fp32', got: {}".format(
                    mp
                )
            )

        if shard_strat == "SHARD_GRAD":
            sharding_strat_fsdp = ShardingStrategy.SHARD_GRAD_OP 
        elif shard_strat == "FULL_SHARD":
            sharding_strat_fsdp = ShardingStrategy.FULL_SHARD
        elif shard_strat == "NO_SHARD":
            sharding_strat_fsdp = ShardingStrategy.NO_SHARD
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'SHARD_GRAD', 'FULL_SHARD' or 'NO_SHARD', got: {}".format(
                    shard_strat
                )
            )

        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=andromeda_auto_wrap_policy,
            mixed_precision=mp_fsdp,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sharding_strategy=sharding_strat_fsdp,
            forward_prefetch=True,
            use_orig_params=True,
        )

        return model

    @staticmethod
    def get_lr_scheduler_with_warmup(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        num_warmup_steps: int,
        max_train_steps: int,
        grad_accumulate_every: int = 1,
        accelerator: Accelerator = None,
    ):
        NUM_WARMUP_STEPS = num_warmup_steps
        GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every
        if accelerator is not None:
            accelerator.print(f"Using {scheduler_type} lr scheduler")
        if scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
                num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
            )
        elif scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
                num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    scheduler_type
                )
            )

    @staticmethod
    def decoupled_optimizer(
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        beta_1: float,
        beta_2: float,
        optimizer_type: str,
        use_fsdp: bool = True,
        accelerator: Accelerator = None,
    ):
        if optimizer_type == "lion":
            optimizer = Lion(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type == "adamw":
            optimizer = AdamW(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type == "deepspeed":
            optimizer = DummyOptim(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type =="stable_adamw":
            optimizer = StableAdamWUnfused(
                grouped_params, lr=learning_rate, betas=(beta_1, beta_2),
            )
        else:
            raise ValueError(
                "Invalid optimizer_type. Expected 'lion', 'adamw', 'deepspeed' or 'stable_adamw', got: {}".format(
                    optimizer_type
                )
            )

        return optimizer

    @staticmethod
    def build_dataloaders():
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        dataset = load_dataset("openwebtext", split="train")

        tokenized_dataset = dataset.map(
            lambda example: tokenizer([t + tokenizer.eos_token for t in example["text"]]),
            batched=True,
            num_proc=TrainAndromeda.CFG.NUM_CPU,
            remove_columns=["text"],
        )

        block_size = TrainAndromeda.CFG.SEQ_LEN

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        train_dataset = tokenized_dataset.map(
            group_texts, batched=True, num_proc=TrainAndromeda.CFG.NUM_CPU,
        )

        return train_dataset

    @staticmethod
    def build_pre_tokenized():
        d0 = load_dataset("conceptofmind/c4_0-to-20_neox_with_eos_8k", split="train")
        return d0

    @staticmethod
    def Train():
        timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
        accelerator = Accelerator(
            gradient_accumulation_steps=TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
            mixed_precision="fp16",
            log_with="wandb",
            kwargs_handlers=[timeout],
        )
        accelerator.init_trackers(
            project_name="Andromeda",
            config={
                "batch_size": TrainAndromeda.CFG.BATCH_SIZE,
                "gradient_accumulate_every": TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
                "learning_rate": TrainAndromeda.CFG.LEARNING_RATE,
                "seq_len": TrainAndromeda.CFG.SEQ_LEN,
            },
            init_kwargs={"wandb": {"entity": TrainAndromeda.CFG.ENTITY_NAME}},
        )
        accelerator.print(f"Total GPUS: {accelerator.num_processes}")
        set_seed(TrainAndromeda.CFG.SEED)
        model = TransformerWrapper(
            num_tokens=64007,
            max_seq_len=8192,
            use_abs_pos_emb=False,
            embedding_provider=AndromedaEmbedding(),
            attn_layers = Decoder(
                dim=2560,
                depth=32,
                dim_head=128,
                heads=24,
                alibi_pos_bias=True,
                alibi_num_heads=12,
                rotary_xpos=True,
                attn_flash=True,
                deepnorm=True,
                shift_tokens=1,
                attn_one_kv_head=True,
                qk_norm=True,
                attn_qk_norm=True,
                attn_qk_norm_dim_scale=True
            )
        ).to(accelerator.device)
        model = AutoregressiveWrapper(model).to(accelerator.device)
        TrainAndromeda.print_num_params(model, accelerator)

        if TrainAndromeda.CFG.USE_FSDP:
            model = TrainAndromeda.fsdp(
                model,
                mp="fp16",
                shard_strat="SHARD_GRAD"
            )

        if TrainAndromeda.CFG.USE_ACTIVATION_CHECKPOINTING:
            TrainAndromeda.activation_checkpointing(model, accelerator)

        model = accelerator.prepare(model)

        if TrainAndromeda.CFG.USE_PRETOKENIZED:
            train_dataset = TrainAndromeda.build_pre_tokenized()
        else:
            train_dataset = TrainAndromeda.build_dataloaders()

        train_loader = DataLoader(
            train_dataset, batch_size=TrainAndromeda.CFG.BATCH_SIZE, collate_fn=default_data_collator,
        )

        optim = TrainAndromeda.decoupled_optimizer(
            model=model,
            learning_rate=TrainAndromeda.CFG.LEARNING_RATE, 
            weight_decay=TrainAndromeda.CFG.WEIGHT_DECAY, 
            beta_1=0.90, 
            beta_2=0.95, 
            optimizer_type='deepspeed',  
            use_fsdp=True,
            accelerator=accelerator
        )

        max_train_steps = math.ceil(len(train_loader) / TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY)
        accelerator.print(f"Max train steps: {max_train_steps}")

        NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
        accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

        if TrainAndromeda.CFG.USE_DEEPSPEED:
            lr_scheduler = DummyScheduler(
                optim, 
                total_num_steps=max_train_steps * accelerator.num_processes, 
                warmup_num_steps=NUM_WARMUP_STEPS
            )
        else:
            lr_scheduler = TrainAndromeda.get_lr_scheduler_with_warmup(
                optimizer=optim,
                scheduler_type="cosine",
                num_warmup_steps=NUM_WARMUP_STEPS,
                max_train_steps=max_train_steps,
                grad_accumulate_every=TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
            )

        optim, train_loader, lr_scheduler = accelerator.prepare(
            optim, train_loader, lr_scheduler
        )

        accelerator.register_for_checkpointing(lr_scheduler)

        max_train_steps = math.ceil(len(train_loader) / TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY)
        accelerator.print(f"Max train steps recalculated: {max_train_steps}")

        total_batch_size = (
            TrainAndromeda.CFG.BATCH_SIZE * accelerator.num_processes * TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY
        )
        accelerator.print(f"Total batch size: {total_batch_size}")

        progress_bar = tqdm(
            range(max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT:
            if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT is not None or TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT != "":
                accelerator.print(f"Resuming from checkpoint {TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT}")
                accelerator.load_state(TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT)
                path = os.path.basename(TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT)
            training_difference = os.path.splitext(path)[0]

            resume_step = (
                int(training_difference.replace("step_", ""))
                * TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY
            )

        if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
            train_loader = accelerator.skip_first_batches(train_loader, resume_step)
            completed_steps += resume_step
            progress_bar.update(resume_step)

        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                inputs = batch["input_ids"].to(accelerator.device)
                loss = model(inputs, return_loss=True)
                accelerator.backward(loss)

                accelerator.log({"loss": loss.item()}, step=step)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(TrainAndromeda.CFG.CHECKPOINTING_STEPS, int):
                if completed_steps % TrainAndromeda.CFG.CHECKPOINTING_STEPS == 0:
                    output_dir = f"step_{completed_steps }"
                    if TrainAndromeda.CFG.OUTPUT_DIR is not None:
                        output_dir = os.path.join(TrainAndromeda.CFG.OUTPUT_DIR, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        accelerator.end_training()

        if TrainAndromeda.CFG.OUTPUT_DIR is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            with accelerator.main_process_first():
                accelerator.save(
                    unwrapped_model.state_dict(), f"{TrainAndromeda.CFG.OUTPUT_DIR}/final/final_model.pt"
                )

if __name__ == "__main__":
    TrainAndromeda.Train()
