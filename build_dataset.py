import multiprocessing
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

from huggingface_hub import HfApi


class DatasetBuilder:
    def __init__(
        self,
        dataset_name,
        seq_len=8192,
        num_cpu=None,
        hf_account_repo=None,
        tokenizer="EleutherAI/gpt-neox-20b",
    ):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.num_cpu = num_cpu or multiprocessing.cpu_count()
        self.hf_account_repo = hf_account_repo
        self.tokenizer = tokenizer

    def build_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        train_dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        dataset = train_dataset.shuffle()

        def tokenize_function(example):
            return tokenizer([t + tokenizer.eos_token for t in example["text"]])

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            # num_proc=self.num_cpu,
            remove_columns=["text"],
            #num_proc=32
        )

        block_size = self.seq_len

        def group_texts(examples):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size

            result = {
                k: [
                    t[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_examples.items()
            }

            return result

        train_tokenized_dataset = tokenized_dataset.map(
            group_texts, batched=True, #num_proc=32#num_proc=self.num_cpu
        )

        #TODO: ValueError: path_or_fileobj must be either an instance of str, bytes or io.BufferedIOBase. If you passed a file-like object, make sure it is in binary mode.
        if self.hf_account_repo:
            # train_tokenized_dataset.push_to_hub(self.hf_account_repo, private=True)
            hf_api = HfApi()
            hf_api.upload_file(
                path_or_fileobj= "TOKENIZED_DATASET",#train_tokenized_dataset, #path to local space
                path_in_repo="README.md",
                repo_id=self.hf_account_repo,
                repo_type="dataset"
            )

        return train_tokenized_dataset

            

builder = DatasetBuilder(
    dataset_name="the_pile_books3",
    seq_len=8192,
    # num_cpu=4,
    hf_account_repo="kye/thepilebooks3-gptneox-8k",
    tokenizer="EleutherAI/gpt-neox-20b",
)

dataset = builder.build_dataset()