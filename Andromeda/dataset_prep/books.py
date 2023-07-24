# from Andromeda.dataset_builder import DatasetBuilder
from andromeda.build_dataset import DatasetBuilder

builder = DatasetBuilder(
    dataset_name="the_pile_books3",
    seq_len=8192,
    num_cpu=4,
    hf_account_repo="kye/the_pile_books3_GPTNeox-8192",
    tokenizer="EleutherAI/gpt-neox-20b",
)

dataset = builder.build_dataset()
