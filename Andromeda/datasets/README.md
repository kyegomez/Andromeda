#  Andromeda

We should train an 100m param, 500m, 1billion parameters with the same hyperparameters as 2 similiar models

[concept of mind's PALM](https://github.com/conceptofmind/PaLM)

[MPT HF](https://huggingface.co/mosaicml/mpt-7b)




## Data prioritization: Prioritize datasets based on their relevance to the desired AI capabilities and the quality of the data.

High priority: openwebtext, super_glue, piqa, Falcon-40B (RefinedWeb-English, RefinedWeb-Europe, Books, Conversations, Code, Technical), glue

Medium priority: bigcode/ta-prompt, bigcode/the-stack-dedup, OpenAssistant/oasst1, ehartford/wizard_vicuna_70k_unfiltered, tiiuae/falcon-refinedweb

Low priority: JosephusCheung/GuanacoDataset, anon8231489123/ShareGPT_Vicuna_unfiltered, togethercomputer/RedPajama-Data, togethercomputer/RedPajama-Data-1T, Anthropic/hh-rlhf, databricks/databricks-dolly-15k, QingyiSi/Alpaca-CoT

## Data preprocessing: Clean, preprocess, and tokenize the datasets to ensure consistency and compatibility with the AI model.

Remove duplicates, irrelevant content, and low-quality data.

Tokenize the text using a suitable tokenizer, such as BPE or SentencePiece.

Split the datasets into training, validation, and testing sets.

## Model architecture: Choose a suitable transformer architecture, such as GPT-3, BERT, or RoBERTa, depending on the desired AI capabilities (e.g., language generation, classification, or question-answering).

## Training strategy: Train the AI model using the prioritized datasets in a multi-stage process.

Stage 1: Pretrain the model on high-priority datasets (openwebtext, super_glue, piqa, Falcon-40B, glue) to build a strong language understanding foundation.

Stage 2: Fine-tune the model on medium-priority datasets (bigcode/ta-prompt, bigcode/the-stack-dedup, OpenAssistant/oasst1, ehartford/wizard_vicuna_70k_unfiltered, tiiuae/falcon-refinedweb) to enhance its performance in specific domains and tasks.

Stage 3: Further fine-tune the model on low-priority datasets (JosephusCheung/GuanacoDataset, anon8231489123/ShareGPT_Vicuna_unfiltered, togethercomputer/RedPajama-Data, togethercomputer/RedPajama-Data-1T, Anthropic/hh-rlhf, databricks/databricks-dolly-15k, QingyiSi/Alpaca-CoT) to capture any additional knowledge and nuances.


Evaluation and iteration: Continuously evaluate the model's performance on the validation and testing sets, and iterate the training process to improve its performance.

Monitor the model's performance using relevant metrics, such as perplexity, F1 score, or BLEU score, depending on the task.
Adjust hyperparameters, learning rate, and training duration as needed to optimize the model's performance.
If necessary, revisit the data prioritization and preprocessing steps to refine the training data.



