#  Andromeda

We should train an 100m param, 500m, 1billion parameters verisions with similiar hyperparameters from these 2 similiar models

[concept of mind's PALM](https://github.com/conceptofmind/PaLM)
Model Size	Num Tokens	Dim	Depth	Dim Head	Heads	Flash Attention	Learning Rate
150 M	50304	768	12	128	8	True	6e-4
410 M	50304	1024	24	128	8	True	3e-4
1 B	50304	2048	16	128	8	True	3e-4


[MPT HF](https://huggingface.co/mosaicml/mpt-7b)

Hyperparameter	Value
n_parameters	6.7B
n_layers	32
n_heads	32
d_model	4096
vocab size	50432
sequence length	2048




## Data prioritization: Prioritize datasets based on their relevance to the desired AI capabilities and the quality of the data.

High priority: C4, openwebtext, super_glue, piqa, Falcon-40B (RefinedWeb-English, RefinedWeb-Europe, Books, Conversations, Code, Technical), glue, tiiuae/falcon-refinedweb, math_dataset

Medium priority:  bigcode/ta-prompt, bigcode/the-stack-dedup, OpenAssistant/oasst1, ehartford/wizard_vicuna_70k_unfiltered, tiiuae/falcon-refinedweb

Low priority: timdettmers/openassistant-guanaco, JosephusCheung/GuanacoDataset,  JosephusCheung/GuanacoDataset, anon8231489123/ShareGPT_Vicuna_unfiltered, togethercomputer/RedPajama-Data, togethercomputer/RedPajama-Data-1T, Anthropic/hh-rlhf, databricks/databricks-dolly-15k, QingyiSi/Alpaca-CoT, alpaca,
distillation, timdettmers/openassistant-guanaco, OpenAssistant/oasst1, dmayhem93/toolformer-v0-postprocessed, openai_humaneval, yahma/alpaca-cleaned, 

## Data preprocessing: Clean, preprocess, and tokenize the datasets to ensure consistency and compatibility with the AI model.

Remove duplicates, irrelevant content, and low-quality data.

Tokenize the text using a suitable tokenizer, such as GPT Neox tokenizer or potentially falcon's tokenizer

Split the datasets into training, validation, and testing sets.


## Training strategy: Train the AI model using the prioritized datasets in a multi-stage process.

Stage 1: Pretrain the model on high-priority datasets (openwebtext, super_glue, piqa, Falcon-40B, glue) to build a strong language understanding foundation.

Stage 2: Fine-tune the model on medium-priority datasets (bigcode/ta-prompt, bigcode/the-stack-dedup, OpenAssistant/oasst1, ehartford/wizard_vicuna_70k_unfiltered, tiiuae/falcon-refinedweb) to enhance its performance in specific domains and tasks.

Stage 3: Further fine-tune the model on low-priority datasets (JosephusCheung/GuanacoDataset, anon8231489123/ShareGPT_Vicuna_unfiltered, togethercomputer/RedPajama-Data, togethercomputer/RedPajama-Data-1T, Anthropic/hh-rlhf, databricks/databricks-dolly-15k, QingyiSi/Alpaca-CoT) to capture any additional knowledge and nuances.


Evaluation and iteration: Continuously evaluate the model's performance on the validation and testing sets, and iterate the training process to improve its performance.

Monitor the model's performance using relevant metrics, such as perplexity, F1 score, or BLEU score, depending on the task.
Adjust hyperparameters, learning rate, and training duration as needed to optimize the model's performance.
If necessary, revisit the data prioritization and preprocessing steps to refine the training data.


# Evaluations and Benchmarks:

[Chain of thought hub](https://github.com/FranxYao/chain-of-thought-hub)
SFT stands for Style Fine-tuning and RLHF stands for Reinforcement Learning and Human Feedback. These are techniques used in natural language processing to improve the quality and accuracy of generated text. The statement suggests that if these techniques are applied correctly to the 65B LLaMA dataset, it is possible to recreate ChatGPT.