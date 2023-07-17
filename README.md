# [READY FOR TRAINING, help us with the strategy!](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Andromeda-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

# Agora
Agora is an new open source Multi-Modality AI Research Organization devoted to advancing Humanity!

Since Andromeda is ready to train Agora is actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at `kye@apac.ai`


![Agora banner](agora-banner.png)

[Join our Agora discord and contribute to this project or 40+ others!](https://discord.gg/qUtxnK2NMf)


# Andromeda: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

![Andromeda Next Generation Open Source Language Model](/andromeda-banner.png)

Andromeda is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Andromeda is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.

---

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/blob/main/LICENSE)

</div>

<div align="center">

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Andromeda)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20Andromeda&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&title=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&t=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Andromeda%20-%20the%20next%20generation%20AI%20shields%20%23Andromeda%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda)

</div>

----


# Usage
There are 2 methods to use Andromeda, 1 by `pip install Andromeda-llm` and the other by `git clone`. [Head over to the Training SOP for more](DOCs/TRAINING.md) and Head over to the Documentation for more!

# Documentation
* [Click here for the documentation.](https://github.com/kyegomez/Andromeda/blob/master/DOCs/DOCUMENTATION.md)

# Method1
First `pip install Andromeda-llm` then

```python
import torch
from Andromeda import Andromeda, Train


x = torch.randint(0, 20000, (1, 1024))

Andromea(x)

# or train

Train()

```


## Method 2

Get started:

1. Clone the repository and install the required packages.


```
git clone https://github.com/kyegomez/Andromeda
cd Andromeda
pip3 install -r requirements.txt
cd Andromeda
python3 training_distributed.py
```

# Training

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py`

# Environment variables

* `ENTITY_NAME` ==> Your wandb project name

* `OUTPUT_DIR` ==> Where you want the weights to go when it's finished training for example inside the root directory you can do something like: `./weights` and it'll create a folder called weights INSIDE of the Andromeda folder

## Dataset building building

Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the build_dataset.py script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:

```python3 Andromeda/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"```



# Inference

```python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "andromeda"``` 

Not yet we need to submit model to pytorch hub


## Get Involved

We're just at the beginning of our journey. As we continue to develop and refine Andromeda, we invite you to join us. Whether you're a developer, researcher, or simply an enthusiast, your insights and contributions can help shape the future of Andromeda.

# Contributing to Andromeda

We are thrilled to invite you to be a part of the Andromeda project. This is not just an open source project but a community initiative, and we value your expertise and creativity. To show our appreciation, we have instituted a unique rewards system that directly compensates contributors from the revenue generated by the Andromeda API.

## Why Contribute

Contributing to Andromeda not only enhances your skills and profile but also comes with financial rewards. When you contribute code, documentation, or any form of improvement to the Andromeda project, you are adding value. As such, we believe it's only fair that you share in the rewards.

## Rewards Program

Here's how the Andromeda Rewards Program works:

1. **Submit a Pull Request:** This can be a code enhancement, bug fix, documentation update, new feature, or any improvement to the project.

2. **Review and Approval:** Our team will review your contribution. If it gets approved and merged, you become eligible for the rewards program.

3. **Revenue Share:** Once your pull request is merged, you will receive a percentage of the revenue generated by the Andromeda API. The percentage will be determined based on the significance and impact of your contribution. 

This means you're not just contributing to an open source project; you're becoming a part of the Andromeda ecosystem. Your efforts can yield ongoing benefits as the Andromeda API grows and evolves.

## Becoming a Paid API

As part of our growth strategy, we will be deploying Andromeda as a Paid API. The revenue generated from this API will not only sustain and further the project, but also fund the rewards program.

## How to Start Contributing

If you're ready to become a part of Andromeda and contribute to the future of multimodal embeddings, here's what you need to do:

1. Fork the repository.

2. Make your improvements or additions in your forked repository.

3. Submit a pull request detailing the changes you've made.

4. Our team will review your submission. If it's approved, it will be merged into the main repository, and you will become part of the Andromeda Rewards Program.

Thank you for considering contributing to Andromeda. Your expertise and commitment to this project are what make it thrive. Let's build the future of multimodal embeddings together.


## Model Architecture 🧠🔧

```python
Andromeda = TransformerWrapper(
    num_tokens=64007,
    max_seq_len=8192,
    use_abs_pos_emb=False,
    # tokenizer=tokenizer,
    embedding_provider=AndromedaEmbedding(),
    attn_layers = Decoder(
        dim=2560, # 2048
        depth=32, # 16
        dim_head=128,
        heads=24,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash = True,
        deepnorm=True,
        shift_tokens=1,
        attn_one_kv_head = True,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
    )
)

Andromeda = AutoregressiveWrapper(Andromeda)
```

## Roadmap 🗺️📍

1. **Training phase**: Train Andromeda on a large-scale dataset to achieve SOTA performance in various natural language processing tasks.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure that leverages techniques such as:

   - Model quantization: Reduce memory and computational requirements without significant loss in performance.
   - Distillation: Train smaller, faster models that retain the knowledge of the larger model.
   - Optimized serving frameworks: Deploy Andromeda using efficient serving frameworks, such as NVIDIA Triton or TensorFlow Serving, for rapid inference.

3. **Continuous improvement**: Continuously fine-tune Andromeda on diverse data sources and adapt it to new tasks and domains.

4. **Community-driven development**: Encourage open-source contributions, including pre-processing improvements, advanced training techniques, and novel use cases.

## Why Andromeda? 🌠💡

Andromeda can potentially be finetuned with 100k+ token sequence length.
Andromeda is a state-of-the-art language model that leverages advanced techniques to optimize its performance and efficiency. Some of these techniques include alibi positional bias, rotary position encodings (xpos), flash attention, and deep normalization (deepnorm). Let's explore the benefits of these techniques and provide some usage examples.

### Alibi Positional Bias

Alibi positional bias allows the model to learn relative positions between tokens, enabling it to better capture the relationships and dependencies between tokens in a sequence.

Usage example:

```python
attn_layers = Decoder(
    ...
    alibi_pos_bias=True,
    alibi_num_heads=4,
    ...
)
```

### Rotary Position Encodings (xpos)

Rotary position encodings introduce a more efficient way to encode positions in the input sequence. They avoid the need for absolute positional embeddings, reducing the model's memory footprint and improving training speed.

Usage example:

```python
attn_layers = Decoder(
    ...
    rotary_xpos=True,
    ...
)
```

### Flash Attention

Flash attention speeds up the self-attention mechanism by reducing the number of attention computations. It accelerates training and inference while maintaining a high level of performance.

Usage example:

```python
attn_layers = Decoder(
    ...
    attn_flash=True,
    ...
)
```

Usage example:

```python
attn_layers = Decoder(
    ...
    deepnorm=True,
    ...
)
```

### Deep Normalization (deepnorm)

Deep normalization is a technique that normalizes the activations within a layer, helping with training stability and convergence. It allows the model to better learn complex patterns and generalize to unseen data.

# Andromeda Principles
- **Efficiency**: Andromeda incorporates cutting-edge optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, resulting in efficient training and inference.

- **Flexibility**: The modular design of Andromeda allows for easy adaptation to various tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Andromeda's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the NLP landscape.

- **Community-driven**: As an open-source project, Andromeda thrives on contributions from the community, fostering an environment of collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟

## Todo:

* [Create the best and most validated, reliable training => finetuning strategy](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Andromeda-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

* [Integrate Token Monster ](https://github.com/alasdairforsythe/tokenmonster)

* Establish 200k instruction sample long for Tool API Calls

* [Train on Gorilla Dataset](https://github.com/ShishirPatil/gorilla)

* Establish FineTuning scripts using quantization + 4bit precision, + other tactics like LoRA

* Establish Reinforcement Scripts to train on rewards from Human and Agent feedback





