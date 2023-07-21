
# [READY FOR TRAINING, help us with the strategy!](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Andromeda-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

# Agora
Agora is an new open source Multi-Modality AI Research Organization devoted to advancing Humanity to a post-scarcity state!

Since Andromeda is ready to train Agora is actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at `kye@apac.ai`

![Agora banner](agora-banner.png)

![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)


---

# Andromeda: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

![Andromeda Next Generation Open Source Language Model](/andromeda-banner.png)

Andromeda is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Andromeda is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM with features like:

* Process Ultra Long Sequences of 32,000-200,000+ context lengths effortlessly :fire: :closed_book: 

* Process those Ultra Long Sequences Ultra Fast with 32,000+ tokens in under 100ms ⚡️ ⚡️ 

* Reliable and actually useful with superior reasoning capabilities :brain: :brain: 

---

<div align="center">

[![Open Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Fconsole.algora.io%2Fapi%2Fshields%2Fkyegomez%2Fbounties%3Fstatus%3Dopen)](https://console.algora.io/org/kyegomez/bounties?status=open)
[![Rewarded Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Fconsole.algora.io%2Fapi%2Fshields%2Fkyegomez%2Fbounties%3Fstatus%3Dcompleted)](https://console.algora.io/org/kyegomez/bounties?status=completed)

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Andromeda)](https://github.com/kyegomez/Andromeda/blob/main/LICENSE)

</div>

<div align="center">

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Andromeda)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20Andromeda&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&title=&summary=&source=)


![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&title=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&t=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Andromeda%20-%20the%20next%20generation%20AI%20shields) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Andromeda%20-%20the%20next%20generation%20AI%20shields%20%23Andromeda%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FAndromeda)

</div>

---




## Table of Contents

- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing to Andromeda](#contributing-to-andromeda)
- [Roadmap](#roadmap)

---

## Usage

There are two methods to use Andromeda but pip does not work now.

1. Clone the repository.

For detailed instructions, refer to the [Training SOP](DOCs/TRAINING.md) and [Documentation](https://github.com/kyegomez/Andromeda/blob/master/DOCs/DOCUMENTATION.md).

## Documentation

For detailed documentation, click [here](https://github.com/kyegomez/Andromeda/blob/master/DOCs/DOCUMENTATION.md).

### Method 1

To get started:

1. Clone the repository and install the required packages:

```
git clone https://github.com/kyegomez/Andromeda
cd Andromeda
pip3 install -r requirements.txt
cd Andromeda
python3 train.py
```

For further instructions, refer to the [Training SOP](DOCs/TRAINING.md).

## Training

1. Set the environment variables:
   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Directory to save the weights (e.g., `./weights`)

2. Configure the training:
   - Accelerate Config
   - Enable Deepspeed 3
   - Accelerate launch train_distributed_accelerate.py

For more information, refer to the [Training SOP](DOCs/TRAINING.md).

## Dataset Building

To preprocess a different dataset similar to the C4 dataset used during training, use the `build_dataset.py` script. This script pre-tokenizes the data, chunks it into blocks of a specified sequence length, and uploads it to the Huggingface hub.

Example command:

```python
python3 Andromeda/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"
```

## Inference

```python
python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "andromeda"
```

(Note: Model submission to PyTorch Hub is still pending.)

## Why Andromeda?

Andromeda offers several advantages:

- And romeda can potentially be fine-tuned with a 100k+ token sequence length.
- It incorporates advanced techniques, including alibi positional bias, rotary position encodings (xpos), flash attention, and deep normalization (deepnorm), to optimize performance and efficiency.

For detailed information about the model architecture and methods, refer to the [Model Architecture](DOCs/MODEL_ARCHITECTURE.md) documentation.

## Andromeda Principles

Andromeda is built on key principles:

- **Efficiency**: Andromeda leverages optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, to achieve efficient training and inference.
- **Flexibility**: The modular design of Andromeda allows easy adaptation to various tasks and domains, making it versatile for a wide range of applications.
- **Scalability**: Andromeda's architecture is designed to scale with increasing computational resources and data sizes, ensuring its relevance in the NLP landscape.
- **Community-driven**: As an open-source project, Andromeda thrives on contributions from the community, fostering collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟


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

As part of our growth strategy, we will be deploying Andromeda as a Paid API. The revenue generated from this API will not only sustain and further the project, but also fund the rewards program. If you contribute anything to make Andromeda you will receive recurring revenue from paid API requests!

## How to Start Contributing

If you're ready to become a part of Andromeda and contribute to the future of multimodal embeddings, here's what you need to do:

1. Fork the repository.

2. Make your improvements or additions in your forked repository.

3. Submit a pull request detailing the changes you've made.

4. Our team will review your submission. If it's approved, it will be merged into the main repository, and you will become part of the Andromeda Rewards Program.

Thank you for considering contributing to Andromeda. Your expertise and commitment to this project are what make it thrive. Let's build the future of multimodal embeddings together.

## Roadmap 🗺️📍

1. **Training phase**: Train Andromeda on a large-scale dataset to achieve SOTA performance in various natural language processing tasks.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure that leverages techniques such as:

   - Model quantization: Reduce memory and computational requirements without significant loss in performance.
   - Distillation: Train smaller, faster models that retain the knowledge of the larger model.
   - Optimized serving frameworks: Deploy Andromeda using efficient serving frameworks, such as NVIDIA Triton or TensorFlow Serving, for rapid inference.

3. **Continuous improvement**: Continuously fine-tune Andromeda on diverse data sources and adapt it to new tasks and domains.

4. **Community-driven development**: Encourage open-source contributions, including pre-processing improvements, advanced training techniques, and novel use cases.


## Todo:

* [Create the best and most validated, reliable training => finetuning strategy](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Andromeda-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

* Establish FineTuning scripts using quantization + 4bit precision, + other tactics like LoRA

* Establish 200k instruction sample long for Tool API Calls

* [Train on Gorilla Dataset](https://github.com/ShishirPatil/gorilla)


* Establish Reinforcement Scripts to train on rewards from Human and Agent feedback













