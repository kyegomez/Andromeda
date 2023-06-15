# Agora
Agora is an new open source Multi-Modality AI Research Organization devoted to advancing Humanity!

![Agora banner](agora-banner.png)

[Join our Agora discord and contribute to this project or 40+ others!](https://discord.gg/qUtxnK2NMf)


# Andromeda: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

![Andromeda Next Generation Open Source Language Model](/andromeda-banner.png)

Andromeda is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Andromeda is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.



# Usage
[label](https://console.cloud.google.com/compute/soleTenancy?authuser%3D2%26project%3Dandromeda-389913)
Get started:

1. Clone the repository and install the required packages.


```
git clone https://github.com/kyegomez/Andromeda
cd Andromeda
pip3 install -r requirements.txt
cd Andromeda
python3 training.py
```

# Training

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py``



## Dataset building building

Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the build_dataset.py script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:

```python3 Andromeda/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"```



# Inference

```python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "andromeda"``` 

Not yet we need to submit model to pytorch hub


## Model Architecture 🧠🔧

```python
model = TransformerWrapper(
        num_tokens=64007,
        max_seq_len=8192,
        use_abs_pos_emb=False,
        tokenizer=tokenizer, # !
        embedding_provider=AndromedaEmbedding(),
        attn_layers = Decoder(
            dim=128, # 2048
            depth=8, # 16
            dim_head=128,
            heads=8,
            alibi_pos_bias=True,
            alibi_num_heads=4,
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

* [Integrate Token Monster ](https://github.com/alasdairforsythe/tokenmonster)

* Establish 200k instruction sample long for Tool API Calls

* [Train on Gorilla Dataset](https://github.com/ShishirPatil/gorilla)

* Establish FineTuning scripts using quantization + 4bit precision, + other tactics like LoRA

* Establish Reinforcement Scripts to train on rewards from Human and Agent feedback





gcloud alpha compute tpus tpu-vm create v2-8 --zone=us-central1-f --version=pytorch-nightly