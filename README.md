# Andromeda: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

![Andromeda Next Generation Open Source Language Model](/andromedabanner.png)

Andromeda is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Andromeda is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.

## Model Architecture 🧠🔧

```python
model = TransformerWrapper(
    num_tokens=64007,
    max_seq_len=8192,
    use_abs_pos_emb=False,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        alibi_pos_bias=True,
        alibi_num_heads=4,
        rotary_xpos=True,
        attn_flash=True,
        deepnorm=True,
        shift_tokens=1,
        attn_one_kv_head=True,
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

### Deep Normalization (deepnorm)

Deep normalization is a technique that normalizes the activations within a layer, helping with training stability and convergence. It allows the model to better learn complex patterns and generalize to unseen data.

Usage example:

```python
attn_layers = Decoder(
    ...
    deepnorm=True,
    ...
)
```

### Training Example

Here's an example of training Andromeda with the provided code snippet:

1. Clone the repository and install the required packages.

```bash
git clone https://github.com/kyegomez/Optimus-Prime.git
cd Optimus-Prime
pip install --upgrade torch
pip install -r requirements.txt
pip install einops
```

2. Run the training script:

```bash
!python3 trainandromeda.py
```

This script will train the Andromeda model on the enwik8 dataset, leveraging the advanced techniques discussed above. The model's progress will be displayed during training, and the model will be saved periodically.

By incorporating these cutting-edge techniques, Andromeda is designed to outperform other language models like OpenAI's GPT-4 and PALM in terms of efficiency, flexibility, and scalability.

# Andromeda Principles
- **Efficiency**: Andromeda incorporates cutting-edge optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, resulting in efficient training and inference.

- **Flexibility**: The modular design of Andromeda allows for easy adaptation to various tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Andromeda's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the NLP landscape.

- **Community-driven**: As an open-source project, Andromeda thrives on contributions from the community, fostering an environment of collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟

# Join Agora
At Agora we're creating Artificial Intelligence's with the impact to potentially solve some of Humanity's biggest problems like labor, disease, and even death.

[Join us and Advance Humanity](https://discord.gg/yqQtRnCH)