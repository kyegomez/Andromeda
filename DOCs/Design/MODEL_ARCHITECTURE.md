
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