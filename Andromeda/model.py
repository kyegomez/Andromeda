from optimus_prime import TransformerWrapper, AndromedaEmbedding, Decoder, AutoregressiveWrapper

from tokenizer import AndromedaTokenizer

andromeda_tokenizer = AndromedaTokenizer()

andromeda_model = TransformerWrapper(
    num_tokens=len(andromeda_tokenizer),
    max_seq_len=8192,
    use_abs_pos_emb=False,

    embedding_provider=AndromedaEmbedding(),

    attn_layers = Decoder(
        dim=32, # 2560
        depth=2, # 32
        dim_head=8, # 128
        heads=4, # 24
        alibi_pos_bias=True,
        alibi_num_heads=4, # 12
        rotary_xpos=False, # Why?!
        attn_flash=True,
        deepnorm=False,
        shift_tokens=1,
        attn_one_kv_head=True,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True
    )
)

andromeda_model = AutoregressiveWrapper(andromeda_model)
