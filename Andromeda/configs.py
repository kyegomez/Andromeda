from Andromeda.model import Andromeda


Andromeda1Billion = Andromeda(
    num_tokens=25000,
    max_seq_len=4192,
    dim=2048,
    depth=16,
    dim_head=128,
    heads=8,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=4, 
    rotary_xpos=True,
    attn_flash=True,  
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)



Andromeda3Billion = Andromeda(
    num_tokens=50432,
    max_seq_len=8192,
    dim=3072,
    depth=24,
    dim_head=128,
    heads=12,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=6, 
    rotary_xpos=True, 
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)



Andromeda7Billion = Andromeda(
    num_tokens=50432,
    max_seq_len=8192,
    dim=4096,
    depth=32,
    dim_head=128,
    heads=16,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=8, 
    rotary_xpos=True, 
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)

Andromeda10Billion = Andromeda(
    num_tokens=50432,
    max_seq_len=8192,
    dim=5120,
    depth=32,
    dim_head=128,
    heads=20,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=4, 
    rotary_xpos=True, 
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)

Andromeda15Billion = Andromeda(
    num_tokens=50432,
    max_seq_len=8192,
    dim=6144,
    depth=40,
    dim_head=128,
    heads=24,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=4, 
    rotary_xpos=True, 
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)

Andromeda20Billion = Andromeda(
    num_tokens=50432,
    max_seq_len=8192,
    dim=7168,
    depth=48,
    dim_head=128,
    heads=28,
    use_abs_pos_emb=False, 
    alibi_pos_bias=True, 
    alibi_num_heads=4, 
    rotary_xpos=True, 
    attn_kv_heads = 2, 
    qk_norm=True, 
    attn_qk_norm=True, 
    attn_qk_norm_dim_scale=True, 
)

#to GPT like 176Billion Parameters 122888 dimension, 96 depth, 96 heads, attn dim head 128