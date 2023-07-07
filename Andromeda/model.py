import torch 
from torch.nn import Module
import bitsandbytes
from optimus_prime import TransformerWrapper, AutoregressiveWrapper, AndromedaEmbedding, Decoder
from transformers import AutoTokenizer

class AndromedaTokenizer:
    def __init__(self):
        self.tokenizer= AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192
        )

    def tokenize_texts(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids





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

class AndromedaClass(Module):
    def __init__(self):
        super().__init__()
        self.embed = bitsandbytes.nn.modules.Embedding(
            320002,
            2048,
            padding_idx=1
        )

        self.output_projection = torch.nn.Linear(
            2048, 32002, bias=False
        )

        self.andromeda = TransformerWrapper(
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

        self.decoder = AutoregressiveWrapper(self.andromeda)

    def forward(self, text_tokens, **kwargs):
        model_input = self.decoder.forward_embedding(text_tokens)[0]
        return self.decoder(model_input, padded_x=model_input[0])
    

        