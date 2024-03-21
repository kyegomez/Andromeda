import torch
import unittest
from andromeda_torch.model import Andromeda


class TestAndromeda(unittest.TestCase):
    def setUp(self):
        self.model = Andromeda()

    def test_initialization(self):
        self.assertIsNotNone(
            self.model.andromeda, "Transformer is not initialized."
        )
        self.assertIsNotNone(
            self.model.decoder,
            "AutoregressiveWrapper is not initialized.",
        )

    def test_forward_pass(self):
        input_tokens = torch.randint(0, 50432, (1, 8192))
        output = self.model(input_tokens)
        self.assertIsInstance(
            output, torch.Tensor, "Output is not a PyTorch tensor."
        )
        self.assertEqual(
            output.shape[0],
            input_tokens.shape[0],
            "Output batch size does not match input.",
        )

    def test_error_handling(self):
        with self.assertRaises(Exception):
            self.model.forward(None)

    def test_model_parameters(self):
        self.assertEqual(
            self.model.Andromeda.num_tokens,
            50432,
            "Number of tokens is not correctly set.",
        )
        self.assertEqual(
            self.model.Andromeda.max_seq_len,
            8192,
            "Max sequence length is not correctly set.",
        )

    def test_model_output(self):
        input_tokens = torch.randint(0, 50432, (1, 8192))
        output1 = self.model(input_tokens)
        output2 = self.model(input_tokens)
        self.assertTrue(
            torch.allclose(output1, output2),
            "Model does not produce consistent output.",
        )


class TestAndromedaExtended(unittest.TestCase):
    def setUp(self):
        self.model = Andromeda()

    def test_input_size(self):
        for seq_len in [512, 1024, 2048, 4096]:
            input_tokens = torch.randint(0, 50432, (1, seq_len))
            output = self.model(input_tokens)
            self.assertEqual(
                output.shape[1],
                seq_len,
                (
                    "Output sequence length does not match input for"
                    f" seq_len={seq_len}."
                ),
            )

    def test_batch_size(self):
        for batch_size in [2, 4, 8, 16]:
            input_tokens = torch.randint(0, 50432, (batch_size, 8192))
            output = self.model(input_tokens)
            self.assertEqual(
                output.shape[0],
                batch_size,
                (
                    "Output batch size does not match input for"
                    f" batch_size={batch_size}."
                ),
            )

    def test_token_range(self):
        for token in [0, 50431]:
            input_tokens = torch.full((1, 8192), fill_value=token)
            output = self.model(input_tokens)
            self.assertIsInstance(
                output,
                torch.Tensor,
                f"Output is not a PyTorch tensor for token={token}.",
            )

    def test_model_depth(self):
        for depth in [16, 32, 64]:
            model = Andromeda(depth=depth)
            self.assertEqual(
                model.Andromeda.attn_layers.depth,
                depth,
                (
                    "Model depth is not correctly set for"
                    f" depth={depth}."
                ),
            )

    def test_model_dim(self):
        for dim in [1280, 2560, 5120]:
            model = Andromeda(dim=dim)
            self.assertEqual(
                model.Andromeda.attn_layers.dim,
                dim,
                (
                    "Model dimension is not correctly set for"
                    f" dim={dim}."
                ),
            )

    def test_model_heads(self):
        for heads in [12, 24, 48]:
            model = Andromeda(heads=heads)
            self.assertEqual(
                model.Andromeda.attn_layers.heads,
                heads,
                (
                    "Number of heads is not correctly set for"
                    f" heads={heads}."
                ),
            )

    def test_model_dim_head(self):
        for dim_head in [64, 128, 256]:
            model = Andromeda(dim_head=dim_head)
            self.assertEqual(
                model.Andromeda.attn_layers.dim_head,
                dim_head,
                (
                    "Head dimension is not correctly set for"
                    f" dim_head={dim_head}."
                ),
            )

    def test_model_alibi_num_heads(self):
        for alibi_num_heads in [6, 12, 24]:
            model = Andromeda(alibi_num_heads=alibi_num_heads)
            self.assertEqual(
                model.Andromeda.attn_layers.alibi_num_heads,
                alibi_num_heads,
                (
                    "Number of alibi heads is not correctly set for"
                    f" alibi_num_heads={alibi_num_heads}."
                ),
            )

    def test_model_shift_tokens(self):
        for shift_tokens in [0, 1, 2]:
            model = Andromeda(shift_tokens=shift_tokens)
            self.assertEqual(
                model.Andromeda.attn_layers.shift_tokens,
                shift_tokens,
                (
                    "Number of shift tokens is not correctly set for"
                    f" shift_tokens={shift_tokens}."
                ),
            )

    def test_model_use_abs_pos_emb(self):
        for use_abs_pos_emb in [True, False]:
            model = Andromeda(use_abs_pos_emb=use_abs_pos_emb)
            self.assertEqual(
                model.Andromeda.use_abs_pos_emb,
                use_abs_pos_emb,
                (
                    "Use absolute position embedding flag is not"
                    " correctly set for"
                    f" use_abs_pos_emb={use_abs_pos_emb}."
                ),
            )

    def test_model_alibi_pos_bias(self):
        for alibi_pos_bias in [True, False]:
            model = Andromeda(alibi_pos_bias=alibi_pos_bias)
            self.assertEqual(
                model.Andromeda.attn_layers.alibi_pos_bias,
                alibi_pos_bias,
                (
                    "Alibi position bias flag is not correctly set"
                    f" for alibi_pos_bias={alibi_pos_bias}."
                ),
            )

    def test_model_rotary_xpos(self):
        for rotary_xpos in [True, False]:
            model = Andromeda(rotary_xpos=rotary_xpos)
            self.assertEqual(
                model.Andromeda.attn_layers.rotary_xpos,
                rotary_xpos,
                (
                    "Rotary position flag is not correctly set for"
                    f" rotary_xpos={rotary_xpos}."
                ),
            )

    def test_model_attn_flash(self):
        for attn_flash in [True, False]:
            model = Andromeda(attn_flash=attn_flash)
            self.assertEqual(
                model.Andromeda.attn_layers.attn_flash,
                attn_flash,
                (
                    "Attention flash flag is not correctly set for"
                    f" attn_flash={attn_flash}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
