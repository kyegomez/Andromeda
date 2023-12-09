import unittest
from andromeda.model import AndromedaTokenizer


class TestAndromedaTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AndromedaTokenizer()

    def test_initialization(self):
        self.assertIsNotNone(self.tokenizer.tokenizer, "Tokenizer is not initialized.")
        self.assertEqual(
            self.tokenizer.tokenizer.eos_token,
            "<eos>",
            "EOS token is not correctly set.",
        )
        self.assertEqual(
            self.tokenizer.tokenizer.pad_token,
            "<pad>",
            "PAD token is not correctly set.",
        )
        self.assertEqual(
            self.tokenizer.tokenizer.model_max_length,
            8192,
            "Model max length is not correctly set.",
        )

    def test_tokenize_texts(self):
        texts = ["Hello, world!", "Andromeda is great."]
        tokenized_texts = self.tokenizer.tokenize_texts(texts)
        self.assertEqual(
            tokenized_texts.shape[0],
            len(texts),
            "Number of tokenized texts does not match input.",
        )
        self.assertTrue(
            all(isinstance(t, torch.Tensor) for t in tokenized_texts),
            "Not all tokenized texts are PyTorch tensors.",
        )

    def test_decode(self):
        texts = ["Hello, world!", "Andromeda is great."]
        tokenized_texts = self.tokenizer.tokenize_texts(texts)
        decoded_texts = [self.tokenizer.decode(t) for t in tokenized_texts]
        self.assertEqual(
            decoded_texts, texts, "Decoded texts do not match original texts."
        )

    def test_len(self):
        num_tokens = len(self.tokenizer)
        self.assertIsInstance(num_tokens, int, "Number of tokens is not an integer.")
        self.assertGreater(num_tokens, 0, "Number of tokens is not greater than 0.")


if __name__ == "__main__":
    unittest.main()
