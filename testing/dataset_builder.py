import unittest 
from andromeda.dataset_builder import DatasetBuilder

class TestDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = DatasetBuilder(dataset_name="tiiuae/falcon-refinedweb")

    def test_initialization(self):
        self.assertEqual(self.builder.dataset_name, "tiiuae/falcon-refinedweb", "Dataset name is not correctly set.")
        self.assertEqual(self.builder.seq_len, 8192, "Sequence length is not correctly set.")
        self.assertEqual(self.builder.tokenizer, "EleutherAI/gpt-neox-20b", "Tokenizer is not correctly set.")

    def test_build_dataset(self):
        dataset = self.builder.build_dataset()
        self.assertIsNotNone(dataset, "Dataset is not built.")
        self.assertTrue(hasattr(dataset, "map"), "Dataset does not have a map method.")

    def test_tokenize_function(self):
        example = {"text": ["Hello, world!", "Andromeda is great."]}
        tokenized_example = self.builder.tokenize_function(example)
        self.assertIsInstance(tokenized_example, dict, "Tokenized example is not a dictionary.")
        self.assertTrue(all(isinstance(t, list) for t in tokenized_example.values()), "Tokenized example values are not lists.")

    def test_group_texts(self):
        examples = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 10}
        grouped_examples = self.builder.group_texts(examples)
        self.assertIsInstance(grouped_examples, dict, "Grouped examples is not a dictionary.")
        self.assertTrue(all(isinstance(t, list) for t in grouped_examples.values()), "Grouped example values are not lists.")
        self.assertTrue(all(len(t) == self.builder.seq_len for t in grouped_examples["input_ids"]), "Grouped example sequences are not the correct length.")

if __name__ == '__main__':
    unittest.main()