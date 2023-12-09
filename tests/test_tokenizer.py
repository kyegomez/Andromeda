import os
import pytest
from unittest.mock import patch, mock_open
from andromeda_torch.model import Tokenizer


# Define a fixture to create a Tokenizer instance
@pytest.fixture
def tokenizer(tmpdir):
    model_path = os.path.join(tmpdir, "tokenizer.model")
    m = mock_open()
    with patch("builtins.open", m):
        m().read.return_value = "This is a mock SentencePiece model."
        t = Tokenizer(model_path=model_path)
    return t


# Test cases for Tokenizer class
class TestTokenizer:
    def test_init_with_model_path(self, tokenizer, tmpdir):
        assert isinstance(tokenizer, Tokenizer)
        assert os.path.isfile(tokenizer.sp_model.model_file)
        assert os.path.exists(tmpdir.join("data"))

    def test_init_with_tokenizer_name(self, tmpdir):
        tokenizer_name = "hf-internal-testing/llama-tokenizer"
        t = Tokenizer(tokenizer_name=tokenizer_name)
        assert isinstance(t, Tokenizer)
        assert os.path.isfile(t.sp_model.model_file)
        assert os.path.exists(tmpdir.join("data"))

    def test_init_invalid_parameters(self):
        with pytest.raises(ValueError):
            Tokenizer()

    def test_encode(self, tokenizer):
        text = "This is a sample text"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert all(isinstance(token, int) for token in encoded)

    def test_decode(self, tokenizer):
        text = "This is a sample text"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert decoded == text

    def test_download_tokenizer(self, tmpdir):
        tokenizer_name = "hf-internal-testing/llama-tokenizer"
        model_path = Tokenizer.download_tokenizer(tokenizer_name)
        assert os.path.isfile(model_path)
        assert os.path.exists(tmpdir.join("data"))

    @patch("requests.get")
    @patch("builtins.open", mock_open())
    def test_download_tokenizer_failed_download(self, mock_requests_get):
        mock_requests_get.return_value.status_code = 404
        tokenizer_name = "invalid-tokenizer"
        with pytest.raises(Exception):
            Tokenizer.download_tokenizer(tokenizer_name)
