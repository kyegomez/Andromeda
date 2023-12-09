import torch
import pytest
from andromeda_torch.model import Andromeda


# Create a fixture to initialize the Andromeda model
@pytest.fixture
def andromeda_model():
    return Andromeda()


# Test cases for Andromeda class
class TestAndromeda:
    def test_init_with_default_parameters(self, andromeda_model):
        assert isinstance(andromeda_model, Andromeda)

    def test_forward_pass(self, andromeda_model):
        # Create a sample input tensor
        input_tokens = torch.randint(0, 100, (16, 128))

        # Ensure the forward pass works without errors
        with torch.no_grad():
            output = andromeda_model(input_tokens)

        assert isinstance(output, torch.Tensor)

    def test_invalid_parameters(self):
        # Test initialization with invalid parameters
        with pytest.raises(Exception):
            Andromeda(num_tokens="invalid")

    def test_forward_pass_with_invalid_input(self, andromeda_model):
        # Test forward pass with invalid input
        with pytest.raises(Exception):
            andromeda_model("invalid_input")

    def test_custom_parameters(self):
        # Test initialization with custom parameters
        custom_params = {
            "num_tokens": 1000,
            "max_seq_len": 512,
            "dim": 1024,
            "depth": 16,
            "dim_head": 64,
            "heads": 8,
            "use_abs_pos_emb": True,
            "alibi_pos_bias": False,
        }
        custom_andromeda = Andromeda(**custom_params)

        assert isinstance(custom_andromeda, Andromeda)

    def test_forward_pass_with_custom_input(self, andromeda_model):
        # Test forward pass with custom input
        custom_input = torch.randn(16, 128, 256)  # Custom input shape
        with torch.no_grad():
            output = andromeda_model(custom_input)

        assert isinstance(output, torch.Tensor)

    def test_model_parameters(self, andromeda_model):
        # Ensure model parameters are accessible and correct
        model_parameters = andromeda_model.parameters()
        assert all(isinstance(param, torch.Tensor) for param in model_parameters)
        assert all(param.requires_grad for param in model_parameters)

    def test_model_training(self, andromeda_model):
        # Test model training with a simple task (e.g., regression)
        optimizer = torch.optim.Adam(andromeda_model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        # Generate random input and target data
        input_data = torch.randn(32, 128)
        target_data = torch.randn(32, 128)

        for _ in range(10):
            optimizer.zero_grad()
            predictions = andromeda_model(input_data)
            loss = loss_fn(predictions, target_data)
            loss.backward()
            optimizer.step()

        assert loss.item() < 0.1  # Ensure the model learns

    def test_model_serialization(self, andromeda_model, tmpdir):
        # Test model serialization and deserialization
        save_path = str(tmpdir.join("andromeda_model.pth"))
        torch.save(andromeda_model.state_dict(), save_path)

        loaded_model = Andromeda()
        loaded_model.load_state_dict(torch.load(save_path))

        assert isinstance(loaded_model, Andromeda)

    def test_performance(self, andromeda_model):
        # Test model inference performance on a large batch
        input_data = torch.randn(256, 128)
        with torch.no_grad():
            output = andromeda_model(input_data)

        assert output.shape == (256, 128)

    @pytest.mark.parametrize("num_tokens", [1000, 2000, 5000])
    def test_custom_num_tokens(self, num_tokens):
        # Test initialization with different numbers of tokens
        custom_andromeda = Andromeda(num_tokens=num_tokens)
        assert isinstance(custom_andromeda, Andromeda)
