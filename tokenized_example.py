from andromeda_torch import Tokenizer
from andromeda_torch.configs import Andromeda1Billion

model = Andromeda1Billion()
tokenizer = Tokenizer()

encoded_text = tokenizer.encode("Hello world!")
out = model(encoded_text)
print(out)
