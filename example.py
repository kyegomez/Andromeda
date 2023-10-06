import torch
# from Andromeda.model import Andromeda
from Andromeda.configs import Andromeda1Billion

model = Andromeda1Billion()

x = torch.randint(0, 256, (1, 1024)).cuda()

out = model(x) # (1, 1024, 20000)
print(out)