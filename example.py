import torch
from Andromeda.model import Andromeda

model =  Andromeda()

x = torch.randint(0, 256, (1, 1024)).cuda()

model(x) # (1, 1024, 20000)