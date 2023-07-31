import torch
from Andromeda.configs import Andromeda1Billion

model =  Andromeda1Billion().cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

model(x) # (1, 1024, 20000)