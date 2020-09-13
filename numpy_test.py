import torch
import numpy as np


np_data = np.arange(6).reshape(2, -1)
torch_data = torch.FloatTensor(np_data)
a = torch.randn(1, 2, 3, 4)
print(
    '\nnumpy', np_data,
    '\ntorch', torch_data,
    a
)

