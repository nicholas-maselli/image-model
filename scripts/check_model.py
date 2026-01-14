import torch
from imagemodel.models.tiny_cnn import TinyCNN

m = TinyCNN()
x = torch.randn(8, 3, 32, 32)
y = m(x)
print(y.shape)  # should be torch.Size([8, 10])
