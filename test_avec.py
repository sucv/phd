from utils.base_class import AVEC19ArrangerNPY
from utils.dataset import AVEC19Dataset
from model.prototype import my_2d1d
import json
import torch

device = torch.device("cuda:2")
data = torch.randint(0, 254, (2, 300, 3, 112, 112), dtype=torch.uint8)
data = torch.tensor(data, dtype=torch.float32).to(device)
model = my_2d1d(feature_dim=512, channels_1D=[128, 128, 128], output_dim=2, kernel_size=3, dropout=0.1).to(device)
output = model(data)
print(output.shape)
