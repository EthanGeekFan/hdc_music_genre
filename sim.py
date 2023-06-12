import torch
import numpy as np

HDC_DIM = 1024

device = torch.device('cuda')
base = torch.rand(HDC_DIM, device=device) * 2 * np.pi - np.pi
base *= 4
symbol = torch.rand(HDC_DIM, device=device) * 2 * np.pi - np.pi
another_symbol = torch.rand(HDC_DIM, device=device) * 2 * np.pi - np.pi
val1 = 0.1
val2 = 1

hdc1 = base * val1 + symbol
hdc2 = base * val2 + symbol

another_hdc1 = base * val1 + another_symbol

similarity = torch.mean(torch.cos(hdc1 - hdc2))
print(similarity)

r1 = torch.rand(HDC_DIM, device=device) * 2 * np.pi - np.pi
r2 = torch.rand(HDC_DIM, device=device) * 2 * np.pi - np.pi
random_similarity = torch.mean(torch.cos(r1 - r2))
print(random_similarity)

another_similarity = torch.mean(torch.cos(hdc1 - another_hdc1))
print(another_similarity)