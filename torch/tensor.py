import torch
import numpy as np


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# 텐서 정의
tensor = torch.randn(3, 4, device=device)
print(tensor)

ones_tensor = torch.ones(3, 4, device=device)
print(ones_tensor)

zeros_tensor = torch.zeros(3, 4, device=device)
print(zeros_tensor)

print("--------------------------연산--------------------------------")

# 텐서 연산
tensor = torch.randn(4, 4, device=device)
print(tensor)

print(tensor[0])
print(tensor[-1])

print(tensor[:, 0])
print(tensor[:, -1])

print("--------------------------concat--------------------------------")

# concat
tensor_a = torch.randn(2, 3)
tensor_b = torch.randn(2, 1)
tensor_c = torch.randn(2, 2)

print(tensor_a)
print(tensor_b)
print(tensor_c)

# dim 1 -> 열기준 합치기 , dim 0 -> 행기준 합치기
tensor_concat = torch.cat([tensor_a, tensor_b, tensor_c], dim=1)
print(tensor_concat)

print("--------------------------산술 연산--------------------------------")

tensor_a = torch.randn(3, 3)
tensor_b = torch.randn(3, 3)

# 행렬곱
print(tensor_a @ tensor_b)

# element-wise 연산 (요소별곱)
print(tensor_a * tensor_b)

print(tensor_a)
# 전치행렬
print(tensor_a.T)