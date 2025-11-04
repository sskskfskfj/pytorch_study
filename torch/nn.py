from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import torch 


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # sequential로 레이어 쌓기
            nn.Linear(28 * 28, 512), # 입력 (28, 28) -> 출력 (512)
            nn.ReLU(), # -> 활성화 함수(비선형성 추가) 모델이 데이터를 더 잘 학습할 수 있도록 만들어줌
            nn.Linear(512, 512), # 입력 (512) -> 출력 (512)
            nn.ReLU(),
            nn.Linear(512, 10), # 입력 (512) -> 출력 (10)
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x) # n차원 벡터를 1차원 벡터로 변환
        logits = self.linear_relu_stack(x) # 모델의 출력 계산 
        return logits


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)

    x = torch.randn(1, 28, 28, device=device) # 입력 데이터 
    logits = model(x) # 모델 출력
    pred_prob = nn.Softmax(dim=1)(logits) # 츌력값을 확률로 변환
    pred_prob = pred_prob.argmax(dim=1) # 확률이 가장 높은 인덱스 반환
    print(f"Predicted class: {pred_prob}")