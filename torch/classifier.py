from torchvision.utils import make_grid

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 입력 차원 3, 필터 갯수 6, 필터 크기 5
        self.pool = nn.MaxPool2d(2, 2) # (2, 2) 크기의 최대 풀링 사용
        self.conv2 = nn.Conv2d(6, 16, 5) # 입력 차원 6, 필터 갯수 16, 필터 크기 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 입력 차원 16 * 5 * 5, 출력 차원 120
        self.fc2 = nn.Linear(120, 84) # 입력 차원 120, 출력 차원 84
        self.fc3 = nn.Linear(84, 10) # 입력 차원 84, 출력 차원 10
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 1차원으로 변환
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict(model_path):
    net = Net()
    net.load_state_dict(torch.load(model_path))
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # 이미지를 출력합니다.
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

if __name__ == "__main__":
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0

        for idx, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad() # 가중치 초기화 
            outputs = net(inputs) # 모델 출력
            loss = criterion(outputs, labels) # 비용 함수
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트

            running_loss += loss.item()
            if idx % 2000 == 1999:
                print(f"[{epoch + 1}, {idx + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    PATH = './net.pth'
    torch.save(net.state_dict(), PATH) # 모델 저장

    predict(PATH)


