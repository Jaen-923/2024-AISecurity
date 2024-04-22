# -*- coding: utf-8 -*-
"""도전 과제.ipynb

1) 코드 실행방법: python filename.py <number of hidden layers> <output size of each hidden layer>

   ex) python xor.py 3 "[8, 4, 2]"

2) 실행 과정

   -  주어진 arguments에 따라 MLP 기반의 XOR 모델 생성

   - XOR 정답 데이터에 수렴하도록 학습,
    수렴하지 않을 시 모델 초기화 및 재학습 진행

     * 정답을 맞출 때까지 모델 학습을 무한 반복할 것

     * 적절한 hyperparameter를 적용하여 적은 회수로 학습이 되도록 할 것

   - 모델이 학습된 parameter를 활용하여 출력 방정식을 출력

   - 출력 방정식을 3차원 그래프로 그리기
"""

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import sys

# 입력 데이터
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)

# XOR 게이트 출력 데이터
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# MLP 모델
class LogicGate_Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LogicGate_Model, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Sigmoid())
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        # 가중치 초기화: Xavier 초기화
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.model(x)

# training 함수
def train(X, y, model, epochs=3000, lr=0.025, loss_fn=nn.BCELoss(), optimizer=optim.SGD, print_every=100):
    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        hypothesis = model(X)
        error = loss_fn(hypothesis, y)
        error.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {error.item()}')

        if error.item() < 0.001:
            break

    return model
    
def print_output_equation(model, hidden_sizes):
    with torch.no_grad():
        idx = 0
        for i, size in enumerate(hidden_sizes):
            weight = model.model[idx].weight.data.numpy()
            bias = model.model[idx].bias.data.numpy()
            print(f"은닉층 {i+1}:")
            print("Weight:", weight)
            print("Bias:", bias)
            print()
            idx += 2
        weight = model.model[idx].weight.data.numpy()
        bias = model.model[idx].bias.data.numpy()
        print()
        print(f"출력층:")
        print("Weight:", weight)
        print("Bias:", bias)
        print()

if __name__ == "__main__":
    while True:
        if len(sys.argv) != 3:
            print("Usage: python filename.py <number of hidden layers> <output size of each hidden layer>")
            sys.exit(1)

        num_hidden_layers = int(sys.argv[1])
        output_sizes = [int(size) for size in sys.argv[2].strip("[]").split(',')]

        # 모델 생성
        input_size = 2
        output_size = 1
        xor_model = LogicGate_Model(input_size, output_sizes, output_size)

        # 모델 학습
        xor_model = train(X, y_xor, xor_model, epochs=5000, optimizer=optim.Adam)

        result_xor = xor_model(X)

        if torch.all(torch.abs(torch.round(result_xor) - y_xor) < 1e-6):
            print()
            print("예측")
            print(result_xor, torch.where(result_xor > 0.5, torch.tensor(1), torch.tensor(0)))

            # 출력 방정식을 3차원 그래프로 그리기
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('XOR Gate')
            ax.set_xlabel('Input 1')
            ax.set_ylabel('Input 2')
            ax.set_zlabel('Output')

            with torch.no_grad():
                x1 = np.linspace(0, 1, 50)
                x2 = np.linspace(0, 1, 50)
                X1, X2 = np.meshgrid(x1, x2)
                Y = xor_model(torch.tensor(np.c_[X1.ravel(), X2.ravel()], dtype=torch.float)).reshape(X1.shape)
                ax.plot_surface(X1, X2, Y, cmap='viridis')

            plt.show()

            # 학습된 모델 출력 방정식
            print()
            print("출력 방정식")
            print_output_equation(xor_model, output_sizes)
            print()
            

            break
        else:
            print(result_xor, torch.where(result_xor > 0.5, torch.tensor(1), torch.tensor(0)))
            print("예측이 정확하지 않습니다. 모델을 재학습합니다.")

            # 은닉층의 개수와 노드 개수를 입력받아 모델을 초기화하고 다시 학습
            num_hidden_layers = int(input("은닉층의 개수를 입력하세요: "))
            output_sizes = [int(size) for size in input("각 은닉층의 노드 개수를 입력하세요 (콤마로 구분하여 입력): ").strip().split(',')]
