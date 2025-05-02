import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

class DataSet(torch.utils.data.Dataset):
    def __init__(self, numSamples, xRange):
        self.dataIn = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True).view(-1,1)
    
    def __len__(self):
        return len(self.dataIn)
    
    def __getitem__(self, idx):
        return self.dataIn[idx]

class Fitter(torch.nn.Module):
    def __init__(self, numHiddenNodes):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(1, numHiddenNodes)
        self.fc2 = torch.nn.Linear(numHiddenNodes, 1)
    
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        return y

def trialFunc(x, n_out):
    return x * (1 - x) * n_out  # Удовлетворяет u(0)=0 и u(1)=0

def dTrialFunc(x, n_out, dndx):
    return (1 - 2*x) * n_out + x*(1 - x)*dndx

def d2TrialFunc(x, n_out, dndx, d2ndx2):
    return -2*n_out + 2*(1 - 2*x)*dndx + x*(1 - x)*d2ndx2

def diffEq(x, f_trial, df_trial, d2f_trial):
    LHS = d2f_trial + f_trial
    RHS = -np.pi**2 * torch.sin(np.pi * x)
    return LHS - RHS

def solution(x):
    return torch.sin(np.pi * x)

def train(network, loader, lossFn, optimiser, numEpochs):
    cost_list = []
    network.train(True)
    for epoch in range(numEpochs):
        for batch in loader:
            n_out = network(batch)
            dndx = grad(n_out, batch, grad_outputs=torch.ones_like(n_out), create_graph=True)[0]
            d2ndx2 = grad(dndx, batch, grad_outputs=torch.ones_like(dndx), retain_graph=True)[0]
            
            f_trial = trialFunc(batch, n_out)
            df_trial = dTrialFunc(batch, n_out, dndx)
            d2f_trial = d2TrialFunc(batch, n_out, dndx, d2ndx2)
            
            diff_eq = diffEq(batch, f_trial, df_trial, d2f_trial)
            cost = lossFn(diff_eq, torch.zeros_like(diff_eq))
            cost.backward()
            optimiser.step()
            optimiser.zero_grad()
        
        cost_list.append(cost.item())
    return cost_list

# Параметры
numHiddenNodes = 20
xRange = [0, 1]
numSamples = 100
batchSize = 20
numEpochs = 5000

# Инициализация
network = Fitter(numHiddenNodes)
trainData = DataSet(numSamples, xRange)
trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, shuffle=True)
optimiser = torch.optim.Adam(network.parameters(), lr=0.001)
lossFn = torch.nn.MSELoss()

# Обучение
costList = train(network, trainLoader, lossFn, optimiser, numEpochs)

# Визуализация
x_test = torch.linspace(0, 1, 100).view(-1,1)
u_pred = trialFunc(x_test, network(x_test)).detach().numpy()
u_exact = solution(x_test).detach().numpy()

plt.plot(x_test, u_pred, 'r-', label='Neural Network')
plt.plot(x_test, u_exact, 'b--', label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title(f'MSE: {np.mean(u_pred - u_exact)**2:.2e}')
plt.show()

plt.semilogy(costList)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Convergence')
plt.show()