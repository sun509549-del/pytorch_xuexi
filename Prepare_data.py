import numpy as np
import torch

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 16)
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 1)
        # self.linear1 = torch.nn.Linear(8, 6)
        # self.linear2 = torch.nn.Linear(6, 4)
        # self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):

    #forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    #backward
    optimizer.zero_grad()
    loss.backward()

    #update
    optimizer.step()
