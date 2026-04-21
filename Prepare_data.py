import numpy as np
import torch
from torch.utils.data import Dataset  #Dataset是一个抽象类，只能继承不能实例化
from torch.utils.data import DataLoader  #可以实例化

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])  #前8列是特征
        self.y_data = torch.from_numpy(xy[:, [-1]])  #最后一列是标签
        self.len = self.y_data.shape[0]  #样本数量

    def __getitem__(self, index):  #索引访问
        return self.x_data[index], self.y_data[index]

    def __len__(self):    #获取数据集大小
        return self.len
    
if __name__ == '__main__':
    # 确保 diabetes.csv.gz 文件在当前目录，否则使用绝对路径
    dataset = DiabetesDataset('diabetes.csv.gz')  # 实例化数据集对象
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            inputs = x_batch
            labels = y_batch
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

