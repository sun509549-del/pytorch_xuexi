import torch

x_raw = torch.Tensor([1.0, 2.0, 3.0])  # 假设y=2*x^2+3*x+1,4的输出为45
y_data = torch.Tensor([6.0, 15.0, 28.0]).reshape(-1, 1)  # 将y_data调整为列向量

# 构造特征矩阵，第一列是x，第二列是x^2
x_data = torch.stack([x_raw** 2, x_raw ], dim=1)  # 构造特征矩阵，第一列是x，第二列是x^2

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)    # 每一个数据样本输入和输出都是1维,即y=w*x+b,bias默认为True,表示训练时需要计算偏置
                                              #Linear(in_features, out_features, bias=True)

    def forward(self, x):     #这里的forward是对父类中的forward进行重写
        return self.linear(x)

model = LinearModel(2, 1)   #model是可调用对象，只要调用model对象，就会执行forward方法，构造计算图

criterion = torch.nn.MSELoss(reduction='sum') 
 # 均方误差损失函数,reduction='sum'表示对所有样本的损失求和，默认是'reduction='mean''，表示对所有样本的损失求平均
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  
#随机梯度下降优化器, model.parameters()会返回模型中所有需要训练的参数，parrameters()会检查model中的所有成员，如果成员里面有相应的权重，就会把这些权重加入到优化器中(加入到最后的结果中)，lr是学习率

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  
#Adam优化器, model.parameters()会返回模型中所有需要训练的参数，parrameters()会检查model中的所有成员，如果成员里面有相应的权重，就会把这些权重加入到优化器中(加入到最后的结果中)，lr是学习率

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch: {epoch}: Loss = {loss.item():.4f}')


#输出
print('w=', model.linear.weight.data.numpy())
print('b=', model.linear.bias.item())


#预测
x_test_raw = torch.tensor([4.0])
x_test = torch.stack([x_test_raw**2, x_test_raw], dim=1)  # 形状 (1, 2)
y_pred = model(x_test)
print('y_pred for x=4 =', y_pred.item())  # 应接近 45.0