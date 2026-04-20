import torch
# print("PyTorch版本:", torch.__version__)
# print("CUDA是否可用:", torch.cuda.is_available())
# print("CUDA版本:", torch.version.cuda)
# print("GPU名称:", torch.cuda.get_device_name(0))



class MyNet():
    def __init__(self):
        self.w = torch.Tensor([[0.5, 0.4]]) #权重w1,w2的初始值
        self.w.requires_grad = True         #tensor需要计算梯度
        self.b = torch.Tensor([[2.0]])        #偏置b的初始值
        self.b.requires_grad = True         #tensor需要计算梯度

    def forward(self, x):
        # 输入 x 为标量，转为特征向量
        features = torch.tensor([[x**2], [x]], dtype=torch.float32)
        out = self.w.mm(features) + self.b
        return out

    def loss(self, x, y):
        y_pred = self.forward(x)
        return (y_pred - y) ** 2


x_data = [1.0, 2.0, 3.0]#假设y=2*x^2+3*x+1,4的输出为45
y_data = [6.0, 15.0, 28.0]

net = MyNet()
optimizer = torch.optim.SGD([net.w, net.b], lr=0.02)   # 使用优化器管理参数

print("before training x=4:", net.forward(4).item())

for epoch in range(500):
    epoch_loss = 0.0
    for x, y in zip(x_data, y_data):
        l = net.loss(x, y)                   #计算前向损失
        optimizer.zero_grad()
        l.backward()                         # 反向传播
        optimizer.step()                    # 更新参数
        epoch_loss += l.item()               # 累加损失
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}平均损失：{epoch_loss / len(x_data):.6f}")


print("训练后预测x=4:", net.forward(4).item())
print("学习到的权重w:", net.w.data.numpy(), "偏置b:", net.b.data.numpy())