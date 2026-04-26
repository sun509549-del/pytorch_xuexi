"""
Softmax 多分类：手写数字识别教学版

本脚本使用 MNIST 手写数字数据集，搭建一个最简单的多分类模型：
    784 个输入特征 -> 10 个类别输出

注意：
1. 模型最后一层直接输出 logits，不需要手动写 softmax。
2. nn.CrossEntropyLoss 会在内部自动完成 LogSoftmax + NLLLoss。
3. 训练结束后会绘制 loss 曲线和准确率曲线，方便观察学习效果。
"""

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# 1. 基础配置
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
DATA_DIR = "./data"

# 如果电脑有 NVIDIA GPU，则优先使用 GPU；否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")


# -----------------------------
# 2. 准备 MNIST 数据集
# -----------------------------
# transforms.ToTensor() 会把图片转成 Tensor，并把像素值从 0~255 缩放到 0~1。
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -----------------------------
# 3. 定义 Softmax 多分类模型
# -----------------------------
class SoftmaxClassifier(nn.Module):
    """最简单的线性多分类器。"""

    def __init__(self):
        super().__init__()
        # MNIST 图片大小是 1 x 28 x 28，展平后是 784 个特征。
        # 输出 10 个数字类别：0, 1, 2, ..., 9。
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # 原始输入形状：[batch_size, 1, 28, 28]
        # 展平后的形状：[batch_size, 784]
        x = x.view(x.size(0), -1)

        # 返回 logits。这里不要写 softmax，因为 CrossEntropyLoss 会处理。
        return self.linear(x)


model = SoftmaxClassifier().to(device)


# -----------------------------
# 4. 损失函数与优化器
# -----------------------------
# CrossEntropyLoss 适合多分类问题：
# 输入：模型输出的 logits，形状为 [batch_size, num_classes]
# 标签：真实类别编号，形状为 [batch_size]
criterion = nn.CrossEntropyLoss()

# 随机梯度下降优化器。
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# 5. 准确率计算函数
# -----------------------------
def calculate_accuracy(data_loader):
    """在指定数据集上计算准确率。"""
    model.eval()

    correct = 0
    total = 0

    # 评估阶段不需要计算梯度，可以节省内存和时间。
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # torch.max(outputs, dim=1) 会返回每一行最大值及其下标。
            # 下标就是模型预测的类别。
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# -----------------------------
# 6. 训练模型
# -----------------------------
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播：得到每个类别的 logits。
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播三步走：清零梯度 -> 反向传播 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计当前 batch 的 loss。
        running_loss += loss.item() * images.size(0)

        # 统计当前 batch 的训练准确率。
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    train_accuracy = correct / total
    test_accuracy = calculate_accuracy(test_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Test Acc: {test_accuracy:.4f}"
    )


# -----------------------------
# 7. 查看几个预测结果
# -----------------------------
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, dim=1)

print("\n前 10 个测试样本预测结果：")
print("真实标签：", labels[:10].cpu().tolist())
print("预测标签：", predicted[:10].cpu().tolist())


# -----------------------------
# 8. 绘制 loss 曲线与准确率曲线
# -----------------------------
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, marker="o", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()

# accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(epochs_range, test_accuracies, marker="s", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
