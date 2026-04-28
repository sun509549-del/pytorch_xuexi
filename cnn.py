"""
CNN 手写数字识别教学版代码

功能：
1. 使用 torchvision 自动下载 MNIST 手写数字数据集。
2. 搭建一个简单的卷积神经网络 CNN。
3. 完整演示训练、测试、准确率统计。
4. 训练结束后绘制并保存 loss 曲线和准确率曲线。

运行方式：
    python cnn.py

运行后会在当前目录生成：
    cnn_training_curves.png
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# 1. 基础配置
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DATA_DIR = "./data"
FIGURE_PATH = Path("cnn_training_curves.png")
RANDOM_SEED = 42


def set_seed(seed):
    """固定随机种子，让每次运行结果尽量接近，便于学习和调试。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

# 如果电脑有 NVIDIA GPU，则优先使用 GPU；否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")


# -----------------------------
# 2. 准备 MNIST 数据集
# -----------------------------
# 当前目录已经有 MNIST 数据集，因此这里设置 download=False。
# torchvision 会从 ./data/MNIST/raw 读取本地数据。
# transforms.ToTensor():
#   把 PIL 图片转换成 Tensor，并把像素值从 0~255 缩放到 0~1。
#
# transforms.Normalize((0.1307,), (0.3081,)):
#   使用 MNIST 数据集常用的均值和标准差做标准化，让模型更容易训练。
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=False,
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=False,
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
# 3. 定义 CNN 模型
# -----------------------------
class CNN(nn.Module):
    """一个适合入门理解的 CNN 手写数字识别模型。"""

    def __init__(self):
        super().__init__()

        # 卷积部分负责从图片中提取局部特征，例如笔画、边缘、形状。
        self.features = nn.Sequential(
            # 输入形状：[batch_size, 1, 28, 28]
            # 1 表示灰度图只有 1 个通道。
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool2d 会把宽高减半：28 x 28 -> 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 再减半：14 x 14 -> 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类部分负责根据 CNN 提取到的特征判断数字类别。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 经过两次池化后，特征图形状是 [batch_size, 32, 7, 7]。
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # 输出 10 个数字类别的 logits：0, 1, 2, ..., 9。
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
print(model)


# -----------------------------
# 4. 定义损失函数和优化器
# -----------------------------
# CrossEntropyLoss 适合多分类任务。
# 注意：模型最后一层不需要写 Softmax，因为 CrossEntropyLoss 内部已经包含相关计算。
criterion = nn.CrossEntropyLoss()

# Adam 通常比普通 SGD 更容易在入门实验中得到较快收敛。
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# 5. 训练一个 epoch
# -----------------------------
def train_one_epoch(epoch):
    """训练一轮，并返回本轮平均 loss 和准确率。"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播：得到模型对每张图片的预测结果。
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播三步：清空旧梯度 -> 计算新梯度 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss.item() 是当前 batch 的平均 loss。
        # 乘以 batch 大小后累加，最后再除以样本总数，得到整轮平均 loss。
        running_loss += loss.item() * images.size(0)

        # outputs 中每一行有 10 个值，最大值所在下标就是预测类别。
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_index % 200 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = running_loss / len(train_dataset)
    accuracy = correct / total
    return average_loss, accuracy


# -----------------------------
# 6. 在测试集上评估模型
# -----------------------------
def evaluate():
    """在测试集上计算平均 loss 和准确率。"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 测试阶段不需要计算梯度，可以节省内存并加快速度。
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(test_dataset)
    accuracy = correct / total
    return average_loss, accuracy


# -----------------------------
# 7. 绘制 loss 和准确率曲线
# -----------------------------
def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """把训练过程中的 loss 和准确率画成曲线，并保存到图片文件。"""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # loss 曲线：越低通常说明模型拟合得越好。
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs_range, test_losses, marker="s", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss Curve")
    plt.grid(True)
    plt.legend()

    # 准确率曲线：越高通常说明分类效果越好。
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(epochs_range, test_accuracies, marker="s", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy Curve")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"\n曲线图已保存到：{FIGURE_PATH.resolve()}")
    plt.show()


# -----------------------------
# 8. 查看一小批预测结果
# -----------------------------
def show_prediction_examples():
    """打印前 10 个测试样本的真实标签和预测标签。"""
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
# 9. 主程序入口
# -----------------------------
if __name__ == "__main__":
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(epoch)
        test_loss, test_accuracy = evaluate()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_accuracy:.4f}"
        )

    show_prediction_examples()
    plot_curves(train_losses, test_losses, train_accuracies, test_accuracies)
