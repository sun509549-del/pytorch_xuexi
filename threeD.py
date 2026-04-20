import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 保持导入，但改用 add_subplot

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 穷举参数空间
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

mse_matrix = np.zeros((len(w_range), len(b_range)))

for i, w in enumerate(w_range):
    for j, b in enumerate(b_range):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w, b)
        mse_matrix[i, j] = l_sum / len(x_data)

# 生成网格坐标（注意：meshgrid 默认行列与矩阵索引相反，需要转置）
W, B = np.meshgrid(w_range, b_range)
Z = mse_matrix.T   # 转置以匹配 W, B 的形状

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Z, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()