import torch

#先确定维度
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

#输入数据
dataset = torch.randn(seq_len,batch_size,  input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input_t in enumerate(dataset):
    print('='*20, idx, '='*20)
    print('输入数据：', input_t.shape)

    hidden = cell(input_t, hidden)

    print('输出数据：', hidden.shape)
    print(hidden)
