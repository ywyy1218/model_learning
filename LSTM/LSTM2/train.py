import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import argparse
from data_loader import *
from model import AirModel


def train_model(model, optimizer, loss_fn, loader):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def validate_model(model, loss_fn, X_train, X_test, best_rmse, best_model_path):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))

    if test_rmse < best_rmse:
        print(f"验证：发现新的最佳测试 RMSE：{test_rmse:.4f}")
        save_model(model, best_model_path)
        best_rmse = test_rmse
    return train_rmse, test_rmse, best_rmse

# 保存
def save_model(model, path):
    torch.save(model, path)
# 加载： torch.load(model.pth)
'''
保存：
torch.save(model.state_dict(), path)
加载：
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''

def pearson(vector1, vector2):
  n = len(vector1)

  sum1 = sum(float(vector1[i]) for i in range(n))
  sum2 = sum(float(vector2[i]) for i in range(n))

  sum1_pow = sum([pow(v, 2.0) for v in vector1])
  sum2_pow = sum([pow(v, 2.0) for v in vector2])

  p_sum = sum([vector1[i]*vector2[i] for i in range(n)])

  num = p_sum - (sum1*sum2/n)
  den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
  if den == 0:
    return 0.0
  return abs(num/den)

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='./data/input.npy', help='train path')
parser.add_argument('--test_file', type=str, default='./data/predict.npy', help='test path')
parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
parser.add_argument('--lookback', type=int, default=50, help='window')
parser.add_argument('--best_model_path', type=str, default='./model.pth', help='best_model_path')
args = parser.parse_args()

train = load_data(args.train_file)
test = load_data(args.test_file)

# print(train.shape, type(train))
# print(test.shape, type(test))


X_train, y_train = create_dataset(train, lookback=args.lookback)
X_test, y_test = create_dataset(test, lookback=args.lookback)


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

best_rmse = float('inf')
for epoch in range(args.epochs):
    model = train_model(model=model, optimizer=optimizer, loss_fn=loss_fn, loader=loader)
    # 验证
#     if epoch % 100 != 0:
#         continue
    train_rmse, test_rmse, best_rmse = validate_model(model=model, loss_fn=loss_fn, X_train=X_train, X_test=X_test,
                                           best_rmse=best_rmse, best_model_path=args.best_model_path)
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# 关闭 PyTorch 的梯度追踪
with torch.no_grad():
    test_plot = np.ones_like(test) * np.nan
    test_plot[args.lookback:] = model(X_test)[:, -1, :]


# 设置图的大小
plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 18})
# plot
plt.plot(test, c='b', label='True')
plt.plot(test_plot, c='g', label='predict')
# 添加网格线
plt.grid(True)
plt.xlabel('time:day')
plt.ylabel('Value')
# 添加图例
plt.legend(loc='upper left')
# 保存图像
plt.savefig('img.png')
# 显示图
plt.show()

