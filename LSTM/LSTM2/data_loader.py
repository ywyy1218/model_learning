# wjx
# 2023/11/17 10:07
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载训练集和测试集数据
train_data = np.load('./data/input.npy')
test_data = np.load('./data/predict.npy')

# 绘制训练集和测试集数据
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data')

# 添加标签和标题
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Training and Test Data')

# 显示图例
plt.legend()

# 显示图表
plt.show()


