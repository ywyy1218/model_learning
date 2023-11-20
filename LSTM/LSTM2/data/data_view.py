# wjx
# 2023/11/17 10:16
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def view_npy_info(file_path):
    try:
        # 加载npy文件
        data = np.load(file_path)

        # 打印数组的形状、数据类型和其他信息
        print("数组形状:", data.shape)
        print("数据类型:", data.dtype)
        print("数组内容:")
        print(data)

    except FileNotFoundError:
        print("文件不存在: {}".format(file_path))
    except Exception as e:
        print("发生错误:", e)

# 指定npy文件路径
npy_file_path1 = "input.npy"
npy_file_path2 = "predict.npy"

# 查看npy文件信息
view_npy_info(npy_file_path1)
'''
数组形状: (2017,)
数据类型: float32
数组内容:
[-0.00512695 -0.01123047 -0.00805664 ... -0.04345703 -0.01269531
 -0.01318359]
 '''
view_npy_info(npy_file_path2)
'''
数组形状: (500,)
数据类型: float32
数组内容:
[ 0.3857422   0.48950195  0.7277832   0.40234375  0.8376465   0.77124023
  0.7783203   0.6508789   0.9177246   0.5541992   0.6376953   0.92993164
  ...]
'''
