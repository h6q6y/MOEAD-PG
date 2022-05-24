import numpy as np
import math
# 函数的维度（目标维度不一致的自行编写目标函数）
Dimention = 30
# 目标函数个数
Func_num = 2
Bound = [0, 1]
a = 0.51
b = 0.5

def Func(X):
    g1_value = g1(X)
    g2_value = g2(X)
    f1 = X[:, 0] + g1_value
    f2 = 1 - np.sqrt(X[:, 0]) + g2_value
    c1 = (a - g1_value) * (g1_value - b)
    c2 = (a - g2_value) * (g2_value - b)
    return np.array([[f1, f2], [c1, c2]])


def g1(X):
    x1 = X[:, 0:1]
    x1 = np.sin(0.5 * math.pi * x1)
    X = X[:, 2:Dimention:2]
    return np.sum((X - x1) ** 2, axis=1)

def g2(X):
    x1 = X[:, 0:1]
    x1 = np.cos(0.5 * math.pi * x1)
    X = X[:, 1:Dimention:2]
    return np.sum((X - x1) ** 2, axis=1)


