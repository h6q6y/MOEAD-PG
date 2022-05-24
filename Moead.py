import torch

import problem.LIRCMOP1 as LIR1
import problem.LIRCMOP2 as LIR2
import numpy as np

class MOEAD:
    Pop = []            # 种群
    Pop_FV = []         # 种群计算出的函数值
    Pop_size = 200      # 种群大小，取决于vector_csv_file/下的xx.csv
    max_gen = 150       # 最大迭代次数,也就是进化代数
    Test_fun = LIR2     # 测试函数
    name2 = 'LIRCMOP2'  # 读理想曲线的文件
    T = int(0.1 * Pop_size)         #邻居数
    delta = 0.9         #the probability of selecting individuals from its neighboor
    n_r = 2
    need_dynamic = True  # 是否动态展示
    M = 2
    def __init__(self):
        self.Init_data()

    def Init_data(self):
        Creat_Pop(self)       # 创建种群

def Creat_Pop(moead):
    Pop = []
    while len(Pop) != moead.Pop_size:
        X = Creat_child(moead)
        Pop.append(X.tolist())
    moead.Pop = Pop

def Creat_child(moead):
    # 创建一个个体
    # （就是一个向量，长度为Dimention，范围在moead.Test_fun.Bound中设定）
    child = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand(
        moead.Test_fun.Dimention)
    return child

def overall_cv(cv):
    cv = np.array(cv)
    cv[cv > 0] = 0
    cv = abs(cv)
    result = sum(cv)
    return result