'''
实现隐语义模型，对隐式数据进行推荐
1.对正样本生成负样本
  -负样本数量相当于正样本
  -物品越热门，越有可能成为负样本
2.使用随机梯度下降法，更新参数
'''
import math
import random

import numpy as np
import pandas as pd
from math import exp

class LFM:
    def __init__(self, rating_data, F=5, alpha=0.1, lmbd=0.1, max_iter=500):
        """
        :param rating_data: rating_data是[(user,[(item,rate)]]类型
        :param F: 隐因子个数
        :param alpha: 学习率
        :param lmbd: 正则化
        :param max_iter:最大迭代次数
        """
        self.F = F
        self.P = dict()  # R=PQ^T，代码中的Q相当于博客中Q的转置
        self.Q = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data = rating_data

        '''随机初始化矩阵P和Q'''
        for user, rates in self.rating_data.items():
            self.P[user] = [random.random() / math.sqrt(self.F)
                            for x in range(self.F)]
            for item, _ in rates.items():
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F)
                                    for x in range(self.F)]

    def train(self):
        """
        随机梯度下降法训练参数P和Q
        """
        for step in range(self.max_iter):
            for user, rates in self.rating_data.items():
                for item, rui in rates.items():
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    for f in range(self.F):
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小

    def predict(self, user, item):
        """
        :param user:
        :param item:
        :return:
        预测用户user对物品item的评分
        """
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F))


if __name__ == '__main__':
    '''用户有A B C，物品有a b c d'''
    rating_data = list()
    rate_A = [('a', 1.0), ('b', 1.0)]
    rating_data.append(('A', rate_A))
    rate_B = [('b', 1.0), ('c', 1.0)]
    rating_data.append(('B', rate_B))
    rate_C = [('c', 1.0), ('d', 1.0)]
    rating_data.append(('C', rate_C))

    lfm = LFM(rating_data, 2)
    lfm.train()
    for item in ['a', 'b', 'c', 'd']:
        print(item, lfm.predict('A', item)) # 计算用户A对各个物品的喜好程度