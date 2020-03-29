# coding=gbk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.naive_bayes import GaussianNB


class MLGraphHelper:
    """提供一些绘画接口的帮助类"""

    @staticmethod
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                            train_sizes=np.linspace(0.1, 1.0, 20)):
        """绘制学习曲线"""
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)  # 设置y的范围
        plt.xlabel("Training example")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)  # 提供模型数据
        """
            cv : int, 交叉验证生成器或可迭代的可选项，确定交叉验证拆分策略。
            1 无，使用默认的3倍交叉验证，
            2 整数，指定折叠数。
            3 要用作交叉验证生成器的对象。
            4 可迭代的yielding训练/测试分裂。
        """
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)  # 求总体标准差
        test_scores_mean = np.mean(test_scores, axis=1)  # 对各行求平均值
        test_scores_std = np.std(train_scores, axis=1)
        plt.grid()  # 设置网格线

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
        plt.plot(train_sizes, test_scores_mean, '-o', color='g', label="Cross-validation score")
        plt.legend(loc="best")  # 设置图例
        return plt


if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target  # 加载样例数据
    print(X)
    print(y)
    title = r"Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)  # 随机打散分成两部分
    estimator = GaussianNB()
    MLGraphHelper.plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1).show()
