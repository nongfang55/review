# coding=gbk
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from minepy import MINE
from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class RandomizedLasso(object):
    pass


class FeatureSelectUtils:
    """用于研究特征相关性"""

    @staticmethod
    def demo1():
        """单变量特征选择"""

        """皮尔森相关系数"""
        np.random.seed(0)
        size = 300
        x = np.random.normal(0, 1, size)
        print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
        print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))

        """互信息和最大信息系数(MIC)"""
        m = MINE()
        x = np.random.uniform(-1, 1, 10000)
        m.compute_score(x, x ** 2)
        print(m.mic())

    @staticmethod
    def demo2():
        """单变量特征选择"""

        """基于学习模型特征排序"""
        """随机森林回归"""
        boston = load_boston()
        X = boston['data']
        Y = boston['target']
        names = boston['feature_names']

        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        scores = []
        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i + 1], Y, scoring="r2",
                                    cv=ShuffleSplit(len(X), 3, .3))
            scores.append((round(np.mean(score), 3), names[i]))
        print(sorted(scores, reverse=True))

    @staticmethod
    def demo3():
        """正则化模型"""

        """L1正则化 lasso"""
        boston = load_boston()
        scaler = StandardScaler()
        X = boston['data']
        Y = boston['target']
        names = boston['feature_names']

        lasso = Lasso(alpha=.3)
        lasso.fit(X, Y)
        print("Lasso model:", lasso.coef_)

        """L2 正则化"""

    @staticmethod
    def demo4():
        """随机森林"""
        """平均不纯度减少"""
        # Load boston housing dataset as an example
        boston = load_boston()
        X = boston["data"]
        Y = boston["target"]
        names = boston["feature_names"]
        rf = RandomForestRegressor()
        rf.fit(X, Y)
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

    @staticmethod
    def demo5():
        boston = load_boston()
        X = boston["data"]
        Y = boston["target"]
        names = boston["feature_names"]
        rf = RandomForestRegressor()
        scores = defaultdict(list)

        # crossvalidate the scores on a number of different random splits of the data
        for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = rf.fit(X_train, Y_train)
            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores[names[i]].append((acc - shuff_acc) / acc)
        print("Features sorted by their score:")
        print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

    @staticmethod
    def demo6():
        """顶层特征选择法"""
        """稳定性选择"""
        boston = load_boston()
        X = boston["data"]
        Y = boston["target"]
        names = boston["feature_names"]

        rlasso = RandomizedLasso(alpha=0.025)
        rlasso.fit(X, Y)
        print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True))

    @staticmethod
    def demo7():
        """递归特征消除 RFE"""

        boston = load_boston()
        X = boston["data"]
        Y = boston["target"]
        names = boston["feature_names"]

        lr = LinearRegression()
        rfe = RFE(lr, n_features_to_select=1)
        rfe.fit(X, Y)
        print("Features sorted by their rank:")
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

    @staticmethod
    def demo8():
        np.random.seed(0)
        size = 750
        X = np.random.uniform(0, 1, (size, 14))
        Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3]
             + 5 * X[:, 4] + np.random.normal(0, 1))
        X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))

        names = ["x%s" % i for i in range(1, 15)]

        ranks = {}

        def rank_to_dict(ranks, names, order=1):
            minmax = MinMaxScaler()
            ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
            ranks = map(lambda x: round(x, 2), ranks)
            return dict(zip(names, ranks))

        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

        ridge = Ridge(alpha=7)
        ridge.fit(X, Y)
        ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

        lasso = Lasso(alpha=.05)
        lasso.fit(X, Y)
        ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
        #
        # rlasso = RandomizedLasso(alpha=0.04)
        # rlasso.fit(X, Y)
        # ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

        # stop the search when 5 features are left (they will get equal scores)
        rfe = RFE(lr, n_features_to_select=5)
        rfe.fit(X, Y)
        ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)

        rf = RandomForestRegressor()
        rf.fit(X, Y)
        ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

        f, pval = f_regression(X, Y, center=True)
        ranks["Corr."] = rank_to_dict(f, names)

        mine = MINE()
        mic_scores = []
        for i in range(X.shape[1]):
            mine.compute_score(X[:, i], Y)
            m = mine.mic()
            mic_scores.append(m)

        ranks["MIC"] = rank_to_dict(mic_scores, names)

        r = {}
        for name in names:
            r[name] = round(np.mean([ranks[method][name]
                                     for method in ranks.keys()]), 2)

        methods = sorted(ranks.keys())
        ranks["Mean"] = r
        methods.append("Mean")

        print("\t%s" % "\t".join(methods))
        for name in names:
            print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))

    @staticmethod
    def demo9():
        """卡方检测测试"""
        iris = load_iris()
        print(iris.data.shape)
        print(iris.target)
        model1 = SelectKBest(chi2, k=2)
        print(model1.fit_transform(iris.data, iris.target))
        print(model1.scores_)
        print(model1.pvalues_)


if __name__ == '__main__':
    FeatureSelectUtils.demo9()
