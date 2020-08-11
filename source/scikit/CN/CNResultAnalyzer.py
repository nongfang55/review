"""
1. 读取output文件，获取指定标签的数据
2. 对CN网络的数据指标与准确度做相关性分析
"""
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from source.utils.ExcelHelper import ExcelHelper
import scipy
from scipy.stats import pearsonr
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

MODULARITY = 'Modularity'
ENTROPY = 'entropy'
TOPK_ACC = 'TopKAccuracy'
SIZE = '社区大小'

def readOutPutFile(project, keys, cid='whole', filter_train=True, filter_test=True):
    """
    key表示具体要获取的数据
    """
    # 拼接output文件
    returns = []
    filename = f"outputCN_{project}_{filter_train}_{filter_test}.xls"
    for key in keys:
        if key == MODULARITY:
            modularitys = []
            row_idx = 2
            row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            while row is not None and modularitys.__len__() < 12:
                if key in row:
                    next_row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx + 1)
                    if next_row[1] == cid:
                       modularitys.append(next_row[4])
                row_idx += 18
                row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            returns.append(modularitys)
        if key == ENTROPY:
            entropys = []
            row_idx = 2
            row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            while row is not None and entropys.__len__() < 12:
                if key in row:
                    next_row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx + 1)
                    if next_row[1] == cid:
                        entropys.append(next_row[5])
                row_idx += 18
                row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            returns.append(entropys)
        if key == SIZE:
            sizes = []
            row_idx = 2
            row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            while row is not None and sizes.__len__() < 12:
                if key in row:
                    next_row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx + 1)
                    if next_row[1] == cid:
                        sizes.append(next_row[2])
                row_idx += 18
                row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            returns.append(sizes)
        if key == TOPK_ACC:
            topks = [[], [], []]
            row_idx = 4
            row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            while row is not None and topks[0].__len__() < 12:
                if key in row:
                    next_row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx + 2)
                    topks[0].append(next_row[2])
                    topks[1].append(next_row[4])
                    topks[2].append(next_row[6])
                row_idx += 18
                row = ExcelHelper().readExcelRow(filename, "result", startRow=row_idx)
            returns.append(topks)
    return returns

def caculateCoin3D(xList, y, xlabel=None, ylabel=None,zlabel=None, t=111):
    """线性相关分析"""
    formatx = []
    for i in range(0, y.__len__()):
        formatx.append([xList[0][i], xList[1][i]*xList[1][i]])
    X2 = sm.add_constant(formatx)
    model = sm.OLS(y, X2).fit()
    print(model.summary())
    print(model.params)
    fig = plt.figure()
    ax = fig.add_subplot(t, projection='3d')
    ax.scatter(xList[0], xList[1], y, c='b', marker='o')
    ax.scatter(xList[0], xList[1], model.fittedvalues, c='r', marker='x')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # 设置三维图形模式
    ax = fig.gca(projection='3d')
    X = np.arange(min(xList[0]), max(xList[0]), 0.01)
    Y = np.arange(min(xList[1]), max(xList[1]), 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = model.params[0] + model.params[1] * X + model.params[2]*Y*Y
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, max(y))

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

def caculateCoin2D(x, y, xlabel=None, ylabel=None, t=111):
    """线性相关分析"""
    correlation, p_value = scipy.stats.pearsonr(x, y)
    print("pearsonr: correlation:{0}, p_value{1}".format(correlation, p_value))
    X2 = sm.add_constant(x)
    model = sm.OLS(y, X2).fit()
    print(model.summary())
    print(model.params)
    TX = np.arange(min(x), max(x), 0.01)
    TY = model.params[0] + model.params[1] * TX
    ax1 = plt.subplot(t)
    ax1.scatter(x, y, c='black', label='real')
    plt.plot(TX, TY, c='blue', label='fitted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def analyzeModularityWithAcc(projects):
    """分析模块度和准确率之间的关系"""
    x = []
    y = [[], [], []]
    for p in projects:
        # 首先读取模块度列表
        modularitys, topks = readOutPutFile(p, cid="whole", keys=[MODULARITY, TOPK_ACC])
        # 求模块度均值
        x.append(np.mean(modularitys))
        # 求top1, 3, 5均值
        y[0].append(np.mean(topks[0]))
        y[1].append(np.mean(topks[1]))
        y[2].append(np.mean(topks[2]))
    """绘制图形"""
    ks = [' 1', ' 3', ' 5']
    for i in range(0, 3):
        caculateCoin2D(x, y[i], t=130 + (i + 1), xlabel=MODULARITY, ylabel=TOPK_ACC + ks[i])
    plt.show()

def analyzeEntropyWithAcc(projects):
    """分析度分布熵和准确率之间的关系"""
    x = []
    y = [[], [], []]
    for p in projects:
        # 首先读取模块度列表
        entropys, topks = readOutPutFile(p, cid="whole", keys=[ENTROPY, TOPK_ACC])
        # 求模块度均值
        x.append(np.mean(entropys))
        # 求top1, 3, 5均值
        y[0].append(np.mean(topks[0]))
        y[1].append(np.mean(topks[1]))
        y[2].append(np.mean(topks[2]))
    """绘制图形"""
    ks = [' 1', ' 3', ' 5']
    for i in range(0, 3):
        caculateCoin2D(x, y[i], t=130 + (i + 1), xlabel=ENTROPY, ylabel=ks[i])
    plt.show()

def analyzeEntropyWithModularity(projects):
    """分析度分布熵和模块度之间的关系"""
    x = []
    y = []
    for p in projects:
        # 首先读取模块度列表
        entropys, modularitys = readOutPutFile(p, cid="whole", keys=[ENTROPY, MODULARITY])
        # 求熵均值
        x.append(np.mean(entropys))
        # 求模块度均值
        y.append(np.mean(modularitys))
    """绘制图形"""
    caculateCoin2D(y, x, t=111, ylabel=MODULARITY, xlabel=ENTROPY)
    plt.show()

def analyzeModularityAndEntropyWithAcc(projects):
    """分析度分布熵和准确率之间的关系"""
    x1 = []
    x2 = []
    y = [[], [], []]
    for p in projects:
        # 首先读取模块度列表
        modularitys, entropys, topks = readOutPutFile(p, cid="whole", keys=[MODULARITY, ENTROPY, TOPK_ACC])
        # 求模块度均值
        x1.append(np.mean(modularitys))
        x2.append(np.mean(entropys))
        # 求top1, 3, 5均值
        y[0].append(np.mean(topks[0]))
        y[1].append(np.mean(topks[1]))
        y[2].append(np.mean(topks[2]))
    """绘制图形"""
    ks = [' 1', ' 3', ' 5']
    for i in range(0, 3):
        caculateCoin3D([x1, x2], y[i], ylabel=ENTROPY, xlabel=MODULARITY, zlabel=TOPK_ACC + ks[i])
        plt.show()

if __name__ == '__main__':
    # projects = ['opencv', 'angular', 'cakephp', 'akka', 'django', 'react', 'symfony', 'babel']
    projects = ['opencv', 'cakephp', 'akka', 'django', 'react', 'symfony',
                'babel', 'angular', 'scikit-learn', 'pandas', 'react',
                'brew', 'metasploit-framework','Baystation12',
                'fastlane', 'salt', 'netty', 'moby', 'xbmc']
    analyzeModularityWithAcc(projects)
    analyzeEntropyWithAcc(projects)
    analyzeModularityAndEntropyWithAcc(projects)
    # analyzeEntropyWithModularity(projects)