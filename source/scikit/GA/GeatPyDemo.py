import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import time

"""目标函数"""


def aim(x):
    return x * np.sin(10 * np.pi * x) + 2.0


if __name__ == '__main__':
    x = np.linspace(-1, 2, 200)
    plt.plot(x, aim(x))

    """变量设置"""
    x1 = [-1, 2]  # 自变量范围
    b1 = [1, 1]  # 自变量边界
    varTypes = np.array([0])  # 自变量类型，0连续，1离散
    Encoding = 'BG'  # 表示采用二进制/格雷编码
    codes = [1]  # 变量编码方式，2个变量都用格雷
    precisions = [4]  # 变量的编码精度
    scales = [0]  # 采用算法刻度
    ranges = np.vstack([x1]).T  # 自变量范围矩阵
    borders = np.vstack([b1]).T  # 自变量边界矩阵

    """遗传算法参数设置"""
    NIND = 40  # 种群个体数目
    MAXGN = 25  # 最大遗传代数
    FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)  # 调用函数创建区域描述器
    Lind = int(np.sum(FieldD[0, :]))  # 计算编码后染色体长度
    obj_trace = np.zeros((MAXGN, 2))  # 定于目标函数值记录器
    var_trace = np.zeros((MAXGN, Lind))  # 定义染色体记录器， 记录每一代最优个体染色体

    """遗传算法进化"""
    start_time = time.time()
    Chrom = ea.crtbp(NIND, Lind)  # 生成种群染色体矩阵
    variable = ea.bs2real(Chrom, FieldD)  # 对初始种群编码
    ObjV = aim(variable)  # 计算初始种群个体目标函数值
    best_ind = np.argmax(ObjV)  # 计算当代最优个体序号

    # 开始进化
    for gen in range(MAXGN):
        FitnV = ea.ranking(-ObjV)  # 根据目标函数大小分配适应度
        Selch = Chrom[ea.selecting('rws', FitnV, NIND - 1), :]  # 选择，采用'rws'轮盘选择
        Selch = ea.recombin('xovsp', Selch, 0.7)  # 重组（两点交叉，交叉概率0.7）
        Selch = ea.mutbin(Encoding, Selch)  # 二进制种群变异
        # 父代子代合并
        Chrom = np.vstack([Chrom[best_ind, :], Selch])
        variable = ea.bs2real(Chrom, FieldD)  # 育种群体编码（2进制转10进制）
        ObjV = aim(variable)
        # 记录
        best_ind = np.argmax(ObjV)  # 计算当代最优个体序号
        obj_trace[gen, 0] = np.sum(ObjV) / NIND  # 记录当代种群目标函数均值
        obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
        var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体变量值

    # 进化结束
    end_time = time.time()

    # 输出结果和绘图
    best_gen = np.argmax(obj_trace[:, [1]])
    print("目标函数最大值:", obj_trace[best_gen, 1])  # 输出目标函数最大值
    variable = ea.bs2real(var_trace[[best_gen], :], FieldD)  # 编码获得表现型
    print('对应决策值变量')
    print(variable[0][0])  # variable是矩阵
    print('用时：', end_time - start_time)
    plt.plot(variable, aim(variable), 'bo')
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])
