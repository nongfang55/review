import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'ZDT1'  # 初始化name
        M = 2  # 初始化M(目标维数)
        maxormins = [1] * M  # 初始化maxormins (目标最大化最小化标记列表 1:最小化该目标  -1：最大化给目标)
        Dim = 2  # 初始化Dim 决策变量维数
        varTypes = [0] * Dim  # 初始化varTypes (决策变量类型, 0：实数, 1:整数)
        lb = [0] * Dim  # 决策变量下界
        ub = [5, 3]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 获得决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        f1 = 4 * x1 ** 2 + 4 * x2 ** 2
        f2 = (x1 - 5)**2 + (x2 - 5)**2
        # 可行性法则处理约束
        pop.CV = np.hstack([(x1 - 5)**2 + x2**2 - 25, -(x1 - 8)**2 - (x2 - 3)**2 + 7.7])
        #求得目标函数值赋给种群pop的Objv
        pop.ObjV = np.hstack([f1, f2])

    # def calReferObjV(self):  # 计算全局最优解作为目标函数参考值
    #     N = 10000  # 预得到10000个真实前沿点
    #     x1 = np.linspace(0, 5, N)
    #     x2 = x1.copy()
    #     x2[x1 >= 3] = 3
    #     return np.vstack((4 * x1**2 + 4 * x2 ** 2, (x1 - 5)**2 + (x2 - 5)**2)).T


if __name__ == '__main__':
    problem = MyProblem()
    """种群设置"""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象
    """算法参数设置"""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化模板对象
    myAlgorithm.MAXGEN = 200  # 最大迭代数量
    myAlgorithm.drawing = 1  # 设置绘图方式（0:不绘图 1：结果图  2：过程动画）
    """调用模板种群进化
    调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    NDSet = myAlgorithm.run()  # 执行算法模板，获取非支配种群
    NDSet.save()  # 结果保存文件

    # 输出
    print("用时：%f秒" % (myAlgorithm.passTime))
    print("评价次数：%d" % (myAlgorithm.evalsNum))
    print("非支配个体数:%d" % (NDSet.sizes))
    print('单位时间找到帕累托前沿点个数:%d' % (int(NDSet.sizes //
                                     myAlgorithm.passTime)))

    # 计算指标
    PF = problem.getReferObjV()  # 获取真实前沿
    if PF is not None and NDSet.sizes != 0:
        GD = ea.indicator.GD(NDSet.ObjV, PF)  # 计算GD指标
        IGD = ea.indicator.IGD(NDSet.ObjV, PF)  # 计算IGD指标
        HV = ea.indicator.HV(NDSet.ObjV, PF)  # 计算Spacing 指标
        Spacing = ea.indicator.Spacing(NDSet.ObjV)
        print('GD', GD)
        print('IGD', IGD)
        print('HV', HV)
        print('Spacing', Spacing)

    """进化过程指标追踪分析"""
    if PF is not None:
        metricName = [['IGD'], ['HV']]
        [NDSet_trace, Metrics] = ea.indicator.moea_tracking(myAlgorithm.pop_trace,
                                                            PF, metricName, problem.maxormins)
        ea.trcplot(Metrics, labels=metricName, titles=metricName)
