import numpy as np
import geatpy as ea
from pandas import DataFrame


class GAProblem(ea.Problem):
    def __init__(self, candicateNum, EXPScoreVector, RSPScoreVector, ACTScoreVector, recommendNum):
        name = 'GA1'  # 初始化name
        M = 1  # 初始化M(目标维数)
        maxormins = [-1] * M  # 初始化maxormins (目标最大化最小化标记列表 1:最小化该目标  -1：最大化给目标)
        Dim = candicateNum  # 初始化Dim 决策变量维数
        varTypes = [1] * Dim  # 初始化varTypes (决策变量类型, 0：实数, 1:整数)
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界

        self.recommend_num = recommendNum  # 推荐人数
        self.EXPScoreVector = EXPScoreVector  # 经验分数
        self.RSPScoreVector = RSPScoreVector  # 回应分数
        self.ACTScoreVector = ACTScoreVector  # 活跃分数
        self.ScoreVector = np.hstack((self.EXPScoreVector, self.RSPScoreVector, self.ACTScoreVector))
        self.CountVector = np.ones((candicateNum, 1))
        self.ratioVector = np.array([[0.4, 0.2, 0.4]]).T    # 计算三个分数的比例

        # 调用父类方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Chrom  # 获得种群染色体矩阵
        Vars = DataFrame(Vars)
        pop.ObjV = np.dot(Vars, self.ScoreVector)  # 矩阵乘法节约时间
        pop.ObjV = np.dot(pop.ObjV, self.ratioVector)

        """增加约束"""
        pop.CV = np.abs(np.dot(Vars, self.CountVector) - self.recommend_num)


def recommendSinglePr(EXPScoreVector, RSPScoreVector, ACTScoreVector, recommendNum, candicateList):
    candicateNum = candicateList.__len__()  # 候选者数量，也是编码的长度
    problem = GAProblem(candicateNum, EXPScoreVector, RSPScoreVector, ACTScoreVector, recommendNum)
    """种群设置"""
    Encoding = 'RI'  # 编码方式
    NIND = 200  # 种群规模，论文上200
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象
    """算法参数设置"""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化模板对象
    myAlgorithm.MAXGEN = 100  # 最大迭代数量
    myAlgorithm.drawing = 0  # 设置绘图方式（0:不绘图 1：结果图  2：过程动画）
    """调用模板种群进化
    调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    NDSet = myAlgorithm.run()  # 执行算法模板，获取非支配种群
    NDSet.save()  # 结果保存文件

    # # 输出
    # print("用时：%f秒" % (myAlgorithm.passTime))
    # print("评价次数：%d" % (myAlgorithm.evalsNum))
    # print("非支配个体数:%d" % (NDSet.sizes))
    # print('单位时间找到帕累托前沿点个数:%d' % (int(NDSet.sizes //
    #                                  myAlgorithm.passTime)))

    # 计算指标
    # PF = problem.getReferObjV()  # 获取真实前沿
    # if PF is not None and NDSet.sizes != 0:
    #     GD = ea.indicator.GD(NDSet.ObjV, PF)  # 计算GD指标
    #     IGD = ea.indicator.IGD(NDSet.ObjV, PF)  # 计算IGD指标
    #     HV = ea.indicator.HV(NDSet.ObjV, PF)  # 计算Spacing 指标
    #     Spacing = ea.indicator.Spacing(NDSet.ObjV)
    #     print('GD', GD)
    #     print('IGD', IGD)
    #     print('HV', HV)
    #     print('Spacing', Spacing)
    #
    # """进化过程指标追踪分析"""
    # if PF is not None:
    #     metricName = [['IGD'], ['HV']]
    #     [NDSet_trace, Metrics] = ea.indicator.moea_tracking(myAlgorithm.pop_trace,
    #                                                         PF, metricName, problem.maxormins)
    #     ea.trcplot(Metrics, labels=metricName, titles=metricName)

    # 自己的做法 论文中没有   计算200个解每个人选的次数，按照次数依次推荐
    candicateCount = np.dot(np.ones((candicateNum, NIND)), NDSet.Chrom)
    scores = {}
    for index, c in enumerate(candicateList):
        scores[c] = candicateCount[0][index]
    return [x[0] for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]]
