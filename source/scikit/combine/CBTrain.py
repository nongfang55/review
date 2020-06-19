# coding=gbk
from math import floor

from source.scikit.FPS.FPSTrain import FPSTrain
from source.scikit.IR.IRTrain import IRTrain
from source.scikit.ML.MLTrain import MLTrain
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.scikit.service.SortAlgorithmUtils import SortAlgorithmUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils


class CBTrain:
    """混合式算法 提供所有给定算法的排列组合"""

    @staticmethod
    def testCBAlgorithmsByMultipleLabels(projects, dates, algorithms):
        """
             algorithm : 混合算法，提供算法的排列组合
             项目 -> 日期 -> 算法排列组合
             每一个项目占一个文件位置  每一个算法组合占一页
          """
        recommendNum = 5  # 推荐数量
        for project in projects:
            excelName = f'outputCB_{project}.xlsx'
            sheetName = 'result'

            """初始化excel文件"""
            ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

            """对不同时间做一个综合统计
               组合的int -> [[],[]....]
            """
            topks = {}
            mrrs = {}
            precisionks = {}
            recallks = {}
            fmeasureks = {}
            """初始化"""
            for i in range(1, 2 ** algorithms.__len__()):
                topks[i] = []
                mrrs[i] = []
                precisionks[i] = []
                recallks[i] = []
                fmeasureks[i] = []

            for date in dates:
                """获得不同算法的推荐列表，答案和pr列表"""
                """不同算法预处理可能会筛去一些pr  pr列表用于做统一"""
                prs = []
                recommendLists = []
                answerLists = []

                """计算不同人之前在训练集review的次数 作为后面综合统计的第二依据"""
                reviewerFreq = DataProcessUtils.getReviewerFrequencyDict(project, date)

                for algorithm in algorithms:
                    print(f"project:{project},  date:{date}, algorithm:{algorithm}")
                    """根据算法获得推荐列表"""
                    recommendList, answerList, prList, convertDict, trainSize = CBTrain.algorithmBody(date, project, algorithm,
                                                                                           recommendNum)
                    # print(recommendList)
                    print("trainSize:", trainSize)

                    """人名还原"""
                    recommendList, answerList = CBTrain.recoverName(recommendList, answerList, convertDict)

                    # print(recommendList)

                    prs.append(prList)
                    recommendLists.append(recommendList)
                    answerLists.append(answerList)

                """不同算法按照共有的pr 顺序调整"""
                prs, recommendLists, answerLists = CBTrain.normList(prs, recommendLists, answerLists)

                """貌似推荐是人名也可以做效果评估 暂时不转化"""
                # CBTrain.convertNameToNumber(recommendLists, answerLists)

                """对不同算法做排列组合"""
                for i in range(1, 2 ** algorithms.__len__()):
                    tempRecommendList = []
                    """不同算法测试的 answer列表相同，取一个即可"""
                    answer = answerLists[0]

                    involve = [0] * algorithms.__len__()
                    k = i
                    for j in range(0, algorithms.__len__()):
                        involve[algorithms.__len__() - j - 1] = k % 2
                        k = floor(k / 2)
                    """组合算法label为excel sheetName"""
                    label = ''
                    for j in range(0, algorithms.__len__()):
                        if involve[j] == 1:
                            if label != '':
                                label = label + '_'
                            label = label + algorithms[j]
                            tempRecommendList.append(recommendLists[j])
                    sheetName = label
                    ExcelHelper().addSheet(filename=excelName, sheetName=sheetName)
                    """波达计数 结合不同投票选出最终名单"""
                    finalRecommendList = []
                    for j in range(0, answer.__len__()):
                        recommendList = SortAlgorithmUtils.BordaCountSortWithFreq([x[j] for x in tempRecommendList],
                                                                                  reviewerFreq)
                        finalRecommendList.append(recommendList)

                    """评价指标"""
                    topk, mrr, precisionk, recallk, fmeasurek = \
                        DataProcessUtils.judgeRecommend(finalRecommendList, answer, recommendNum)

                    """结果写入excel"""
                    DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date)

                    """累积评价指标"""
                    topks[i].append(topk)
                    mrrs[i].append(mrr)
                    precisionks[i].append(precisionk)
                    recallks[i].append(recallk)
                    fmeasureks[i].append(fmeasurek)

            """对指标做综合评判"""
            for i in range(1, 2 ** algorithms.__len__()):
                involve = [0] * algorithms.__len__()
                k = i
                for j in range(0, algorithms.__len__()):
                    involve[algorithms.__len__() - j - 1] = k % 2
                    k = floor(k / 2)
                """组合算法label为excel sheetName"""
                label = ''
                for j in range(0, algorithms.__len__()):
                    if involve[j] == 1:
                        if label != '':
                            label = label + '_'
                        label = label + algorithms[j]
                sheetName = label
                DataProcessUtils.saveFinallyResult(excelName, sheetName, topks[i], mrrs[i], precisionks[i], recallks[i],
                                                   fmeasureks[i])

    @staticmethod
    def algorithmBody(date, project, algorithmName, recommendNum=5):
        if algorithmName == StringKeyUtils.STR_ALGORITHM_FPS:
            return FPSTrain.algorithmBody(date, project, recommendNum)
        elif algorithmName == StringKeyUtils.STR_ALGORITHM_IR:
            return IRTrain.algorithmBody(date, project, recommendNum)
        elif algorithmName == StringKeyUtils.STR_ALGORITHM_RF_M:
            return MLTrain.algorithmBody(date, project, algorithmType=0, recommendNum=recommendNum, featureType=1)
        elif algorithmName == StringKeyUtils.STR_ALGORITHM_SVM:
            return MLTrain.algorithmBody(date, project, algorithmType=7, recommendNum=recommendNum, featureType=1)

    @staticmethod
    def recoverName(recommendList, answerList, convertDict):
        """通过映射字典把人名还原"""
        tempDict = {k: v for v, k in convertDict.items()}
        recommendList = [[tempDict[i] for i in x] for x in recommendList]
        answerList = [[tempDict[i] for i in x] for x in answerList]
        return recommendList, answerList

    @staticmethod
    def convertNameToNumber(recommendLists, answerLists):
        pass

    @staticmethod
    def normList(prs, recommendLists, answerLists):
        """不同的推荐由于预处理的原因，可能测试的pr的case和顺序不同，以防万一做规范化"""

        """单个输入不用规范"""
        if prs.__len__() == 1:
            return prs, recommendLists, answerLists

        normRecommendLists = []
        normAnswerLists = []
        normPrs = []

        """从所有pr中找出共有的"""
        normPrs = prs[0].copy()
        for prList in prs:
            normPrs = [i for i in normPrs if i in prList]

        """依据共有的pr列表对答案做排序和筛选"""
        pos = -1
        for prList in prs:
            pos += 1
            recommendList = []
            answerList = []
            originRecommendList = recommendLists[pos]
            originAnswerList = answerLists[pos]
            for pr in normPrs:
                index = prList.index(pr)
                recommendList.append(originRecommendList[index])
                answerList.append(originAnswerList[index])
            normRecommendLists.append(recommendList)
            normAnswerLists.append(answerList)
        return normPrs, normRecommendLists, normAnswerLists


if __name__ == '__main__':
    # dates = [(2018, 1, 2019, 11), (2018, 1, 2019, 12)]
    # dates = [(2018, 1, 2019, 5), (2018, 1, 2019, 6), (2018, 1, 2019, 7), (2018, 1, 2019, 8), (2018, 1, 2019, 9)]
    # dates = [(2019, 1, 2019, 8), (2019, 1, 2019, 9)]
    # dates = [(2018, 1, 2019, 3)]
    dates = [(2018, 1, 2019, 1)]
    projects = ['cakephp']
    # projects = ['bitcoin']
    algorithms = [StringKeyUtils.STR_ALGORITHM_RF_M]
    # algorithms = [StringKeyUtils.STR_ALGORITHM_SVM]
    CBTrain.testCBAlgorithmsByMultipleLabels(projects, dates, algorithms)
