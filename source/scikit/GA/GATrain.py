# coding=gbk
import os
import time
from datetime import datetime

import pandas
from sklearn import preprocessing
from sklearn.preprocessing import normalize

from source.config.projectConfig import projectConfig
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.GA import GAProblem
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper
import numpy as np

class GATrain:

    """基于NSGA-II 的多目优化方案"""

    @staticmethod
    def TestAlgorithm(project, dates, response_limit_time=8, active_limit_time=10):
        """整合 训练数据"""
        recommendNum = 5  # 推荐数量
        excelName = f'outputGA_{project}.xlsx'
        sheetName = 'result'

        """计算累积数据"""
        topks = []
        mrrs = []
        precisionks = []
        recallks = []
        fmeasureks = []

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])
        for date in dates:
            startTime = datetime.now()
            recommendList, answerList, prList, convertDict, trainSize = GATrain.algorithmBody(date, project,
                                                                                              recommendNum,
                                                                                              response_limit_time,
                                                                                              active_limit_time)
            """根据推荐列表做评价"""
            topk, mrr, precisionk, recallk, fmeasurek = \
                DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

            topks.append(topk)
            mrrs.append(mrr)
            precisionks.append(precisionk)
            recallks.append(recallk)
            fmeasureks.append(fmeasurek)

            """结果写入excel"""
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

        """计算历史累积数据"""
        DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks,
                                           fmeasureks)

    @staticmethod
    def preProcess(df, dates):
        """参数说明
            df：读取的dataframe对象
            dates:四元组，后两位作为测试的年月 (,,year,month)
           """

        """注意： 输入文件中已经带有列名了"""

        """处理NAN"""
        df.dropna(how='any', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pr_number')))

        print("before drop:", df.shape)
        df = df.copy(deep=True)
        df.drop(columns=['review_user_login', 'review_created_at', 'repo_full_name'], inplace=True)
        df.drop_duplicates(['pr_number', 'pr_created_at', 'filename'], inplace=True)
        print("after drop:", df.shape)

        """对已经有的特征向量和标签做训练集的拆分"""
        train_data = df.loc[df['label'] == False].copy(deep=True)
        test_data = df.loc[df['label']].copy(deep=True)

        train_data.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)

        """问题转化为多标签问题
            train_data_y   [{pull_number:[(r1, t1), (r2, t2), ...]}, ... ,{}]
        """

        train_data_y = {}
        for pull_number in train_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = []
            for index, row in tagDict[pull_number].drop_duplicates(['review_user_login']).iterrows():
                reviewers.append((row['review_user_login'], row['review_created_at']))
            train_data_y[pull_number] = reviewers

        test_data_y = {}
        for pull_number in test_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = []
            for index, row in tagDict[pull_number].drop_duplicates(['review_user_login']).iterrows():
                reviewers.append((row['review_user_login'], row['review_created_at']))
            test_data_y[pull_number] = reviewers

        return train_data, train_data_y, test_data, test_data_y, convertDict


    @staticmethod
    def algorithmBody(date, project, recommendNum=5, response_limit_time=8, active_limit_time=10):

        """提供单个日期和项目名称
           返回推荐列表和答案
           这个接口可以被混合算法调用
        """
        print(date)
        df = None
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            # print(y, m)
            filename = projectConfig.getGADataPath() + os.sep + f'GA_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = GATrain.preProcess(df, date)

        prList = list(test_data.keys())
        prList.sort()

        recommendList, answerList = GATrain.RecommendByGA(train_data, train_data_y, test_data,
                                                          test_data_y, recommendNum=recommendNum,
                                                          response_limit_time=response_limit_time,
                                                          active_limit_time=active_limit_time)

        """新增返回测试 训练集大小，用于做统计"""

        """新增返回训练集 测试集大小"""
        trainSize = (list(set(train_data['pr_number'])).__len__(), list(set(test_data['pr_number'])).__len__())
        print(trainSize)

        return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def RecommendByGA(train_data, train_data_y, test_data, test_data_y, recommendNum=5, response_limit_time=8,
                      active_limit_time=10):
        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pr_number')))
        trainDict = dict(list(train_data.groupby('pr_number')))
        testTuple = sorted(testDict.items(), key=lambda x: x[0], reverse=True)
        now = min(test_data['pr_created_at'])  # 作为窗口时间基准
        now = datetime.strptime(now, "%Y-%m-%d %H:%M:%S")

        """遍历所有隶属数据，找到所有候选者，以及对应列表"""
        candicateList = []
        for v in train_data_y.values():
            for r in v:
                if r[0] not in candicateList:
                    candicateList.append(r[0])

        for test_pull_number, test_df in testTuple:
            EXPScore = {}  # 计算经验得分
            RSPScore = {}  # 回应分数
            ACTScore = {}  # 活跃分数
            """这里预先计算每个候选者的三个方面的分数，节约计算时间"""

            """先计算经验分数 和FPS并无二样"""
            for train_pull_number, train_df in trainDict.items():
                paths1 = list(train_df['filename'])
                paths2 = list(test_df['filename'])
                score = 0
                for filename1 in paths1:
                    for filename2 in paths2:
                        score += FPSAlgorithm.LCSubseq_2(filename1, filename2)
                score /= paths1.__len__() * paths2.__len__()
                for reviewer in train_data_y[train_pull_number]:
                    if EXPScore.get(reviewer[0], None) is None:
                        EXPScore[reviewer[0]] = 0
                    EXPScore[reviewer[0]] += score

            """其次计算响应概率分数"""
            for reviewer in candicateList:
                reviewCount = 0  # 总共review的次数
                usefulReviewCount = 0  # 在响应窗口中的review次数
                for pr_num, train_df in trainDict.items():
                    if reviewer in [x[0] for x in train_data_y[pr_num]]:
                        reviewCount += 1
                        """计算时间差"""
                        pr_created_at = datetime.strptime(list(train_df['pr_created_at'])[0], "%Y-%m-%d %H:%M:%S")
                        review_created_at = None
                        for r, t in train_data_y[pr_num]:
                            if r == reviewer:
                                review_created_at = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                        """计算时间差"""
                        if pr_created_at is not None:
                            review_gap_second = (review_created_at - pr_created_at).total_seconds()
                            window_gap_second = response_limit_time * 3600  # 小时换算
                            if review_gap_second <= window_gap_second:
                                usefulReviewCount += 1
                RSPScore[reviewer] = usefulReviewCount / reviewCount

            """最后计算活跃分数"""
            for pr_num, train_df in trainDict.items():
                # 计算now 和 该 pr 的间隔时间
                pr_created_at = datetime.strptime(list(train_df['pr_created_at'])[0], "%Y-%m-%d %H:%M:%S")
                pr_gap_second = (now - pr_created_at).total_seconds()
                window_gap_second = active_limit_time * 3600 * 24  # 天换算
                if pr_gap_second <= window_gap_second:
                    for reviewer in [x[0] for x in train_data_y[pr_num]]:
                        if ACTScore.get(reviewer, None) is None:
                            ACTScore[reviewer] = 1
                        else:
                            ACTScore[reviewer] += 1
                else:
                    for reviewer in [x[0] for x in train_data_y[pr_num]]:
                        if ACTScore.get(reviewer, None) is None:
                            ACTScore[reviewer] = 0
                        else:
                            ACTScore[reviewer] += 0
            """建立三个分数列向量，便于计算"""
            EXPScoreVector = np.array([[EXPScore[x] for x in candicateList]]).T
            RSPScoreVector = np.array([[RSPScore[x] for x in candicateList]]).T
            ACTScoreVector = np.array([[ACTScore[x] for x in candicateList]]).T

            """分数归一化"""
            EXPScoreVector = preprocessing.MinMaxScaler().fit_transform(EXPScoreVector)
            RSPScoreVector = preprocessing.MinMaxScaler().fit_transform(RSPScoreVector)
            ACTScoreVector = preprocessing.MinMaxScaler().fit_transform(ACTScoreVector)

            recommendList.append(GAProblem.recommendSinglePr(EXPScoreVector,
                                                             RSPScoreVector, ACTScoreVector, recommendNum,
                                                             candicateList))
            answerList.append([x[0] for x in test_data_y[test_pull_number]])

        return [recommendList, answerList]


if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6)]
    projects = ['akka', 'django', 'cakephp']
    active_limit_time = 10  # 活跃时间，单位为天
    response_limit_time = 8  # 回应窗口时间，单位为h
    for p in projects:
        GATrain.TestAlgorithm(p, dates, response_limit_time, active_limit_time)
