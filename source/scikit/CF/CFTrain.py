# coding=gbk
import math
import os
import time
from datetime import datetime

from pandas import DataFrame

from source.config.projectConfig import projectConfig
from source.scikit.CF.LFM import LFM
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper


class CFTrain:

    @staticmethod
    def testCFAlgorithm(project, dates):
        """整合 训练数据"""
        recommendNum = 5  # 推荐数量
        excelName = f'outputCF_{project}.xls'
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
            recommendList, answerList, prList, convertDict, trainSize = CFTrain.algorithmBody(date, project,
                                                                                              recommendNum)

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
    def recoverName(recommendList, answerList, convertDict):
        """通过映射字典把人名还原"""
        tempDict = {k: v for v, k in convertDict.items()}
        recommendList = [[tempDict[i] for i in x] for x in recommendList]
        answerList = [[tempDict[i] for i in x] for x in answerList]
        return recommendList, answerList


    @staticmethod
    def algorithmBody(date, project, recommendNum=5):

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
            filename = projectConfig.getCFDataPath() + os.sep + f'CF_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = CFTrain.preProcess(df, date)

        prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
        prList.sort()

        recommendList, answerList = CFTrain.RecommendByCF(date, train_data, train_data_y, test_data,
                                                          test_data_y, convertDict, recommendNum=recommendNum)

        """新增返回测试 训练集大小，用于做统计"""
        trainSize = (train_data.shape, test_data.shape)
        print(trainSize)

        return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def preProcess(df, dates):
        """参数说明
                    df：读取的dataframe对象
                    dates:四元组，后两位作为测试的年月 (,,year,month)
                   """

        """注意： 输入文件中已经带有列名了"""

        """空comment的review包含na信息，但作为结果集是有用的，所以只对训练集去掉na"""
        # """处理NAN"""
        # df.dropna(how='any', inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # df.fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['pr_author', 'reviewer'])

        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pull_number')))

        """对已经有的特征向量和标签做训练集的拆分"""
        train_data = df.loc[df['label'] == False].copy(deep=True)
        test_data = df.loc[df['label']].copy(deep=True)

        train_data.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)

        """8ii处理NAN"""
        train_data.dropna(how='any', inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        train_data.fillna(value='', inplace=True)

        """过滤掉评论时间在数据集时间范围内之后的数据"""
        # 结束时间：数据集pr最晚的创建时间
        pr_created_time_data = train_data['pr_created_at']
        end_time = max(pr_created_time_data.to_list())
        train_data = train_data[train_data['comment_at'] <= end_time]
        train_data.reset_index(drop=True, inplace=True)

        test_data_y = {}
        for pull_number in test_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['reviewer'])['reviewer'])
            test_data_y[pull_number] = reviewers

        train_data_y = {}
        for pull_number in train_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['reviewer'])['reviewer'])
            train_data_y[pull_number] = reviewers

        return train_data, train_data_y, test_data, test_data_y, convertDict

    @staticmethod
    def RecommendByCF(date, train_data, train_data_y, test_data, test_data_y, convertDict, recommendNum=5):
        """协同过滤推荐算法"""
        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pull_number')))
        testTuple = sorted(testDict.items(), key=lambda x: x[0])

        """The start time and end time are highly related to the selection of training set"""
        # 开始时间：数据集pr最早的创建时间
        pr_created_time_data = train_data['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())

        # 结束时间：数据集pr最晚的创建时间
        pr_created_time_data = train_data['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())


        """构造review -> request关系"""
        print("start building reviewer -> request relations....")
        start = datetime.now()
        Mat = {}
        grouped_train_data = train_data.groupby([train_data['pull_number'], train_data['reviewer']])
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = CFTrain.caculateWeight(group, start_time, end_time)
            if not Mat.__contains__(relation[1]):
                Mat[relation[1]] = {}
            Mat[relation[1]][relation[0]] = weight

        print("finish building reviewer -> request relations. cost time: {0}s".format(datetime.now() - start))

        # print("start train lfm model....")
        # start = datetime.now()
        # lfm = LFM(Mat)
        # lfm.train()
        # print("finish train lfm model. cost time: {0}s".format(datetime.now() - start))

        candidates = Mat.keys()
        for test_pull_number, test_df in testTuple:
            answerList.append(test_data_y[test_pull_number])
            candidates_score = {}
            colSet = CFTrain.transform(test_pull_number, train_data, test_data)
            for candidate in candidates:
                score = 0
                # lfm_effect = 0
                for pr in colSet:
                    if pr in Mat[candidate].keys():
                        score += Mat[candidate][pr]
                    # else:
                    #     lfm_score = lfm.predict(candidate, pr)
                    #     score += lfm_score
                    #     lfm_effect += lfm_score
                candidates_score[candidate] = score
            scoresTuple = sorted(candidates_score.items(), key=lambda x: x[1], reverse=True)
            scoresTuple = scoresTuple[0:recommendNum]
            recommend_reviewers = list(map(lambda x: x[0], scoresTuple))
            recommendList.append(recommend_reviewers)
        return recommendList, answerList

    @staticmethod
    def caculateWeight(comment_records, start_time, end_time):
        weight_lambda = 0.8
        weight = 0
        comment_records.drop(columns=['filename'], inplace=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)
        """遍历每条评论，计算权重"""
        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['comment_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))
            """计算t值: the element t(ij,r,n) is a time-sensitive factor """
            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * t
            weight += cm_weight
        return weight

    @staticmethod
    def transform(target_pr, train_data, test_data):
        print("start to find similar pr....")
        start = datetime.now()
        """查找相似pr列表"""
        prList = list(train_data.drop_duplicates(['pull_number'])['pull_number'])
        prList.sort()

        scores = {}
        paths1 = list(set(test_data[test_data['pull_number'] == target_pr]['filename']))
        for pr in prList:
            if pr == target_pr:
                continue
            score = 0
            paths2 = list(set(train_data[train_data['pull_number'] == pr]['filename']))
            for filename1 in paths1:
                for filename2 in paths2:
                    score += FPSAlgorithm.LCSubseq_2(filename1, filename2)
            score /= paths1.__len__() * paths2.__len__()
            scores[pr] = score
        scoresTuple = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        len = scoresTuple.__len__()
        scoresTuple = scoresTuple[0:math.floor(len/10)]
        res = list(map(lambda x: x[0], scoresTuple))
        print("start to find similar pr. cost time: {0}s".format(datetime.now() - start))
        return res

if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
             (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # dates = [(2017, 1, 2018, 1)]
    # projects = ['opencv', 'cakephp', 'yarn', 'akka', 'django', 'react', 'xbmc', 'symfony', 'babel', 'angular']
    # projects = ['opencv']
    projects = ['cakephp', 'yarn', 'akka', 'django', 'angular']
    for p in projects:
        CFTrain.testCFAlgorithm(p, dates)