# coding=gbk
import os
import time
from datetime import datetime

import pandas

from source.config.projectConfig import projectConfig
from source.data.bean.PullRequest import PullRequest
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service import MultisetHelper
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class PBTrain:

    @staticmethod
    def TestAlgorithm(project, dates):
        """整合 训练数据"""
        recommendNum = 5  # 推荐数量
        excelName = f'outputPB_{project}.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])
        for date in dates:
            startTime = datetime.now()
            recommendList, answerList, prList, convertDict, trainSize = PBTrain.algorithmBody(date, project,
                                                                                              recommendNum)
            """根据推荐列表做评价"""
            topk, mrr, precisionk, recallk, fmeasurek = \
                DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

            """结果写入excel"""
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

    @staticmethod
    def preProcess(df, dates):
        """参数说明
            df：读取的dataframe对象
            dates:四元组，后两位作为测试的年月 (,,year,month)
           """

        """注意： 输入文件中已经带有列名了"""

        t1 = datetime.now()

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

        """先尝试所有信息团在一起"""
        df = df[['pr_number', 'pr_title', 'pr_body', 'label']].copy(deep=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        """用于收集所有文本向量分词"""
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

        textList = []
        """由于特殊性  PB算法的训练集不是dataFrame
           { p1:set1, p2:set2, ... }
        """
        train_data = {}
        test_data = {}
        for row in df.itertuples(index=False, name='Pandas'):
            tempList = []
            """获取pull request的number"""
            pr_num = getattr(row, 'pr_number')
            label = getattr(row, 'label')

            """获取pull request的标题"""
            pr_title = getattr(row, 'pr_title')
            pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]

            """初步尝试提取词干效果反而下降了 。。。。"""

            """对单词做提取词干"""
            pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
            tempList.extend(pr_title_word_list)

            """pull request的body"""
            pr_body = getattr(row, 'pr_body')
            pr_body_word_list = [x for x in FleshReadableUtils.word_list(pr_body) if x not in stopwords]
            """对单词做提取词干"""
            pr_body_word_list = nltkFunction.stemList(pr_body_word_list)
            tempList.extend(pr_body_word_list)

            wordSet = MultisetHelper.WordMultiset()
            wordSet.add(tempList)

            if label == 0:
                train_data[pr_num] = wordSet
            else:
                test_data[pr_num] = wordSet

        print("train size:", train_data.items().__len__())
        print("test size:", test_data.items().__len__())

        """问题转化为多标签问题
            train_data_y   [{pull_number:[(r1, s1), (r2, s2), ...]}, ... ,{}]
            
            r 代表reviewer
            s 代表集合
        """

        train_data_y = {}
        for pull_number in df.loc[df['label'] == False]['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            tempDf = tagDict[pull_number][['review_user_login', 'comment_body']].copy(deep=True)
            commentDict = dict(list(tempDf.groupby('review_user_login')))
            reviewerList = []
            for reviewer in reviewers:
                commentDf = commentDict[reviewer]
                wordSet = MultisetHelper.WordMultiset()
                for row in commentDf.itertuples(index=False, name='Pandas'):
                    comment = getattr(row, 'comment_body')
                    comment_body_word_list = [x for x in FleshReadableUtils.word_list(comment) if x not in stopwords]
                    """对单词做提取词干"""
                    comment_body_word_list = nltkFunction.stemList(comment_body_word_list)
                    wordSet.add(comment_body_word_list)
                reviewerList.append((reviewer, wordSet))
            train_data_y[pull_number] = reviewerList

        test_data_y = {}
        for pull_number in df.loc[df['label'] == True]['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            tempDf = tagDict[pull_number][['review_user_login', 'comment_body']].copy(deep=True)
            commentDict = dict(list(tempDf.groupby('review_user_login')))
            reviewerList = []
            for reviewer in reviewers:
                commentDf = commentDict[reviewer]
                wordSet = MultisetHelper.WordMultiset()
                for row in commentDf.itertuples(index=False, name='Pandas'):
                    comment = getattr(row, 'comment_body')
                    comment_body_word_list = [x for x in FleshReadableUtils.word_list(comment) if x not in stopwords]
                    """对单词做提取词干"""
                    comment_body_word_list = nltkFunction.stemList(comment_body_word_list)
                    wordSet.add(comment_body_word_list)
                reviewerList.append((reviewer, wordSet))
            test_data_y[pull_number] = reviewerList

        print("preprocess cost time:", datetime.now() - t1)
        return train_data, train_data_y, test_data, test_data_y, convertDict

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
            filename = projectConfig.getPBDataPath() + os.sep + f'PB_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = PBTrain.preProcess(df, date)

        prList = None

        recommendList, answerList = PBTrain.RecommendByPB(train_data, train_data_y, test_data,
                                                          test_data_y, recommendNum=recommendNum)

        """新增返回测试 训练集大小，用于做统计"""

        """新增返回训练集 测试集大小"""
        trainSize = (train_data.items().__len__(), test_data.items().__len__())
        print(trainSize)

        return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def RecommendByPB(train_data, train_data_y, test_data, test_data_y, recommendNum=5):
        """使用用户画像
           并且是多标签分类
        """""

        recommendList = []  # 最后多个case推荐列表
        answerList = []

        t1 = datetime.now()
        """生成用户画像"""
        candicates = {}
        for pr_num in train_data.keys():
            prSet = train_data[pr_num]
            reviewers = train_data_y[pr_num]
            for reviewer, commentSet in reviewers:
                if candicates.get(reviewer, None) is None:
                    candicates[reviewer] = commentSet.copy()
                    candicates[reviewer].add(prSet)
                else:
                    candicates[reviewer].add(commentSet)
                    candicates[reviewer].add(prSet)
        print("user profile cost time:", datetime.now() - t1)

        for pr_num in test_data.keys():
            recommendScore = {}
            prSet = test_data[pr_num]
            """依次计算候选者的相关系数"""
            for candicate, profile in candicates.items():
                score = prSet.TverskyIndex(profile, 0, 1)
                recommendScore[candicate] = score
            targetRecommendList = [x[0] for x in
                                   sorted(recommendScore.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]]

            recommendList.append(targetRecommendList)
            answerList.append([x[0] for x in test_data_y[pr_num]])

        return [recommendList, answerList]


if __name__ == '__main__':
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
    #          (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
    #          (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6)]
    # dates = [(2017, 1, 2017, 2), (2017, 1, 2017, 3), (2017, 1, 2017, 4), (2017, 1, 2017, 5), (2017, 1, 2017, 6),
    #          (2017, 1, 2017, 7)]
    projects = ['cakephp']
    for p in projects:
        PBTrain.TestAlgorithm(p, dates)
