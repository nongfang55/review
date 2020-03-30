# coding=gbk
import os
import time
from datetime import datetime
from math import sqrt

import pandas
from gensim import corpora, models

from source.config.projectConfig import projectConfig
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper


class IRTrain:
    """作为基于信息检索的reviewer推荐"""

    @staticmethod
    def testIRAlgorithm(project, dates):  # 多个case, 元组代表总共的时间跨度,最后一个月用于测试
        """
           algorithm : 基于信息检索
        """

        recommendNum = 5  # 推荐数量
        excelName = f'outputIR.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

        df = None

        for date in dates:
            startTime = datetime.now()
            for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
                y = int((i - i % 12) / 12)
                m = i % 12
                if m == 0:
                    m = 12
                    y = y - 1

                print(y, m)

                filename = projectConfig.getRootPath() + r'\data\train\all' + \
                           os.sep + f'ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
                if df is None:
                    df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                else:
                    temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                    df = df.append(temp)  # 合并

            df.reset_index(inplace=True, drop=True)
            """df做预处理"""
            train_data, train_data_y, test_data, test_data_y = IRTrain.preProcess(df, (date[2], date[3]))

            print("train data:", train_data.__len__())
            # print("traindatay:", train_data_y)
            print("test data:", test_data.__len__())
            # print("testdatay:", test_data_y)

            """根据算法获得推荐列表"""
            recommendList, answerList = IRTrain.RecommendByIR(train_data, train_data_y, test_data,
                                                              test_data_y, recommendNum=recommendNum)

            """根据推荐列表做评价"""
            topk, mrr = DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

            """结果写入excel"""
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

    @staticmethod
    def preProcess(df, testDate):
        """参数说明
         df：读取的dataframe对象
         testDate:作为测试的年月 (year,month)
        """

        """注意： 输入文件中已经带有列名了"""

        """处理NAN"""
        df.fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == testDate[0] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == testDate[1]))

        """先对输入数据做精简 只留下感兴趣的数据"""
        df = df[['pr_number', 'review_id', 'review_comment_id', 'pr_title', 'pr_body',
                 'commit_commit_message', 'review_comment_body', 'review_user_login', 'label']].copy(deep=True)

        print("before filter:", df.shape)
        df.drop_duplicates(['pr_number', 'review_id', 'review_comment_id'], inplace=True)
        print("after filter:", df.shape)
        # print(df)

        """处理一个review有多个comment的场景  把comment都合并到一起"""

        comments = df['review_comment_body'].groupby(df['review_id']).sum()  # 一个review的所有评论字符串连接
        print(comments.index)

        """留下去除comment之后的信息 去重"""
        df = df[['pr_number', 'review_id', 'pr_title', 'pr_body', 'review_user_login',
                 'commit_commit_message', 'label']].copy(deep=True)
        print(df.shape)
        df.drop_duplicates(inplace=True)
        print(df.shape)

        df = (df.join(comments, on=['review_id'], how='inner')).copy(deep=True).reset_index(drop=True)
        print(df)

        """对人名字做数字处理"""
        DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        print(df)


        """先尝试所有信息团在一起"""

        """用于收集所有文本向量分词"""
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

        textList = []
        for row in df.itertuples(index=True, name='Pandas'):
            """获取pull request的标题"""
            pr_title = getattr(row, 'pr_title')
            pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]

            """初步尝试提取词干效果反而下降了 。。。。"""

            # """对单词做提取词干"""
            # pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
            textList.append(pr_title_word_list)

            """pull request的body"""
            pr_body = getattr(row, 'pr_body')
            pr_body_word_list = [x for x in FleshReadableUtils.word_list(pr_body) if x not in stopwords]
            # """对单词做提取词干"""
            # pr_body_word_list = nltkFunction.stemList(pr_body_word_list)
            textList.append(pr_body_word_list)

            """review 的comment"""
            review_comment = getattr(row, 'review_comment_body')
            review_comment_word_list = [x for x in FleshReadableUtils.word_list(review_comment) if x not in stopwords]
            # """对单词做提取词干"""
            # review_comment_word_list = nltkFunction.stemList(review_comment_word_list)
            textList.append(review_comment_word_list)

            """review的commit的 message"""
            commit_message = getattr(row, 'commit_commit_message')
            commit_message_word_list = [x for x in FleshReadableUtils.word_list(commit_message) if x not in stopwords]
            # """对单词做提取词干"""
            # commit_message_word_list = nltkFunction.stemList(commit_message_word_list)
            textList.append(commit_message_word_list)

        print(textList.__len__())

        """对分词列表建立字典 并提取特征数"""
        dictionary = corpora.Dictionary(textList)
        print('词典：', dictionary)

        feature_cnt = len(dictionary.token2id)
        print("词典特征数：", feature_cnt)

        """根据词典建立语料库"""
        corpus = [dictionary.doc2bow(text) for text in textList]
        print('语料库:', corpus)

        """语料库训练TF-IDF模型"""
        tfidf = models.TfidfModel(corpus)

        """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
        wordVectors = []
        for i in range(0, df.shape[0]):
            words = []
            for j in range(0, 4):
                words.extend(textList[4 * i + j])
            # print(words)
            wordVectors.append(dict(tfidf[dictionary.doc2bow(words)]))
        print(wordVectors)

        """对已经有的本文特征向量和标签做训练集和测试集的拆分"""

        train_data_y = df.loc[df['label'] == False]['review_user_login'].copy(deep=True)
        test_data_y = df.loc[df['label']]['review_user_login'].copy(deep=True)

        """训练集"""
        print(train_data_y.index)
        train_data = [wordVectors[x] for x in train_data_y.index]
        train_data_y.reset_index(drop=True, inplace=True)

        """测试集"""
        print(test_data_y.index)
        test_data = [wordVectors[x] for x in test_data_y.index]
        test_data_y.reset_index(drop=True, inplace=True)

        """返回特征是一个稀疏矩阵的字典"""

        return train_data, train_data_y, test_data, test_data_y

    @staticmethod
    def RecommendByIR(train_data, train_data_y, test_data, test_data_y, recommendNum=5):
        """使用信息检索

        """""
        initScore = {}  # 建立分数字典
        recommendList = []  # 最后多个case推荐列表
        for y in train_data_y:
            if initScore.get(y, None) is None:
                initScore[y] = 0

        for targetData in test_data:  # 对每一个case做推荐
            recommendScore = initScore.copy()  # 复制重复利用
            pos = 0
            for trainData in train_data:
                reviewer = train_data_y[pos]
                pos += 1
                #
                # print("targetData:", targetData)
                # print("trainData:", trainData)
                score = IRTrain.cos(targetData, trainData)
                # print("score:", score)
                recommendScore[reviewer] += score

            targetRecommendList = [x[0] for x in
                                   sorted(recommendScore.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]]
            # print(targetRecommendList)
            recommendList.append(targetRecommendList)
        # print(recommendList)
        answer = [[x] for x in test_data_y]
        # print(answer)
        return [recommendList, answer]

    @staticmethod
    def cos(dict1, dict2):
        """计算两个代码稀疏矩阵字典的计算余弦"""
        if isinstance(dict1, dict) and isinstance(dict2, dict):
            """先计算模长"""
            l1 = 0
            for v in dict1.values():
                l1 += v * v
            l2 = 0
            for v in dict2.values():
                l2 += v * v

            mul = 0
            """计算向量相乘"""
            for key in dict1.keys():
                if dict2.get(key, None) is not None:
                    mul += dict1[key] * dict2[key]
            return mul / (sqrt(l1) * sqrt(l2))


if __name__ == '__main__':
    dates = [(2018, 4, 2019, 4), (2018, 4, 2019, 3), (2018, 4, 2019, 2), (2018, 4, 2019, 1),
             (2018, 4, 2018, 12), (2018, 4, 2018, 11), (2018, 4, 2018, 10)]
    IRTrain.testIRAlgorithm('scala', dates)
