# coding=gbk
import math
import os
import time
from datetime import datetime
from math import sqrt

import numpy
import pandas
from gensim import corpora, models
from pandas import DataFrame

from source.config.projectConfig import projectConfig
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.ML.MLTrain import MLTrain
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class IR_ACTrain:
    """作为基于信息检索的reviewer推荐  但是是AC算法提到的方式"""

    @staticmethod
    def testAlgorithm(project, dates, filter_train=False, filter_test=False, error_analysis=False,
                      test_type=StringKeyUtils.STR_TEST_TYPE_SLIDE):  # 多个case, 元组代表总共的时间跨度,最后一个月用于测试
        """
           algorithm : 基于信息检索
        """

        recommendNum = 5  # 推荐数量
        excelName = f'outputIR_AC_{project}_{filter_train}_{filter_test}_{error_analysis}.xlsx'
        sheetName = 'result'

        """计算累积数据"""
        topks = []
        mrrs = []
        precisionks = []
        recallks = []
        fmeasureks = []
        recommend_positive_success_pr_ratios = []  # pr 中有推荐成功人选的比例
        recommend_positive_success_time_ratios = []  # 推荐pr * 人次 中有推荐成功人选的频次比例
        recommend_negative_success_pr_ratios = []  # pr 中有推荐人选Hit 但被滤掉的pr的比例
        recommend_negative_success_time_ratios = []  # 推荐pr * 人次中有推荐人选Hit 但是被滤掉的pr的比例
        recommend_positive_fail_pr_ratios = []  # pr 中有推荐人选推荐错误的pr比例
        recommend_positive_fail_time_ratios = []  # pr 中有pr * 人次有推荐错误的频次比例
        recommend_negative_fail_pr_ratios = []  # pr 中有推荐人选不知道是否正确的比例
        recommend_negative_fail_time_ratios = []  # pr中有pr * 人次有不知道是否正确的比例
        error_analysis_datas = None

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])
        for date in dates:
            startTime = datetime.now()
            """根据推荐列表做评价"""

            recommendList, answerList, prList, convertDict, trainSize = IR_ACTrain.algorithmBody(date, project,
                                                                                                 recommendNum,
                                                                                                 filter_train=filter_train,
                                                                                                 filter_test=filter_test,
                                                                                                 test_type=test_type)

            topk, mrr, precisionk, recallk, fmeasurek = \
                DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

            topks.append(topk)
            mrrs.append(mrr)
            precisionks.append(precisionk)
            recallks.append(recallk)
            fmeasureks.append(fmeasurek)

            error_analysis_data = None
            filter_answer_list = None
            if error_analysis:
                if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
                    y = date[2]
                    m = date[3]
                    filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerData(project, date, prList,
                                                                                             convertDict, filename,
                                                                                             'review_user_login',
                                                                                             'pr_number')
                elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                    fileList = []
                    for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
                        y = int((i - i % 12) / 12)
                        m = i % 12
                        if m == 0:
                            m = 12
                            y = y - 1
                        fileList.append(
                            projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv')

                    filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerDataByIncrement(project, prList,
                                                                                                        convertDict,
                                                                                                        fileList,
                                                                                                        'review_user_login',
                                                                                                        'pr_number')

                # recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio, recommend_negative_success_pr_ratio, \
                # recommend_negative_success_time_ratio, recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio, \
                # recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio = DataProcessUtils.errorAnalysis(
                #     recommendList, answerList, filter_answer_list, recommendNum)
                # error_analysis_data = [recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio,
                #                        recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio,
                #                        recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio,
                #                        recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio]

                recommend_positive_success_pr_ratio, recommend_negative_success_pr_ratio, recommend_positive_fail_pr_ratio, \
                recommend_negative_fail_pr_ratio = DataProcessUtils.errorAnalysis(
                    recommendList, answerList, filter_answer_list, recommendNum)
                error_analysis_data = [recommend_positive_success_pr_ratio,
                                       recommend_negative_success_pr_ratio,
                                       recommend_positive_fail_pr_ratio,
                                       recommend_negative_fail_pr_ratio]

                # recommend_positive_success_pr_ratios.append(recommend_positive_success_pr_ratio)
                # recommend_positive_success_time_ratios.append(recommend_positive_success_time_ratio)
                # recommend_negative_success_pr_ratios.append(recommend_negative_success_pr_ratio)
                # recommend_negative_success_time_ratios.append(recommend_negative_success_time_ratio)
                # recommend_positive_fail_pr_ratios.append(recommend_positive_fail_pr_ratio)
                # recommend_positive_fail_time_ratios.append(recommend_positive_fail_time_ratio)
                # recommend_negative_fail_pr_ratios.append(recommend_negative_fail_pr_ratio)
                # recommend_negative_fail_time_ratios.append(recommend_negative_fail_time_ratio)

                recommend_positive_success_pr_ratios.append(recommend_positive_success_pr_ratio)
                recommend_negative_success_pr_ratios.append(recommend_negative_success_pr_ratio)
                recommend_positive_fail_pr_ratios.append(recommend_positive_fail_pr_ratio)
                recommend_negative_fail_pr_ratios.append(recommend_negative_fail_pr_ratio)

            if error_analysis_data:
                # error_analysis_datas = [recommend_positive_success_pr_ratios, recommend_positive_success_time_ratios,
                #                         recommend_negative_success_pr_ratios, recommend_negative_success_time_ratios,
                #                         recommend_positive_fail_pr_ratios, recommend_positive_fail_time_ratios,
                #                         recommend_negative_fail_pr_ratios, recommend_negative_fail_time_ratios]
                error_analysis_datas = [recommend_positive_success_pr_ratios,
                                        recommend_negative_success_pr_ratios,
                                        recommend_positive_fail_pr_ratios,
                                        recommend_negative_fail_pr_ratios]

            """结果写入excel"""
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            print("cost time:", datetime.now() - startTime)

        """推荐错误可视化"""
        DataProcessUtils.recommendErrorAnalyzer2(error_analysis_datas, project,
                                                 f'IR_AC_{test_type}_{filter_train}_{filter_test}')

        """计算历史累积数据"""
        DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks,
                                           fmeasureks, error_analysis_datas)

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, filter_train=False, filter_test=False,
                      test_type=StringKeyUtils.STR_TEST_TYPE_SLIDE):

        """提供单个日期和项目名称
           返回推荐列表和答案
           这个接口可以被混合算法调用
        """
        df = None
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            print(y, m)
            filename = None
            if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
                if i < date[2] * 12 + date[3]:
                    if filter_train:
                        filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    else:
                        filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    if filter_test:
                        filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    else:
                        filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                if filter_test:
                    filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getIR_ACDataPath() + os.sep + f'IR_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
            df.reset_index(inplace=True, drop=True)
            """df做预处理"""
            """预处理新增返回测试pr列表 2020.4.11"""
            train_data, train_data_y, test_data, test_data_y, convertDict = IR_ACTrain.preProcessBySlide(df, date)

            prList = list(test_data['pr_number'])

            """根据算法获得推荐列表"""
            recommendList, answerList = IR_ACTrain.RecommendByIR_AC_SLIDE(train_data, train_data_y, test_data,
                                                                          test_data_y, recommendNum=recommendNum)
            trainSize = (train_data.shape[0], test_data.shape[0])
            return recommendList, answerList, prList, convertDict, trainSize
        elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
            """df做预处理"""
            """新增人名映射字典"""
            test_data, test_data_y, convertDict = IR_ACTrain.preProcessByIncrement(df, date)

            prList = list(test_data.drop_duplicates(['pr_number'])['pr_number'])
            """增量预测第一个pr不预测"""
            prList.sort()
            prList.pop(0)
            recommendList, answerList = IR_ACTrain.RecommendByIR_AC_INCREMENT(test_data, test_data_y,
                                                                              recommendNum=recommendNum)

            """新增返回测试 训练集大小，用于做统计"""

            """新增返回训练集 测试集大小"""
            trainSize = (test_data.shape)
            print(trainSize)

            # """输出推荐名单到文件"""
            # DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict)

            return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def preProcessByIncrement(df, dates):
        """参数说明
         df：读取的dataframe对象
         dates:作为测试的年月四元组
        """
        """注意： 输入文件中已经带有列名了"""

        """处理NAN"""
        df.dropna(how='any', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.fillna(value='', inplace=True)

        """创建时间转化为时间戳"""
        df['pr_created_at'] = df['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['pr_created_at'] = df['pr_created_at'] / (2400 * 36)

        """先对输入数据做精简 只留下感兴趣的数据"""
        df = df[['pr_number', 'pr_title', 'review_user_login', 'pr_created_at']].copy(deep=True)

        print("before filter:", df.shape)
        df.drop_duplicates(inplace=True)
        print("after filter:", df.shape)
        """对人名字做数字处理"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pr_number')))
        """先尝试所有信息团在一起"""
        df = df[['pr_number', 'pr_title', 'pr_created_at']].copy(deep=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        """用于收集所有文本向量分词"""
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

        textList = []
        for row in df.itertuples(index=False, name='Pandas'):
            tempList = []
            """获取pull request的标题"""
            pr_title = getattr(row, 'pr_title')
            pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]

            """初步尝试提取词干效果反而下降了 。。。。"""

            """对单词做提取词干"""
            pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
            tempList.extend(pr_title_word_list)
            textList.append(tempList)

        print(textList.__len__())
        """对分词列表建立字典 并提取特征数"""
        dictionary = corpora.Dictionary(textList)
        print('词典：', dictionary)

        feature_cnt = len(dictionary.token2id)
        print("词典特征数：", feature_cnt)

        """根据词典建立语料库"""
        corpus = [dictionary.doc2bow(text) for text in textList]
        # print('语料库:', corpus)
        """语料库训练TF-IDF模型"""
        tfidf = models.TfidfModel(corpus)

        """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
        wordVectors = []
        for i in range(0, df.shape[0]):
            wordVectors.append(dict(tfidf[dictionary.doc2bow(textList[i])]))

        """测试集"""
        test_data = wordVectors
        """填充为向量"""
        test_data = DataProcessUtils.convertFeatureDictToDataFrame(test_data, featureNum=feature_cnt)
        test_data['pr_number'] = list(df['pr_number'])
        test_data['pr_created_at'] = list(df['pr_created_at'])
        """问题转化为多标签问题
            train_data_y   [{pull_number:[r1, r2, ...]}, ... ,{}]
        """

        test_data_y = {}
        for pull_number in list(df['pr_number']):
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        """train_data ,test_data 最后一列是pr number test_data_y 的形式是dict"""
        return test_data, test_data_y, convertDict

    @staticmethod
    def preProcessBySlide(df, dates):
        """参数说明
         df：读取的dataframe对象
         dates:作为测试的年月四元组
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

        """创建时间转化为时间戳"""
        df['pr_created_at'] = df['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['pr_created_at'] = df['pr_created_at'] / (24 * 3600)

        """先对输入数据做精简 只留下感兴趣的数据"""
        df = df[['pr_number', 'pr_title', 'review_user_login', 'label', 'pr_created_at']].copy(deep=True)

        print("before filter:", df.shape)
        df.drop_duplicates(inplace=True)
        print("after filter:", df.shape)
        """对人名字做数字处理"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pr_number')))
        """先尝试所有信息团在一起"""
        df = df[['pr_number', 'pr_title', 'label', 'pr_created_at']].copy(deep=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        """用于收集所有文本向量分词"""
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

        textList = []
        for row in df.itertuples(index=False, name='Pandas'):
            tempList = []
            """获取pull request的标题"""
            pr_title = getattr(row, 'pr_title')
            pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]

            """初步尝试提取词干效果反而下降了 。。。。"""

            """对单词做提取词干"""
            pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
            tempList.extend(pr_title_word_list)
            textList.append(tempList)

        print(textList.__len__())
        """对分词列表建立字典 并提取特征数"""
        dictionary = corpora.Dictionary(textList)
        print('词典：', dictionary)

        feature_cnt = len(dictionary.token2id)
        print("词典特征数：", feature_cnt)

        """根据词典建立语料库"""
        corpus = [dictionary.doc2bow(text) for text in textList]
        # print('语料库:', corpus)
        """语料库训练TF-IDF模型"""
        tfidf = models.TfidfModel(corpus)

        """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
        wordVectors = []
        for i in range(0, df.shape[0]):
            wordVectors.append(dict(tfidf[dictionary.doc2bow(textList[i])]))

        """对已经有的本文特征向量和标签做训练集和测试集的拆分"""

        trainData_index = df.loc[df['label'] == False].index
        testData_index = df.loc[df['label'] == True].index

        """训练集"""
        train_data = [wordVectors[x] for x in trainData_index]
        """测试集"""
        test_data = [wordVectors[x] for x in testData_index]
        """填充为向量"""
        train_data = DataProcessUtils.convertFeatureDictToDataFrame(train_data, featureNum=feature_cnt)
        test_data = DataProcessUtils.convertFeatureDictToDataFrame(test_data, featureNum=feature_cnt)
        train_data['pr_number'] = list(df.loc[df['label'] == False]['pr_number'])
        test_data['pr_number'] = list(df.loc[df['label'] == True]['pr_number'])
        train_data['pr_created_at'] = list(df.loc[df['label'] == False]['pr_created_at'])
        test_data['pr_created_at'] = list(df.loc[df['label'] == True]['pr_created_at'])

        """问题转化为多标签问题
            train_data_y   [{pull_number:[r1, r2, ...]}, ... ,{}]
        """

        train_data_y = {}
        for pull_number in df.loc[df['label'] == False]['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            train_data_y[pull_number] = reviewers

        test_data_y = {}
        for pull_number in df.loc[df['label'] == True]['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        """train_data ,test_data 最后一列是pr number test_data_y 的形式是dict"""
        return train_data, train_data_y, test_data, test_data_y, convertDict

    @staticmethod
    def RecommendByIR_AC_SLIDE(train_data, train_data_y, test_data, test_data_y, recommendNum=5, l=-1):
        """使用信息检索  
           并且是多标签分类
        """""

        recommendList = []  # 最后多个case推荐列表
        answerList = []

        """对IR 矩阵乘法优化"""
        """计算train_data的矩阵"""
        df_train = train_data.copy(deep=True)
        df_train.drop(columns=['pr_created_at', 'pr_number'], inplace=True)
        """计算test_data矩阵"""
        df_test = test_data.copy(deep=True)
        df_test.drop(columns=['pr_created_at', 'pr_number'], inplace=True)

        """计算距离"""
        DIS = DataFrame(numpy.dot(df_test, df_train.T))  # train_num x test_num

        test_pr_list = tuple(test_data['pr_number'])
        train_pr_list = tuple(train_data['pr_number'])
        test_time_list = tuple(test_data['pr_created_at'])
        train_time_list = tuple(train_data['pr_created_at'])

        for index_test, pr_test in enumerate(test_pr_list):
            print(index_test)
            time1 = test_time_list[index_test]
            recommendScore = {}
            for index_train, pr_train in enumerate(train_pr_list):
                reviewers = train_data_y[pr_train]
                time2 = train_time_list[index_train]
                """计算时间差"""
                gap = (time1 - time2)
                """计算相似度不带上最后一个pr number"""
                score = DIS.iloc[index_test][index_train]
                score *= math.pow(gap, -l)
                #
                for reviewer in reviewers:
                    if recommendScore.get(reviewer, None) is None:
                        recommendScore[reviewer] = 0
                    recommendScore[reviewer] += score

            targetRecommendList = [x[0] for x in
                                   sorted(recommendScore.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]]
            # print(targetRecommendList)
            recommendList.append(targetRecommendList)
            answerList.append(test_data_y[pr_test])

        return [recommendList, answerList]

    @staticmethod
    def RecommendByIR_AC_INCREMENT(test_data, test_data_y, recommendNum=5, l=1):
        """使用信息检索  
           并且是多标签分类
        """""

        recommendList = []  # 最后多个case推荐列表
        answerList = []

        """对IR 矩阵乘法优化"""
        """计算test_data矩阵"""
        df_test = test_data.copy(deep=True)
        df_test.drop(columns=['pr_number', 'pr_created_at'], inplace=True)

        """计算距离"""
        DIS = DataFrame(numpy.dot(df_test, df_test.T))  # train_num x test_num

        test_data.sort_values(by=['pr_number'], inplace=True, ascending=True)
        prList = tuple(test_data['pr_number'])
        test_time_list = tuple(test_data['pr_created_at'])

        for test_index, test_pull_number in enumerate(prList):
            if test_index == 0:
                """第一个pr没有历史  无法推荐"""
                continue
            print("index:", test_index, " now:", prList.__len__())
            recommendScore = {}
            """添加正确答案"""
            time1 = test_time_list[test_index]

            train_pr_list = prList[:test_index]
            for train_index, train_pull_number in enumerate(train_pr_list):
                time2 = test_time_list[train_index]
                score = DIS.iloc[test_index][train_index]
                """计算时间差"""
                gap = (time1 - time2)
                """计算相似度不带上最后一个pr number"""
                score *= math.pow(gap, -l)
                for reviewer in test_data_y[train_pull_number]:
                    if recommendScore.get(reviewer, None) is None:
                        recommendScore[reviewer] = 0
                    recommendScore[reviewer] += score

            """人数不足随机填充"""
            if recommendScore.items().__len__() < recommendNum:
                for i in range(0, recommendNum):
                    recommendScore[f'{StringKeyUtils.STR_USER_NONE}_{i}'] = 0

            targetRecommendList = [x[0] for x in
                                   sorted(recommendScore.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]]
            recommendList.append(targetRecommendList)
            answerList.append(test_data_y[test_pull_number])

        return [recommendList, answerList]

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

            if mul == 0:
                return 0
            return mul / (sqrt(l1) * sqrt(l2))

    @staticmethod
    def cos2(tuple1, tuple2):
        if tuple1.__len__() != tuple2.__len__():
            raise Exception("tuple length not equal!")
        """计算两个元组的余弦"""
        """先计算模长"""
        l1 = 0
        for v in tuple1:
            l1 += v * v
        l2 = 0
        for v in tuple2:
            l2 += v * v
        mul = 0
        """计算向量相乘"""
        len = tuple1.__len__()
        for i in range(0, len):
            mul += tuple1[i] * tuple2[i]
        if mul == 0:
            return 0
        return mul / (sqrt(l1) * sqrt(l2))


if __name__ == '__main__':
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
    #          (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
    #          (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
    #          (2017, 1, 2018, 6)]
    dates = [(2018, 1, 2018, 2)]
    # projects = ['opencv', 'cakephp', 'akka', 'xbmc', 'babel', 'symfony', 'brew', 'django', 'netty', 'scikit-learn']
    projects = ['opencv']
    for p in projects:
        for test_type in [StringKeyUtils.STR_TEST_TYPE_SLIDE]:
            for t in [False]:
                if test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                    dates = [(2018, 1, 2018, 2)]
                IR_ACTrain.testAlgorithm(p, dates, filter_train=t, filter_test=t, error_analysis=True, test_type=test_type)
