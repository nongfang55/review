# coding=gbk
import os
from datetime import datetime
import heapq
import time
from math import ceil

import numpy
from gensim import models, corpora
from pandas import DataFrame
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from source.config.projectConfig import projectConfig
from source.data.service.DataSourceHelper import  appendFilePathFeatureVector
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.ML.MultipleLabelAlgorithm import MultipleLabelAlgorithm
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper


class SVM_CTrain:

    @staticmethod
    def preProcess(df, date, project, isSTD=False, isNOR=False):
        """参数说明
        df：读取的dataframe对象
        testDate:作为测试的年月 (year,month)
        isSTD:对数据是否标准化
        isNOR:对数据是否归一化
        """
        print("start df shape:", df.shape)
        """过滤NA的数据"""
        df.dropna(axis=0, how='any', inplace=True)
        print("after fliter na:", df.shape)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == date[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == date[3]))
        df.reset_index(drop=True, inplace=True)

        """对人名字做数字处理"""
        """频率不过的评审者在编号之前就已经过滤了，不用考虑分类不连续的情况"""
        """这里reviewer_user_login 放在 第一个否则会影响candicateNum这个变量在后面的引用"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        print(df.shape)
        candicateNum = max(df.loc[df['label'] == 0]['review_user_login'])
        print("candicate Num:", candicateNum)

        """先对输入数据做精简 只留下感兴趣的数据"""
        df = df[['pr_number', 'pr_title', 'pr_body', 'review_user_login', 'label']].copy(deep=True)

        print("before filter:", df.shape)
        df.drop_duplicates(inplace=True)
        print("after filter:", df.shape)


        """先提前统计正确答案"""
        tagDict = dict(list(df.groupby('pr_number')))

        train_data = df.loc[df['label'] == 0].copy(deep=True)
        test_data = df.loc[df['label'] == 1].copy(deep=True)

        """问题转化为多标签问题
            train_data_y   [{pull_number:[r1, r2, ...]}, ... ,{}]
        """
        train_data_y = {}
        pull_number_list = train_data.drop_duplicates(['pr_number']).copy(deep=True)['pr_number']
        for pull_number in pull_number_list:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            train_data_y[pull_number] = reviewers

        train_data.drop(columns=['review_user_login'], inplace=True)
        train_data.drop_duplicates(inplace=True)
        train_data.drop_duplicates(subset=['pr_number'], inplace=True)
        """训练集 结果做出多标签分类通用的模式"""
        train_data_y = DataProcessUtils.convertLabelListToDataFrame(train_data_y, pull_number_list, candicateNum)

        test_data_y = {}
        pull_number_list = test_data.drop_duplicates(['pr_number']).copy(deep=True)['pr_number']
        for pull_number in test_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        test_data.drop(columns=['review_user_login'], inplace=True)
        test_data.drop_duplicates(inplace=True)
        """pr_number  经过去重"""
        test_data.drop_duplicates(subset=['pr_number'], inplace=True)
        # test_data_y = DataProcessUtils.convertLabelListToDataFrame(test_data_y, pull_number_list, candicateNum)
        test_data_y = DataProcessUtils.convertLabelListToListArray(test_data_y, pull_number_list)

        """获得pr list"""
        prList = list(test_data['pr_number'])

        """先尝试所有信息团在一起"""
        df = df[['pr_number', 'pr_title', 'pr_body', 'label']].copy(deep=True)
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

            """pull request的body"""
            pr_body = getattr(row, 'pr_body')
            pr_body_word_list = [x for x in FleshReadableUtils.word_list(pr_body) if x not in stopwords]
            """对单词做提取词干"""
            pr_body_word_list = nltkFunction.stemList(pr_body_word_list)
            tempList.extend(pr_body_word_list)
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

        """参数规范化"""
        if isSTD:
            stdsc = StandardScaler()
            train_data_std = stdsc.fit_transform(train_data)
            test_data_std = stdsc.transform(test_data)
            return train_data_std, train_data_y, test_data_std, test_data_y, convertDict, prList
        elif isNOR:
            maxminsc = MinMaxScaler()
            train_data_std = maxminsc.fit_transform(train_data)
            test_data_std = maxminsc.transform(test_data)
            return train_data_std, train_data_y, test_data_std, test_data_y, convertDict, prList
        else:
            return train_data, train_data_y, test_data, test_data_y, convertDict, prList

    @staticmethod
    def changeStringToNumber(data, columns, startNum=0):  # 对dataframe的一些特征做文本转数字  input: dataFrame，需要处理的某些列
        if isinstance(data, DataFrame):
            count = startNum
            convertDict = {}  # 用于转换的字典  开始为1
            for column in columns:
                pos = 0
                for item in data[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    data.at[pos, column] = convertDict[item]
                    pos += 1

    @staticmethod
    def getSeriesBarPlot(series):
        #  获得 输入数据的柱状分布图
        import matplotlib.pyplot as plt

        fig = plt.figure()
        # fig.add_subplot(2, 1, 1)
        counts = series.value_counts()
        print(counts)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        counts.plot(kind='bar')
        plt.title('项目rails的评审者历史统计')
        plt.xlabel('成员')
        plt.ylabel('评审次数')
        plt.show()

    @staticmethod
    def testSVM_CAlgorithms(projects, dates, filter_train=False, filter_test=False, error_analysis=True):
        """
           CN s废中涉及空间向量的SVM 由于特征和输入无法和ML兼容，单独开一个文件
        """
        startTime = datetime.now()

        for project in projects:
            excelName = f'outputSVM_C_{project}_{filter_train}_{filter_test}_{error_analysis}.xlsx'
            recommendNum = 5  # 推荐数量
            sheetName = 'result'
            """初始化excel文件"""
            ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])
            """初始化项目抬头"""
            content = ["项目名称：", project]
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

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

            for date in dates:
                recommendList, answerList, prList, convertDict, trainSize = SVM_CTrain.algorithmBody(date, project,
                                                                                                   recommendNum,
                                                                                                   filter_train=filter_train,
                                                                                                   filter_test=filter_test)
                """根据推荐列表做评价"""
                topk, mrr, precisionk, recallk, fmeasurek = \
                    DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

                topks.append(topk)
                mrrs.append(mrr)
                precisionks.append(precisionk)
                recallks.append(recallk)
                fmeasureks.append(fmeasurek)

                error_analysis_data = None
                if error_analysis:
                    y = date[2]
                    m = date[3]
                    filename = projectConfig.getSVM_CDataPath() + os.sep + f'SVM_C_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerData(project, date,
                                                                                             prList,
                                                                                             convertDict, filename,
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
                DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date, error_analysis_data)

                """文件分割"""
                content = ['']
                ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
                content = ['训练集', '测试集']
                ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

                print("cost time:", datetime.now() - startTime)
                """推荐错误可视化"""
                DataProcessUtils.recommendErrorAnalyzer2(error_analysis_datas, project, f'SVM_C_{filter_train}_{filter_test}')

                """计算历史累积数据"""
                DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks, fmeasureks,
                                                   error_analysis_datas)

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, filter_train=False, filter_test=False):
        df = None
        """对需求文件做合并 """
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            print(y, m)
            if i < date[2] * 12 + date[3]:
                if filter_train:
                    filename = projectConfig.getSVM_CDataPath() + os.sep + f'SVM_C_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getSVM_CDataPath() + os.sep + f'SVM_C_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            else:
                if filter_test:
                    filename = projectConfig.getSVM_CDataPath() + os.sep + f'SVM_C_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getSVM_CDataPath() + os.sep + f'SVM_C_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """获取测试的 pull number列表"""
        train_data, train_data_y, test_data, test_data_y, convertDict, prList = SVM_CTrain.preProcess(df, date, project, isNOR=True)
        print("train data:", train_data.shape)
        print("test data:", test_data.shape)

        recommendList, answerList = MultipleLabelAlgorithm.RecommendBySVM(train_data, train_data_y, test_data,
                                                                          test_data_y, recommendNum)
        trainSize = (train_data.shape[0], test_data.shape[0])

        """保存推荐结果到本地"""
        DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict, key=project + str(date))

        return recommendList, answerList, prList, convertDict, trainSize


if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
             (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # dates = [(2017, 1, 2017, 2)]
    projects = ['opencv', 'cakephp', 'akka']
    # projects = ['opencv', 'cakephp', 'akka', 'xbmc', 'babel', 'symfony', 'brew', 'django', 'netty', 'scikit-learn']
    for t in [False]:
         SVM_CTrain.testSVM_CAlgorithms(projects, dates, filter_train=t, filter_test=t, error_analysis=True)

