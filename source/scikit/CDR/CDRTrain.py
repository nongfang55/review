# coding=gbk
import os
from datetime import datetime
import heapq
import time
from math import ceil

import numpy
from pandas import DataFrame
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from source.config.projectConfig import projectConfig
from source.data.service.DataSourceHelper import  appendFilePathFeatureVector
from source.scikit.ML.MultipleLabelAlgorithm import MultipleLabelAlgorithm
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper


class CDRTrain:

    @staticmethod
    def preProcess(df, date, project, isSTD=False, isNOR=False, m=3):
        """参数说明
        df：读取的dataframe对象
        testDate:作为测试的年月 (year,month)
        isSTD:对数据是否标准化
        isNOR:对数据是否归一化
        m: 超参数，窗口时间
        """
        print("start df shape:", df.shape)
        """过滤NA的数据"""
        df.dropna(axis=0, how='any', inplace=True)
        print("after fliter na:", df.shape)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == date[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == date[3]))
        df['label_y'] = df['pr_created_at'].apply(lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year)
        df['label_m'] = df['pr_created_at'].apply(lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon)
        df.reset_index(drop=True, inplace=True)

        """更正说明：由于PCA不能训练集和测试集同时降维，否则相当于使用了后面的信息
            所以添加之前必须两者分别处理 4.13 
            append 函数必须在表明label后面使用"""

        """添加File Path Features"""
        df = appendFilePathFeatureVector(df, project, date, 'pr_number')


        """读取User Follow的信息"""
        user_follow_relation_path = projectConfig.getUserFollowRelation()
        userFollowRelation = pandasHelper.readTSVFile(
            os.path.join(user_follow_relation_path, f'userFollowRelation.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        def isInTimeGap(x, m, maxYear, maxMonth):
            d = x['label_y'] * 12 + x['label_m']
            d2 = maxYear * 12 + maxMonth
            return d >= d2 - m

        """对人名字做数字处理"""
        """频率不过的评审者在编号之前就已经过滤了，不用考虑分类不连续的情况"""
        """这里reviewer_user_login 放在 第一个否则会影响candicateNum这个变量在后面的引用"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login', 'pr_user_login'])

        print(df.shape)
        candicateNum = max(df.loc[df['label'] == 0]['review_user_login'])
        print("candicate Num:", candicateNum)

        """计算contributor set"""
        contribute_list = list(set(df.loc[df['label'] == 1]['pr_user_login']))
        reviewer_list = list(set(df.loc[df['label'] == 0]['review_user_login']))

        """添加Relation ship Features"""
        """对 train set和test set的处理方式稍微不同   train set数据统计依照之前pr
            而训练集的统计数据只限制于trianset
        """

        """把  df 的pr_created_at 和 comment_at 转化为时间戳"""
        df['pr_created_at'] = df['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['comment_at'] = df['comment_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['response_time'] = df['comment_at'] - df['pr_created_at']

        """Prior Evaluation  reviewer cm 之前 review co的次数
           Recent Evaluation reviewer cm 在 m 个月 reivew co的次数
           Follow Relation  co 是否follow cm
           Follower Relation  cm 是否follow co
        """
        startTime = datetime.now()
        prior_evaluation = {}
        recent_evaluation = {}
        follower_relation = {}
        following_relation = {}
        followMap = {}
        for k in convertDict.keys():
            """获取 reviewer 的 follow 列表"""
            follower_list = list(set(userFollowRelation.loc[userFollowRelation['login'] == k]['following_login']))
            for f in follower_list:
                if f in convertDict.keys():
                    followMap[(convertDict[k], convertDict[f])] = 1

        for reviewer in reviewer_list:
            prior_evaluation[reviewer] = []
            recent_evaluation[reviewer] = []
            follower_relation[reviewer] = []
            following_relation[reviewer] = []
        cols = list(df.columns)

        for data in df.itertuples(index=False, name='Pandas'):
            if data.__len__() < 14:
                pullNumber = getattr(data, 'pr_number')
                author = getattr(data, 'pr_user_login')
                label = getattr(data, 'label')
                label_m = getattr(data, 'label_m')
                label_y = getattr(data, 'label_y')
            else:
                pullNumber = data[cols.index("pr_number")]
                author = data[cols.index("pr_user_login")]
                label = data[cols.index("label")]
                label_m = data[cols.index("label_m")]
                label_y = data[cols.index("label_y")]

            temp = None
            if label == 0:
                temp = df.loc[df['pr_number'] < pullNumber]
            else:
                temp = df.loc[df['label'] == 0]
            temp = temp.loc[df['pr_user_login'] == author].copy(deep=True)
            """依次遍历每个候选者统计"""
            prior_evaluation_dict = dict(temp['review_user_login'].value_counts())
            for r in reviewer_list:
                prior_evaluation[r].append(prior_evaluation_dict.get(r, 0))
            """temp 二次过滤  选m个月以内的"""
            if temp.shape[0] > 0:
                if label == 0:
                    temp['target'] = temp.apply(lambda x: isInTimeGap(x, m, label_y, label_m), axis=1)
                else:
                    temp['target'] = temp.apply(lambda x: isInTimeGap(x, m, date[2], date[3]), axis=1)
                temp = temp.loc[temp['target'] == 1]
            """依次遍历每个候选者统计"""
            recent_evaluation_dict = dict(temp['review_user_login'].value_counts())
            for r in reviewer_list:
                recent_evaluation[r].append(recent_evaluation_dict.get(r, 0))
            """添加 follow 和 following 信息"""
            for r in reviewer_list:
                follower_relation[r].append(followMap.get((author, r), 0))
                following_relation[r].append(followMap.get((r, author), 0))

        """添加"""
        for r in reviewer_list:
            df[f'prior_evaluation_{r}'] = prior_evaluation[r]
            df[f'recent_evaluation_{r}'] = recent_evaluation[r]
            df[f'follower_relation_{r}'] = follower_relation[r]
            df[f'following_relation_{r}'] = following_relation[r]

        print("prior cost time:", datetime.now() - startTime)
        startTime = datetime.now()

        # 开始时间：数据集开始时间的前一天
        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)
        # 结束时间：数据集的最后一天
        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        """Activeness Feature 添加"""
        total_pulls = {}  # 项目有的所有pr
        evaluate_pulls = {}  # co 之前review的数量
        recent_pulls = {}  # co 最近m月 review的数量
        evaluate_time = {}  # co 平均回应时间
        last_time = {}  # co 最后一次reivew 的时间间隔
        first_time = {}  # co 第一次review的时间间隔
        for reviewer in reviewer_list:
            total_pulls[reviewer] = []
            evaluate_pulls[reviewer] = []
            recent_pulls[reviewer] = []
            evaluate_time[reviewer] = []
            last_time[reviewer] = []
            first_time[reviewer] = []
        count = 0
        cols = list(df.columns)

        index_pr_number = cols.index("pr_number")
        index_pr_label = cols.index("label")
        index_pr_label_m = cols.index("label_m")
        index_pr_label_y = cols.index("label_y")

        for data in df.itertuples(index=False):
            print("count for active:", count)
            count += 1
            pullNumber = data[index_pr_number]
            label = data[index_pr_label]
            label_m = data[index_pr_label_m]
            label_y = data[index_pr_label_y]
            temp = None
            if label == 0:
                temp = df.loc[df['pr_number'] < pullNumber].copy(deep=True)
            else:
                temp = df.loc[df['label'] == 0].copy(deep=True)
            """依次遍历每个候选者统计"""
            total_pull_number = list(set(temp['pr_number'])).__len__()
            res_reviewer_list = reviewer_list.copy()

            groups = dict(list(temp.groupby('review_user_login')))
            """先遍历有tempDf的reviewer"""
            for r, tempDf in groups.items():
                total_pulls[r].append(total_pull_number)
                res_reviewer_list.remove(r)
                if tempDf.shape[0] == 0:
                    """没有历史 认为age=0， 间隔是最大间隔"""
                    first_time[r].append(0)
                    last_time[r].append(end_time - start_time)
                else:
                    pr_created_time_list = list(tempDf['pr_created_at'])
                    first_review_time = min(pr_created_time_list)
                    last_review_time = max(pr_created_time_list)
                    first_time[r].append(end_time - first_review_time)
                    last_time[r].append(end_time - last_review_time)
                evaluate_pulls[r].append(tempDf.shape[0])

                """平均回应时间统计"""
                if tempDf.shape[0] > 0:
                    evaluate_avg = sum(tempDf['response_time'])
                    evaluate_avg /= tempDf.shape[0]
                else:
                    evaluate_avg = end_time - start_time
                evaluate_time[r].append(evaluate_avg)

            for r in res_reviewer_list:
                total_pulls[r].append(total_pull_number)
                evaluate_pulls[r].append(0)
                first_time[r].append(0)
                last_time[r].append(end_time - start_time)
                evaluate_avg = end_time - start_time
                evaluate_time[r].append(evaluate_avg)
                # recent_pulls[r].append(0)

            """过滤k个月 重新计算"""
            if label == 0:
                if temp.shape[0] > 0:
                    temp['target'] = temp.apply(lambda x: isInTimeGap(x, m, label_y, label_m), axis=1)
                    temp = temp.loc[temp['target'] == 1]
            else:
                if temp.shape[0] > 0:
                    temp['target'] = temp.apply(lambda x: isInTimeGap(x, m, date[2], date[3]), axis=1)
                    temp = temp.loc[temp['target'] == 1]

            res_reviewer_list = reviewer_list.copy()
            groups = dict(list(temp.groupby('review_user_login')))
            """先遍历有tempDf的reviewer"""
            for r, tempDf in groups.items():
                recent_pulls[r].append(tempDf.shape[0])
                res_reviewer_list.remove(r)

            for r in res_reviewer_list:
                recent_pulls[r].append(0)

        """Activeness Feature增加到 dataframe"""
        for r in reviewer_list:
            df[f'total_pulls_{r}'] = total_pulls[r]
            df[f'evaluate_pulls_{r}'] = evaluate_pulls[r]
            df[f'recent_pulls_{r}'] = recent_pulls[r]
            df[f'first_time_{r}'] = first_time[r]
            df[f'last_time_{r}'] = last_time[r]
            df[f'evaluate_time_{r}'] = evaluate_time[r]

        print("active cost time:", datetime.now() - startTime)

        tagDict = dict(list(df.groupby('pr_number')))

        """对已经有的特征向量和标签做训练集的拆分"""
        train_data = df.loc[df['label'] == False].copy(deep=True)
        test_data = df.loc[df['label']].copy(deep=True)

        train_data.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)

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
        test_data.drop_duplicates(subset=['pr_number'], inplace=True)
        # test_data_y = DataProcessUtils.convertLabelListToDataFrame(test_data_y, pull_number_list, candicateNum)
        test_data_y = DataProcessUtils.convertLabelListToListArray(test_data_y, pull_number_list)

        """获得pr list"""
        prList = list(test_data['pr_number'])

        """去除pr number"""
        test_data.drop(columns=['pr_number'], inplace=True)
        train_data.drop(columns=['pr_number'], inplace=True)

        test_data.drop(columns=['pr_created_at', 'pr_user_login',
                                'comment_at', 'label_y', 'label_m', 'response_time'], inplace=True)
        train_data.drop(columns=['pr_created_at',  'pr_user_login',
                                'comment_at', 'label_y', 'label_m', 'response_time'], inplace=True)
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
    def testCDRAlgorithms(projects, dates, filter_train=False, filter_test=False, error_analysis=True):
        """
           CoreDevRec 由于特征和输入无法和ML兼容，单独开一个文件
        """
        startTime = datetime.now()

        for project in projects:
            excelName = f'output_{project}_CDR_{filter_train}_{filter_test}_{error_analysis}.xlsx'
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
                recommendList, answerList, prList, convertDict, trainSize = CDRTrain.algorithmBody(date, project,
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
                    filename = projectConfig.getCDRDataPath() + os.sep + f'CDR_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
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
                DataProcessUtils.recommendErrorAnalyzer2(error_analysis_datas, project, f'CDR')

                """计算历史累积数据"""
                DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks, fmeasureks,
                                                   error_analysis_datas)

    @staticmethod
    def algorithmBody(date, project, algorithmType, recommendNum=5, filter_train=False, filter_test=False):
        df = None
        """对需求文件做合并 """
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            if i < date[2] * 12 + date[3]:
                if filter_train:
                    filename = projectConfig.getCDRDataPath() + os.sep + f'CDR_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getCDRDataPath() + os.sep + f'CDR_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            else:
                if filter_test:
                    filename = projectConfig.getCDRDataPath() + os.sep + f'CDR_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getCDRDataPath() + os.sep + f'CDR_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """获取测试的 pull number列表"""
        train_data, train_data_y, test_data, test_data_y, convertDict, prList = CDRTrain.preProcess(df, date, project, isNOR=True)
        print("train data:", train_data.shape)
        print("test data:", test_data.shape)

        recommendList, answerList = MultipleLabelAlgorithm.RecommendBySVM(train_data, train_data_y, test_data,
                                                                          test_data_y, recommendNum)
        trainSize = (train_data.shape[0], test_data.shape[0])

        """保存推荐结果到本地"""
        DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict, key=project + str(date))

        return recommendList, answerList, prList, convertDict, trainSize


if __name__ == '__main__':
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
    #          (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
    #          (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    dates = [(2017, 1, 2017, 2)]
    projects = ['opencv']
    CDRTrain.testCDRAlgorithms(projects, dates)

