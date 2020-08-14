# coding=gbk
import math
import os
import time
from datetime import datetime

import pandas

from source.config.projectConfig import projectConfig
from source.data.bean.PullRequest import PullRequest
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class FPS_ACTrain:

    @staticmethod
    def TestAlgorithm(project, dates, filter_train=False, filter_test=False, error_analysis=False,
                      test_type=StringKeyUtils.STR_TEST_TYPE_SLIDE):
        """  2020.8.6
        增加两个参数  filter_train 和  filter_test
         分别用来区别是否使用change trigger过滤的数据集

        2020.8.13 增加参数test_type来区别训练的区别
        test_type_slide 就是之前的滑动窗口
        test_type_increment 是AC算法的做法，训练集细化到pr
        这时候  filter_test发挥作用，但是filter_train失效
        """
        """整合 训练数据"""

        recommendNum = 5  # 推荐数量
        excelName = f'outputFPS_AC_{project}_{filter_train}_{filter_test}_{error_analysis}_{test_type}.xlsx'
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
            recommendList, answerList, prList, convertDict, trainSize = FPS_ACTrain.algorithmBody(date, project,
                                                                                                  recommendNum,
                                                                                                  filter_train=filter_train,
                                                                                                  filter_test=filter_test,
                                                                                                  test_type=test_type)
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
                filter_answer_list = None
                if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
                    y = date[2]
                    m = date[3]
                    filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerData(project, date, prList,
                                                                                             convertDict, filename,
                                                                                             'review_user_login',
                                                                                             'pull_number')
                elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                    fileList = []
                    for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
                        y = int((i - i % 12) / 12)
                        m = i % 12
                        if m == 0:
                            m = 12
                            y = y - 1
                        fileList.append(projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv')

                    filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerDataByIncrement(project, prList,
                                                                                             convertDict, fileList,
                                                                                             'review_user_login',
                                                                                             'pull_number')

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
            DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date,
                                        error_analysis_data)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

        """推荐错误可视化"""
        DataProcessUtils.recommendErrorAnalyzer2(error_analysis_datas, project, f'FPS_AC_{filter_train}_{filter_train}_{test_type}')

        """计算历史累积数据"""
        DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks,
                                           fmeasureks, error_analysis_datas)

    @staticmethod
    def preProcessBySlide(df, dates):
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

        """时间转为时间戳"""
        df['test'] = df['pr_created_at']
        df['pr_created_at'] = df['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pull_number')))

        print("before drop:", df.shape)
        df = df.copy(deep=True)
        df.drop(columns=['review_user_login', 'repo_full_name'], inplace=True)
        df.drop_duplicates(['pull_number', 'commit_sha', 'file_filename'], inplace=True)
        print("after drop:", df.shape)

        """对已经有的特征向量和标签做训练集的拆分"""
        train_data = df.loc[df['label'] == False].copy(deep=True)
        test_data = df.loc[df['label']].copy(deep=True)

        train_data.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)

        """问题转化为多标签问题
            train_data_y   [{pull_number:[r1, r2, ...]}, ... ,{}]
        """

        train_data_y = {}
        for pull_number in train_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            train_data_y[pull_number] = reviewers

        test_data_y = {}
        for pull_number in test_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        return train_data, train_data_y, test_data, test_data_y, convertDict

    @staticmethod
    def preProcessByIncrement(df, dates):
        """参数说明
            df：读取的dataframe对象
            dates:四元组，时间跨度相当于都是测试集, 没有作用
        """

        """注意： 输入文件中已经带有列名了"""

        """处理NAN"""
        df.dropna(how='any', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.fillna(value='', inplace=True)

        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login'])
        """先对tag做拆分"""
        tagDict = dict(list(df.groupby('pull_number')))

        """时间转为时间戳"""
        df['pr_created_at'] = df['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

        print("before drop:", df.shape)
        df = df.copy(deep=True)
        df.drop(columns=['review_user_login', 'repo_full_name'], inplace=True)
        df.drop_duplicates(['pull_number', 'commit_sha', 'file_filename'], inplace=True)
        print("after drop:", df.shape)

        test_data = df
        """问题转化为多标签问题
            train_data_y   [{pull_number:[r1, r2, ...]}, ... ,{}]
        """

        test_data_y = {}
        for pull_number in df.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        return test_data, test_data_y, convertDict

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, filter_train=True, filter_test=True,
                      test_type=StringKeyUtils.STR_TEST_TYPE_SLIDE):

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

            print(y, m)
            filename = None
            if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
                if i < date[2] * 12 + date[3]:
                    if filter_train:
                        filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    else:
                        filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    if filter_test:
                        filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                    else:
                        filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                if filter_test:
                    filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getFPS_ACDataPath() + os.sep + f'FPS_AC_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'

            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)

        if test_type == StringKeyUtils.STR_TEST_TYPE_SLIDE:
            """df做预处理"""
            """新增人名映射字典"""
            train_data, train_data_y, test_data, test_data_y, convertDict = FPS_ACTrain.preProcessBySlide(df, date)

            prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
            """2020.8.1 本来FPS的pr顺序是倒序，现在改为正序，便于和其他算法推荐名单比较"""
            prList.sort()

            recommendList, answerList = FPS_ACTrain.RecommendByFPS_AC_SLIDE(train_data, train_data_y, test_data,
                                                                            test_data_y, recommendNum=recommendNum)

            """新增返回测试 训练集大小，用于做统计"""

            """新增返回训练集 测试集大小"""
            trainSize = (train_data.shape, test_data.shape)
            print(trainSize)

            # """输出推荐名单到文件"""
            # DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict)

            return recommendList, answerList, prList, convertDict, trainSize
        elif test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
            """df做预处理"""
            """新增人名映射字典"""
            test_data, test_data_y, convertDict = FPS_ACTrain.preProcessByIncrement(df, date)

            prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
            """增量预测第一个pr不预测"""

            """2020.8.1 本来FPS的pr顺序是倒序，现在改为正序，便于和其他算法推荐名单比较"""
            prList.sort()
            prList.pop(0)

            recommendList, answerList = FPS_ACTrain.RecommendByFPS_AC_INCREMENT(test_data, test_data_y, recommendNum=recommendNum)

            """新增返回测试 训练集大小，用于做统计"""

            """新增返回训练集 测试集大小"""
            trainSize = (test_data.shape)
            print(trainSize)

            # """输出推荐名单到文件"""
            # DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict)

            return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def RecommendByFPS_AC_SLIDE(train_data, train_data_y, test_data, test_data_y, recommendNum=5, l=1):
        """多标签分类的FPS"""

        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pull_number')))
        trainDict = dict(list(train_data.groupby('pull_number')))
        testTuple = sorted(testDict.items(), key=lambda x: x[0], reverse=False)
        for test_pull_number, test_df in testTuple:
            scores = {}  # 初始化分数字典
            """添加正确答案"""
            answerList.append(test_data_y[test_pull_number])
            for train_pull_number, train_df in trainDict.items():
                paths1 = list(train_df['file_filename'])
                paths2 = list(test_df['file_filename'])
                score = 0

                """计算时间差"""
                gap = (list(test_df['pr_created_at'])[0] -list(train_df['pr_created_at'])[0]) / (3600 * 24)

                for filename1 in paths1:
                    for filename2 in paths2:
                        score += FPSAlgorithm.LCP_2(filename1, filename2) * math.pow(gap, -1)
                score /= paths1.__len__() * paths2.__len__()
                for reviewer in train_data_y[train_pull_number]:
                    if scores.get(reviewer, None) is None:
                        scores[reviewer] = 0
                    scores[reviewer] += score
            recommendList.append([x[0] for x in sorted(scores.items(),
                                                       key=lambda d: d[1], reverse=True)[0:recommendNum]])

        return [recommendList, answerList]

    @staticmethod
    def RecommendByFPS_AC_INCREMENT(test_data, test_data_y, recommendNum=5, l=1):
        """多标签分类的FPS"""

        """获取prList"""
        prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
        prList.sort()

        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pull_number')))
        for pr_index, test_pull_number in enumerate(prList):
            if pr_index == 0:
                """第一个pr没有历史  无法推荐"""
                continue
            test_df = testDict[test_pull_number]
            scores = {}  # 初始化分数字典
            """添加正确答案"""
            answerList.append(test_data_y[test_pull_number])

            train_pr_list = prList[:pr_index]
            for train_pull_number in train_pr_list:
                train_df = testDict[train_pull_number]
                paths1 = list(train_df['file_filename'])
                paths2 = list(test_df['file_filename'])
                score = 0

                """计算时间差"""
                gap = (list(test_df['pr_created_at'])[0] - list(train_df['pr_created_at'])[0]) / (3600 * 24)

                for filename1 in paths1:
                    for filename2 in paths2:
                        score += FPSAlgorithm.LCP_2(filename1, filename2) * math.pow(gap, -l)  # 只计算最长前缀
                score /= paths1.__len__() * paths2.__len__()
                for reviewer in test_data_y[train_pull_number]:
                    if scores.get(reviewer, None) is None:
                        scores[reviewer] = 0
                    scores[reviewer] += score

            """人数不足随机填充"""
            if scores.items().__len__() < recommendNum:
                for i in range(0, recommendNum):
                    scores[f'{StringKeyUtils.STR_USER_NONE}_{i}'] = -1
            recommendList.append([x[0] for x in sorted(scores.items(),
                                                       key=lambda d: d[1], reverse=True)[0:recommendNum]])

        return [recommendList, answerList]


if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
             (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
    #          (2017, 1, 2018, 6)]
    projects = ['opencv', 'cakephp', 'akka', 'xbmc', 'babel', 'symfony', 'brew', 'django', 'netty', 'scikit-learn']
    # projects = ['opencv']
    for p in projects:
        for test_type in [StringKeyUtils.STR_TEST_TYPE_INCREMENT]:
            for t in [False]:
                if test_type == StringKeyUtils.STR_TEST_TYPE_INCREMENT:
                    dates = [(2018, 1, 2018, 12)]
                FPS_ACTrain.TestAlgorithm(p, dates, filter_train=t, filter_test=t, error_analysis=True,
                                          test_type=test_type)
