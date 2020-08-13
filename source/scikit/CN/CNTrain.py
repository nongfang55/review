# coding=gbk
import math
import operator
import os
import time
from datetime import datetime
from functools import cmp_to_key

import pandas
import pyecharts
from gensim import corpora, models
from pyecharts.options import series_options

from source.config.projectConfig import projectConfig
from source.data.bean.PullRequest import PullRequest
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.CN.Gragh import Graph
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.Gephi import Gephi
from source.utils.Gexf import Gexf
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper
from collections import deque
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from pyecharts import options as opts
from pyecharts.charts import Graph as EGraph
import scipy
from scipy.stats import pearsonr
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import statsmodels.api as sm


class CNTrain:
    """基于CN，对于同一个contributor的所有pr都是一样的结果"""
    PACCache = {}
    PNCCache = {}
    freq = None  # 是否已经生成过频繁集
    topKCommunityActiveUser = []  # 各community最活跃的成员

    @staticmethod
    def clean():
        CNTrain.PACCache = {}
        CNTrain.PNCCache = {}
        CNTrain.freq = None  # 是否已经生成过频繁集
        # TODO 暂时不清除
        # CNTrain.topKCommunityActiveUser = []  # 各community最活跃的成员

    @staticmethod
    def testCNAlgorithm(project, dates, filter_train=False, filter_test=False, is_split=False, error_analysis=False):
        """整合 训练数据"""
        """2020.8.7 新增参数 filter_data 和 error_analysis
           filter_train 判断是否使用 changetrigger过滤的训练数据
           filter_test 判断是否使用 changetrigger过滤的验证数据
           error_analysis 表示是否开启chang_trigger过滤答案的错误统计机制
        """
        recommendNum = 5  # 推荐数量
        excelName = f'outputCN_{project}_{filter_train}_{filter_test}_{error_analysis}.xlsx'
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
            CNTrain.clean()
            startTime = datetime.now()
            prList, convertDict, trainSize, communities_data= CNTrain.algorithmBody(date, project,
                                                                                              recommendNum,
                                                                                              filter_train=filter_train,
                                                                                              filter_test=filter_test,
                                                                                              is_split=is_split)

            communitiesTuple = sorted(communities_data.items(), key=lambda x: x[0])
            for cid, c_data in communitiesTuple:
                """根据推荐列表做评价"""
                topk, mrr, precisionk, recallk, fmeasurek = \
                    DataProcessUtils.judgeRecommend(c_data['recommend_list'], c_data['answer_list'], recommendNum)
                communities_data[cid]['topk'] = topk
                communities_data[cid]['mrr'] = mrr
                communities_data[cid]['precisionk'] = precisionk
                communities_data[cid]['recallk'] = recallk
                communities_data[cid]['fmeasurek'] = fmeasurek

            print("project: {0}, modularity: {1}, entropy: {2}, avg_variance: {3}".format(project,
                                                                       communities_data['whole']['modularity'],
                                                                       communities_data['whole']['entropy'],
                                                                       communities_data['whole']['avg_variance']))

            error_analysis_data = None
            if error_analysis:
                y = date[2]
                m = date[3]
                filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerData(project, date, prList,
                                                                                         convertDict, filename,
                                                                                         'reviewer', 'pull_number')
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
                    communities_data['whole']['recommend_list'], communities_data['whole']['answer_list'], filter_answer_list, recommendNum)
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

            topks.append(communities_data['whole']['topk'])
            mrrs.append(communities_data['whole']['mrr'])
            precisionks.append(communities_data['whole']['precisionk'])
            recallks.append(communities_data['whole']['recallk'])
            fmeasureks.append(communities_data['whole']['fmeasurek'])

            """结果写入excel"""
            DataProcessUtils.saveResult_Community_Version(excelName, sheetName, communities_data, date)

            error_analysis_data = None
            if error_analysis:
                y = date[2]
                m = date[3]
                filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                filter_answer_list = DataProcessUtils.getAnswerListFromChangeTriggerData(project, date, prList,
                                                                                         convertDict, filename,
                                                                                         'reviewer', 'pull_number')
                # recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio, recommend_negative_success_pr_ratio, \
                # recommend_negative_success_time_ratio, recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio, \
                # recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio = DataProcessUtils.errorAnalysis(
                #     recommendList, answerList, filter_answer_list, recommendNum)
                # error_analysis_data = [recommend_positive_success_pr_ratio, recommend_positive_success_time_ratio,
                #                        recommend_negative_success_pr_ratio, recommend_negative_success_time_ratio,
                #                        recommend_positive_fail_pr_ratio, recommend_positive_fail_time_ratio,
                #                        recommend_negative_fail_pr_ratio, recommend_negative_fail_time_ratio]

                recommend_positive_success_pr_ratio, recommend_negative_success_pr_ratio, recommend_positive_fail_pr_ratio,\
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
                                        error_analysis_data))

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

        """推荐错误可视化"""
        DataProcessUtils.recommendErrorAnalyzer2(error_analysis_datas, project, 'CN')

        """计算历史累积数据"""
        DataProcessUtils.saveFinallyResult(excelName, sheetName, topks, mrrs, precisionks, recallks,
                                           fmeasureks, error_analysis_datas)

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, filter_train=False, filter_test=False, is_split=False):

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
            if i < date[2] * 12 + date[3]:
                if filter_train:
                    filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            else:
                if filter_test:
                    filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getCNDataPath() + os.sep + f'CN_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = CNTrain.preProcess(df, date)

        if not is_split:
            prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
            prList.sort()

            prList, communities_data = CNTrain.RecommendByCN(project, date, train_data, train_data_y, test_data,
                                                              test_data_y, convertDict, recommendNum=recommendNum)
        else:
            prList, communities_data = CNTrain.RecommendByCNSplit(project, date, train_data,
                                                                                    train_data_y, test_data,
                                                                                    test_data_y, convertDict,
                                                                                    recommendNum=recommendNum)
        """保存推荐结果到本地"""
        DataProcessUtils.saveRecommendList(prList, communities_data['whole']['recommend_list'], communities_data['whole']['answer_list'], convertDict, communities_data['whole']['author_list'], key=project + str(date) + str(filter_train) + str(filter_test))

        """新增返回测试 训练集大小，用于做统计"""
        # from source.scikit.combine.CBTrain import CBTrain
        # recommendList, answerList = CBTrain.recoverName(recommendList, answerList, convertDict)
        """新增返回训练集 测试集大小"""
        trainSize = (train_data.shape, test_data.shape)
        print(trainSize)

        return prList, convertDict, trainSize, communities_data

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
        df['pr_title'].fillna(value='', inplace=True)
        df['pr_body'].fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['pr_author', 'reviewer'])

        train_data = df.loc[df['label'] == False]
        train_data.reset_index(drop=True, inplace=True)
        test_data = df.loc[df['label'] == True]
        test_data.reset_index(drop=True, inplace=True)

        """8ii处理NAN"""
        train_data.dropna(how='any', inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        train_data.fillna(value='', inplace=True)

        """先对tag做拆分"""
        trainDict = dict(list(train_data.groupby('pull_number')))
        testDict = dict(list(test_data.groupby('pull_number')))

        """过滤掉评论时间在数据集时间范围内之后的数据"""
        end_time = str(dates[2]) + "-" + str(dates[3]) + "-" + "01 00:00:00"
        train_data = train_data[train_data['commented_at'] < end_time]
        train_data.reset_index(drop=True, inplace=True)

        test_data_y = {}
        for pull_number in test_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(testDict[pull_number].drop_duplicates(['reviewer'])['reviewer'])
            test_data_y[pull_number] = reviewers

        train_data_y = {}
        for pull_number in train_data.drop_duplicates(['pull_number'])['pull_number']:
            reviewers = list(trainDict[pull_number].drop_duplicates(['reviewer'])['reviewer'])
            train_data_y[pull_number] = reviewers

        return train_data, train_data_y, test_data, test_data_y, convertDict

    @staticmethod
    def RecommendByCN(project, date, train_data, train_data_y, test_data, test_data_y, convertDict, recommendNum=5):
        # TODO 原算法也要统计一下分社区的结果
        """评论网络推荐算法"""
        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pull_number')))
        authorList = []  # 用于统计最后的推荐结果
        typeList = []
        testTuple = sorted(testDict.items(), key=lambda x: x[0])
        hasProcesedPrs = []
        communities_data = {}

        # echarts画图
        # CNTrain.drawCommentGraph(project, date, graph, convertDict)

        # gephi社区发现
        graph, communities, modularity, entropy, avg_variance = CNTrain.getCommunities(project, date, train_data, convertDict, "whole")
        print("the whole graph modularity: {0}, community count: {1}".format(modularity, communities.__len__()))
        CNTrain.searchTopKByGephi(graph, communities, recommendNum)

        communities_data['whole'] = {
            'modularity': modularity,
            'size': graph.node_list.__len__(),
            'community_count': communities.__len__(),
            'entropy': entropy,
            'avg_variance': avg_variance
        }

        for test_pull_number, test_df in testTuple:
            test_df.reset_index(drop=True, inplace=True)
            answerList.append(test_data_y[test_pull_number])
            pr_author = test_df.at[0, 'pr_author']
            node = graph.get_node(pr_author)
            authorList.append(pr_author)

            cid = CNTrain.getUserCid(pr_author, communities)
            if communities_data.get(cid, None) is None:
                communities_data[cid] = {
                    'modularity': '',
                    'size': '',
                    'community_count': '',
                    'recommend_list': [],
                    'answer_list': [],
                    'author_list': [],
                    'type_list': [],
                    'entropy': ''
                }
            communities_data[cid]['author_list'].append(pr_author)
            communities_data[cid]['answer_list'].append(test_data_y[test_pull_number])

            if node is not None and node.connectedTo:
                """PAC推荐"""
                res = CNTrain.recommendByPAC(graph, pr_author, recommendNum)
                recommendList.append(res)
                communities_data[cid]['recommend_list'].append(res)
                typeList.append('PAC')
                communities_data[cid]['type_list'].append('PAC')
            elif node is not None:
                """PNC推荐"""
                res = CNTrain.recommendByPNC(train_data, graph, pr_author, recommendNum)
                recommendList.append(res)
                communities_data[cid]['recommend_list'].append(res)
                typeList.append('PNC')
                communities_data[cid]['type_list'].append('PNC')
            else:
                """Gephi推荐"""
                recommendList.append(CNTrain.topKCommunityActiveUser)
                communities_data[cid]['recommend_list'].append(CNTrain.topKCommunityActiveUser)
                typeList.append('Gephi')
                communities_data[cid]['type_list'].append('Gephi')

            hasProcesedPrs.append(test_pull_number)

        communities_data['whole']['recommend_list'] = recommendList
        communities_data['whole']['answer_list'] = answerList
        communities_data['whole']['author_list'] = authorList
        communities_data['whole']['type_list'] = typeList
        return hasProcesedPrs, communities_data

    @staticmethod
    def getUserCid(user, communities):
        for cid, community in communities.items():
            if user in community:
                return cid
        return 'other'

    @staticmethod
    def RecommendByCNSplit(project, date, train_data, train_data_y, test_data, test_data_y, convertDict, recommendNum=5):
        """评论网络推荐算法，社区分割版本"""
        recommendList = []
        answerList = []
        authorList = []  # 用于统计最后的推荐结果
        typeList = []
        hasProcesedPrs = []  # 记录已经处理过的测试pr, 因为社区分割后的推荐结果是乱序的，所以需要记录真实的pr顺序
        communities_data = {}    # 记录每个社区最后的结果，用于写入结果
        testDict = dict(list(test_data.groupby('pull_number')))
        testTuple = sorted(testDict.items(), key=lambda x: x[0])

        """
        communities_data = {
            # cid: whole, 0, 1, 2, 3
            "cid":{
                modularity: 0.01,  模块度
                size: 134,  社区大小，即包含的用户数
                community_count: 7, 子社区数量
                recommend_list: [], 
                answer_list: [], 
                author_list: [],
                type_list: []
            }
        }
        """

        # echarts画图
        # CNTrain.drawCommentGraph(project, date, graph, convertDict)

        """用全量train_data获取社区分布"""
        graph, communities, modularity = CNTrain.getCommunities(project, date, train_data, convertDict, key="whole")
        print("the whole graph modularity: {0}, community count: {1}".format(modularity, communities.__len__()))

        """初始化全量社区数据"""
        communities_data['whole'] = {
            'modularity': modularity,
            'size': graph.node_list.__len__(),
            'community_count': communities.__len__()
        }

        # 获取整个社区内活跃的topk用户
        CNTrain.searchTopKByGephi(graph, communities, recommendNum)

        """遍历社区，分社区推荐"""
        for cid, community in communities.items():
            """清除社区推荐数据缓存"""
            CNTrain.clean()

            """生成子社区数据集"""
            sub_recommendList = []
            sub_answerList = []
            sub_authorList = []
            sub_typeList = []

            sub_train_data = train_data[
                (train_data['reviewer'].isin(community)) & (train_data['pr_author'].isin(community))]
            sub_train_data_y = {}
            sub_trainDict = dict(list(sub_train_data.groupby('pull_number')))
            for pull_number in sub_train_data.drop_duplicates(['pull_number'])['pull_number']:
                reviewers = list(sub_trainDict[pull_number].drop_duplicates(['reviewer'])['reviewer'])
                sub_train_data_y[pull_number] = reviewers
            sub_test_data = test_data[test_data['pr_author'].isin(community)]
            """如果子社区测试数据集为空，跳过该社区"""
            if sub_test_data.empty:
                continue

            """生成子社区网络图"""
            sub_graph, sub_communities, sub_modularity = CNTrain.getCommunities(project, date, sub_train_data, convertDict, key=cid)
            print("the {0} graph modularity: {1}, community count: {2}".format(cid, sub_modularity, sub_communities.__len__()))

            """初始化子社区数据"""
            communities_data[cid] = {
                'modularity': sub_modularity,
                'size': sub_graph.node_list.__len__(),
                'community_count': sub_communities.__len__()
            }

            """开始推荐"""
            sub_testDict = dict(list(sub_test_data.groupby('pull_number')))
            sub_testTuple = sorted(sub_testDict.items(), key=lambda x: x[0])
            for test_pull_number, test_df in sub_testTuple:
                test_df.reset_index(drop=True, inplace=True)

                answerList.append(test_data_y[test_pull_number])
                # 在子社区结果里也保存一份
                sub_answerList.append(test_data_y[test_pull_number])

                pr_author = test_df.at[0, 'pr_author']
                node = sub_graph.get_node(pr_author)
                authorList.append(pr_author)
                sub_authorList.append(pr_author)

                hasProcesedPrs.append(test_pull_number)

                if node is not None and node.connectedTo:
                    """PAC推荐"""
                    res = CNTrain.recommendByPAC(sub_graph, pr_author, recommendNum)
                    recommendList.append(res)
                    # 在子社区结果里也保存一份
                    sub_recommendList.append(res)
                    typeList.append('PAC')
                    sub_typeList.append('PAC')
                elif node is not None:
                    """PNC推荐"""
                    res = CNTrain.recommendByPNC(sub_train_data, sub_graph, pr_author, recommendNum)
                    recommendList.append(res)
                    # 在子社区结果里也保存一份
                    sub_recommendList.append(res)
                    typeList.append('PNC')
                    sub_typeList.append('PNC')
                else:
                    """Gephi推荐"""
                    res =CNTrain.topKCommunityActiveUser
                    recommendList.append(res)
                    # 在子社区结果里也保存一份
                    sub_recommendList.append(res)
                    typeList.append('Gephi')
                    sub_typeList.append('Gephi')
            communities_data[cid]['recommend_list'] = sub_recommendList
            communities_data[cid]['answer_list'] = sub_answerList
            communities_data[cid]['author_list'] = sub_authorList
            communities_data[cid]['type_list'] = sub_typeList

        """处理在分社区推荐中未处理的pr（这些用户在之前的图中未出现过，归作other社区）"""
        """清除社区推荐数据缓存"""
        CNTrain.clean()

        """生成子社区数据集"""
        sub_recommendList = []
        sub_answerList = []
        sub_authorList = []
        sub_typeList = []

        """初始化子社区数据"""
        communities_data['other'] = {
            'modularity': None,
            'community_count': 0
        }

        for test_pull_number, test_df in testTuple:
            if test_pull_number in hasProcesedPrs:
                continue
            test_df.reset_index(drop=True, inplace=True)

            answerList.append(test_data_y[test_pull_number])
            sub_answerList.append(test_data_y[test_pull_number])

            pr_author = test_df.at[0, 'pr_author']
            authorList.append(pr_author)
            sub_authorList.append(pr_author)

            hasProcesedPrs.append(test_pull_number)

            """对于孤立用户，直接用Gephi推荐"""
            recommendList.append(CNTrain.topKCommunityActiveUser)
            sub_recommendList.append(CNTrain.topKCommunityActiveUser)
            typeList.append('Gephi')
            sub_typeList.append('Gephi')

        communities_data['other']['recommend_list'] = sub_recommendList
        communities_data['other']['answer_list'] = sub_answerList
        communities_data['other']['author_list'] = sub_authorList
        communities_data['other']['type_list'] = sub_typeList
        communities_data['other']['size'] = list(set(sub_authorList)).__len__()

        communities_data['whole']['recommend_list'] = recommendList
        communities_data['whole']['answer_list'] = answerList
        communities_data['whole']['author_list'] = authorList
        communities_data['whole']['type_list'] = typeList
        return hasProcesedPrs, communities_data

    @staticmethod
    def caculateWeight(comment_records, start_time, end_time):
        weight_lambda = 0.8
        weight = 0
        pr_cnt = 0

        grouped_comment_records = comment_records.groupby(comment_records['pull_number'])
        for pr, comments in grouped_comment_records:
            comments.reset_index(inplace=True, drop=True)
            """遍历每条评论，计算权重"""
            for cm_idx, cm_row in comments.iterrows():
                cm_timestamp = time.strptime(cm_row['commented_at'], "%Y-%m-%d %H:%M:%S")
                cm_timestamp = int(time.mktime(cm_timestamp))
                """计算t值: the element t(ij,r,n) is a time-sensitive factor """
                t = (cm_timestamp - start_time) / (end_time - start_time)
                cm_weight = math.pow(weight_lambda, cm_idx) * t
                weight += cm_weight
            pr_cnt += 1
        return weight, pr_cnt

    @staticmethod
    def recommendByPAC(graph, contributor, recommendNum):
        """For a PAC, it is natural to recommend the user who has previously interacted with the contributor directly"""
        if CNTrain.PACCache.__contains__(contributor):
            return CNTrain.PACCache[contributor]

        """用BFS算法找到topK"""
        start = graph.get_node(contributor)
        queue = deque([start])
        recommendList = []
        topk = recommendNum
        while queue:
            node = queue.popleft()
            node.rank_edges()
            node.marked = []
            while node.marked.__len__() < len(node.connectedTo):
                if topk == 0:
                    CNTrain.PACCache[contributor] = recommendList
                    return recommendList
                tmp = node.best_neighbor()
                node.mark_edge(tmp)
                """跳过推荐者已被包含过，推荐者reviewe次数<2 或是本人的情况"""
                if recommendList.__contains__(tmp.id) or tmp.in_cnt < 2 or tmp.id == contributor:
                    continue
                queue.append(tmp)
                recommendList.append(tmp.id)
                topk -= 1

        """拿topk做补充"""
        if recommendList.__len__() < recommendNum:
            recommendList.extend(CNTrain.topKCommunityActiveUser)

        """缓存结果"""
        CNTrain.PACCache[contributor] = recommendList[0:recommendNum]
        return recommendList

    @staticmethod
    def recommendByPNC(train_data, graph, contributor, recommendNum):
        """For a PNC, since there is no prior knowledge of which developers used to review the submitter’s pull-request"""

        """生成Apriori数据集"""
        if CNTrain.freq is None:
            grouped_train_data = train_data.groupby(train_data['pull_number'])
            apriori_dataset = []
            for pull_number, group in grouped_train_data:
                reviewers = group['reviewer'].to_list()
                """过滤in_cnt<2的review"""
                # reviewers = [x for x in reviewers if graph.get_node(x).in_cnt < 2]
                apriori_dataset.append(list(set(reviewers)))
            te = TransactionEncoder()
            # 进行 one-hot 编码
            te_ary = te.fit(apriori_dataset).transform(apriori_dataset)
            df = pandas.DataFrame(te_ary, columns=te.columns_)

            # 利用 Apriori算法 找出频繁项集
            print("start gen apriori......")
            # TODO top-k频繁项集计算
            freq = apriori(df, min_support=0.01, use_colnames=True)
            CNTrain.freq = freq.sort_values(by="support", ascending=False)
            print("finish gen apriori!!!")

        """直接从缓存取结果"""
        if CNTrain.PNCCache.__contains__(contributor):
            return CNTrain.PNCCache[contributor]

        """找到和自己review兴趣相近的用户作为reviewer"""
        recommendList = []
        for idx, row in CNTrain.freq.iterrows():
            community = list(row['itemsets'])
            community = sorted(community,
                   key=lambda x:graph.get_node(x).in_cnt, reverse=True)
            if contributor in community and community.__len__() > 1:
                community.remove(contributor)
                recommendList.extend(community)

        # TODO 因为频繁项集数目很少，大部分用户都找不到和自己review兴趣相近的用户，所以这里用topKActive的用户补充
        if recommendList.__len__() < recommendNum:
            """此时可能还没计算topKActiveContributor"""
            recommendList.extend(CNTrain.topKCommunityActiveUser)
        recommendList = recommendList[0:recommendNum]

        """缓存结果"""
        CNTrain.PNCCache[contributor] = recommendList
        return recommendList

    @staticmethod
    def searchTopKByGephi(graph, communities, recommendNum=5):
        """利用Gephi发现社区，推荐各社区活跃度最高的人"""
        """筛选出成员>2的社区"""
        communities = {k: v for k, v in communities.items() if v.__len__() >= 2}
        # 按照community size排序
        communities = sorted(communities.items(), key=lambda d: d[1].__len__(), reverse=True)

        # 循环遍历communiity，从各个community找出最活跃(入度)的topK
        topKActiveContributor = []
        for i in range(0, recommendNum):
            for community in communities:
                """community内部按入度排序"""
                community_uids = sorted(community[1],
                       key=lambda x:graph.get_node(int(x)).in_cnt, reverse=True)
                for user in community_uids:
                    user = int(user)
                    if user in topKActiveContributor:
                        continue
                    topKActiveContributor.append(user)
                    break
                if topKActiveContributor.__len__() == recommendNum:
                    break
            if topKActiveContributor.__len__() == recommendNum:
                break

        CNTrain.topKCommunityActiveUser = topKActiveContributor[0:recommendNum]

    @staticmethod
    def drawCommentGraph(project, date, graph, convertDict):
        nodes = []
        links = []
        tempDict = {k: v for v, k in convertDict.items()}

        """遍历图，找出in_cnt和weight的最大和最小值，数据归一化"""
        in_min, in_max, w_min, w_max = [0, 0, 0, 0]
        for key, node in graph.node_list.items():
            in_max = max(in_max, node.in_cnt)
            for weight in node.connectedTo.values():
                w_max = max(w_max, weight)

        in_during = in_max - in_min
        w_during = w_max - w_min
        for key, node in graph.node_list.items():
            nodes.append({
                "name": tempDict[node.id],
                "symbolSize": 10 * (node.in_cnt - in_min) / in_during,
                "value": node.in_cnt
            })
            for to, weight in node.connectedTo.items():
                links.append({
                    "source": tempDict[node.id],
                    "target": tempDict[to.id],
                    "lineStyle": {
                        "width": 10 * (weight - w_min) / w_during
                    }
                })

        file_name = f'graph/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_cn-graph.html'
        EGraph().add("user",
                     nodes=nodes,
                     links=links,
                     repulsion=8000,
                     layout="circular",
                     is_rotate_label=True,
                     linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
                     ) \
                .set_global_opts(
                title_opts=opts.TitleOpts(title="cn-graph"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
                ) \
                .render(file_name)


    @staticmethod
    def genGephiData(project, date, graph, convertDict, key):
        # key标识社区是whole还是单个社区
        file_name = f'{os.curdir}/gephi/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_{key}_network.gexf'

        gexf = Gexf("reviewer_recommend", file_name)
        gexf_graph = gexf.addGraph("directed", "static", file_name)

        gexf_graph.addNodeAttribute("r_cnt", defaultValue='0', type="integer")
        gexf_graph.addNodeAttribute("a_cnt", defaultValue='0', type="integer")

        tempDict = {k: v for v, k in convertDict.items()}

        """遍历图，weight的最大值，数据归一化"""
        w_min, w_max = (0, 0)
        for key, node in graph.node_list.items():
            for weight in node.connectedTo.values():
                w_max = max(w_max, weight)

        w_during = w_max - w_min
        # 边编号
        e_idx = 0
        for key, node in graph.node_list.items():
            gexf_graph.addNode(id=str(node.id), label=tempDict[node.id],
                               attrs=[{'id': 0, 'value': node.reviewer_cnt},
                                      {'id': 1, 'value': node.author_cnt}]
                               )
            for to, weight in node.connectedTo.items():
                gexf_graph.addNode(id=str(to.id), label=tempDict[to.id],
                                   attrs=[{'id': 0, 'value': to.reviewer_cnt},
                                          {'id': 1, 'value': to.author_cnt}]
                                   )
                gexf_graph.addEdge(id=e_idx, source=str(node.id), target=str(to.id), weight=10 * (weight - w_min) / w_during)
                e_idx += 1

        output_file = open(file_name, "wb")
        gexf.write(output_file)
        output_file.close()
        return file_name

    @staticmethod
    def getCommunities(project, date, data, convertDict, g_key):
        """给定一个数据集, 得出社区分布"""
        # key标识社区是whole还是单个社区

        start = datetime.now()
        # 开始时间：数据集开始时间的前一天
        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)
        # 结束时间：数据集的最后一天
        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)
        """构造评论网络"""
        graph = Graph()
        grouped_train_data = data.groupby([data['pr_author'], data['reviewer']])
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            cn_weight, pr_cnt = CNTrain.caculateWeight(group, start_time, end_time)
            graph.add_edge(relation[0], relation[1], cn_weight, pr_cnt)
        print("finish building comments networks! ! ! cost time: {0}s".format(datetime.now() - start))

        """计算网络标准结构熵"""
        degree_dict = {}

        # 结点度之和
        total_degree = 0
        for key, node in graph.node_list.items():
            degree = node.in_cnt + node.connectedTo.items().__len__()
            degree_dict[key] = degree
            total_degree += degree
        # 计算结点重要度
        degree_i_dict = {}
        for key, node in graph.node_list.items():
            degree = node.in_cnt + node.connectedTo.items().__len__()
            degree_i_dict[key] = degree/total_degree
        # 计算最大、最小标准结构熵
        N = graph.node_list.items().__len__()
        e_max = math.log(N)
        e_min = math.log(4 * (N - 1))/2
        e = 0
        for key, node in graph.node_list.items():
            e += degree_i_dict[key] * math.log(degree_i_dict[key])
        e = e * (-1)
        entropy = (e - e_min)/(e_max - e_min)

        """生成gephi数据"""
        file_name = CNTrain.genGephiData(project, date, graph, convertDict, g_key)
        """利用gephi划分社区"""
        from source.utils.Gephi import Gephi
        communities, modularity = Gephi().getCommunity(graph_file=file_name)
        """筛选出成员>2的社区"""
        communities = {k: v for k, v in communities.items() if v.__len__() > 2}



        # 计算保存各社区的度列表
        variance_dict = {
            'whole': []
        }

        """格式化为int， 计算子社区度标准差"""
        variance = 0
        for cid, community in communities.items():
            variance_dict[cid] = []
            format_community = []
            for u in community:
                format_community.append(int(u))
                variance_dict['whole'].append(degree_dict[int(u)])
                variance_dict[cid].append(degree_dict[int(u)])
            communities[cid] = format_community
            variance += np.std(variance_dict[cid], ddof=1)

        avg_variance = variance/communities.items().__len__()

        return graph, communities, modularity, entropy, avg_variance

if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
             (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # projects = ['angular', 'babel', 'react']
    projects = ['next.js']
    for p in projects:
        CNTrain.testCNAlgorithm(p, dates, filter_train=True, filter_test=True,  error_analysis=True)