# coding=gbk
import math
import os
import time
from datetime import datetime

from source.config.projectConfig import projectConfig
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.HG.Edge import Edge
from source.scikit.HG.HyperGraph import HyperGraph
from source.scikit.HG.Node import Node
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.Gexf import Gexf
from source.utils.pandas.pandasHelper import pandasHelper
import numpy as np


class HGTrain:

    """超图建立网络做评审者推荐"""

    @staticmethod
    def TestAlgorithm(project, dates, alpha=0.8, K=20, c=1):
        """整合 训练数据"""
        recommendNum = 5  # 推荐数量
        excelName = f'outputHG_{project}_{alpha}_{K}_{c}.xlsx'
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
            recommendList, answerList, prList, convertDict, trainSize = HGTrain.algorithmBody(date, project,
                                                                                              recommendNum,
                                                                                              alpha=alpha, K=K,
                                                                                              c=c)
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

        """空comment的review包含nan信息，但作为结果集是有用的，所以只对训练集去掉na"""
        # """处理NAN"""
        # df.dropna(how='any', inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # df.fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['author_user_login', 'review_user_login'])

        """对已经有的特征向量和标签做训练集的拆分"""
        train_data = df.loc[df['label'] == False].copy(deep=True)
        test_data = df.loc[df['label']].copy(deep=True)

        train_data.drop(columns=['label'], inplace=True)
        test_data.drop(columns=['label'], inplace=True)

        """处理NAN"""
        train_data.dropna(how='any', inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        train_data.fillna(value='', inplace=True)

        """先对tag做拆分"""
        trainDict = dict(list(train_data.groupby('pr_number')))
        testDict = dict(list(test_data.groupby('pr_number')))

        """注意： train_data 和 test_data 中有多个comment和filename的组合"""
        test_data_y = {}
        for pull_number in test_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = list(testDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        train_data_y = {}
        for pull_number in train_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = list(trainDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            train_data_y[pull_number] = reviewers

        return train_data, train_data_y, test_data, test_data_y, convertDict

    @staticmethod
    def algorithmBody(date, project, recommendNum=5, alpha=0.98, K=20, c=1):

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
            filename = projectConfig.getHGDataPath() + os.sep + f'HG_ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = HGTrain.preProcess(df, date)

        prList = list(set(test_data['pr_number']))
        prList.sort()

        recommendList, answerList, authorList = HGTrain.RecommendByHG(train_data, train_data_y, test_data,
                                                          test_data_y, date, project, convertDict, recommendNum=recommendNum,
                                                          alpha=alpha, K=K, c=c)

        """保存推荐结果，用于做统计"""
        DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict, key=project + str(date),
                                           authorList=authorList)

        """新增返回训练集 测试集大小"""
        trainSize = (train_data.shape[0], test_data.shape[0])
        print(trainSize)

        return recommendList, answerList, prList, convertDict, trainSize

    @staticmethod
    def createTrainDataGraph(train_data, train_data_y, trainPrDis, prToRevMat, authToRevMat, reviewerFreqDict, c):
        """通过训练集计算出超图 测试对象的顶点的边需要额外加入"""

        graph = HyperGraph()

        """先添加PR的顶点"""
        prList = list(set(train_data['pr_number']))
        prList.sort()  # 从小到大排序
        prList = tuple(prList)
        for pr in prList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_PR, contentKey=pr, description=f"pr:{pr}")

        """增加author的顶点"""
        authorList = list(set(train_data['author_user_login']))
        for author in authorList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_AUTHOR, contentKey=author, description=f"author:{author}")

        """增加reviewer的顶点"""
        reviewerList = list(set(train_data['review_user_login']))
        for reviewer in reviewerList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_REVIEWER, contentKey=reviewer, description=f"reviewer:{reviewer}")

        """增加pr之间的边"""
        for p1 in prList:
            node_1 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p1)
            for p2 in prList:
                weight = trainPrDis.get((p1, p2), None)
                if weight is not None:
                    node_2 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p2)
                    graph.add_edge(nodes=[node_1.id, node_2.id], edgeType=Edge.STR_EDGE_TYPE_PR_DIS,
                                   weight=weight * c, description=f"pr distance between {p1} and {p2}",
                                   queryBeforeAdd=True)

        """增加pr和reviewer的边  这里暂时reviewer不合并在一起 weight需要考虑"""
        for pr in prList:
            reviewers = train_data_y[pr]
            for reviewer in reviewers:
                pr_node = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, pr)
                reviewer_node = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, reviewer)
                # 权重参考CF的版本
                graph.add_edge(nodes=[pr_node.id, reviewer_node.id], edgeType=Edge.STR_EDGE_TYPE_PR_REVIEW_RELATION,
                               weight=prToRevMat[pr][reviewer], description=f" pr review relation between pr {pr} and reviewer {reviewer}",                                   nodeObjects=[pr_node, reviewer_node])

        # """增加pr 和 author的边"""
        for pr in prList:
            author = list(set(train_data.loc[train_data['pr_number'] == pr]['author_user_login']))[0]
            pr_node = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, pr)
            author_node = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, author)
            graph.add_edge(nodes=[pr_node.id, author_node.id], edgeType=Edge.STR_EDGE_TYPE_PR_AUTHOR_RELATION,
                           weight=1, description=f" pr author relation between pr {pr} and author {author}",
                           nodeObjects=[pr_node, author_node])

        # """增加 author 和 reviewer 的边"""
        # 权重均为1的版本
        # userList = [x for x in authorList if x in reviewerList]
        # for user in userList:
        #     author_node = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, user)
        #     reviewer_node = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, user)
        #     graph.add_edge(nodes=[author_node.id, reviewer_node.id], edgeType=Edge.STR_EDGE_TYPE_AUTHOR_REVIEWER_RELATION,
        #                    weight=1, description=f"author reviewer relation for {user}",
        #                    nodeObjects=[author_node, reviewer_node])

        # # 权重参考CN的版本
        # for author, relations in authToRevMat.items():
        #     for reviewer, weight in relations.items():
        #         author_node = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, author)
        #         reviewer_node = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, reviewer)
        #         graph.add_edge(nodes=[author_node.id, reviewer_node.id], edgeType=Edge.STR_EDGE_TYPE_AUTHOR_REVIEWER_RELATION,
        #                        weight=weight, description=f"author reviewer relation for {author}",
        #                        nodeObjects=[author_node, reviewer_node])

        # """更新图的几个矩阵"""
        # graph.updateMatrix()
        return graph

    @staticmethod
    def getTrainDataPrDistance(train_data, K, pathDict, date, prCreatedTimeMap):
        """计算在trainData中各个 pr 之间的距离 通过路径相似度比较
           {(num1, num2) -> s1}  其中num1 < num2
           每个顶点取最相似的 K 个作为连接对象，节约空间
           注意  可能有些顶点会有超过K条边
        """
        trainPrDis = {}  # 用于记录pr的距离
        # 开始时间：数据集开始时间的前一天
        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)
        # 结束时间：数据集的最后一天
        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        print(train_data.shape)
        data = train_data[['pr_number', 'filename']].copy(deep=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(inplace=True, drop=True)
        prList = list(set(data['pr_number']))
        prList.sort()  # 从小到大排序
        scoreMap = {}  # 统计所有pr之间相似度的分数
        for index, p1 in enumerate(prList):
            scores = {}  # 记录
            print("now pr:", index, " all:", prList.__len__())
            for p2 in prList:
                if p1 < p2:
                    paths1 = list(pathDict[p1]['filename'])
                    paths2 = list(pathDict[p2]['filename'])
                    score = 0
                    for filename1 in paths1:
                        for filename2 in paths2:
                            # score += FPSAlgorithm.LCP_2(filename1, filename2)
                            score += FPSAlgorithm.LCS_2(filename1, filename2) + \
                                 FPSAlgorithm.LCSubseq_2(filename1, filename2) +\
                                 FPSAlgorithm.LCP_2(filename1, filename2) +\
                                 FPSAlgorithm.LCSubstr_2(filename1, filename2)
                    score /= paths1.__len__() * paths2.__len__()
                    # t1 = list(train_data.loc[train_data['pr_number'] == p1]['pr_created_at'])[0]
                    # t2 = list(train_data.loc[train_data['pr_number'] == p2]['pr_created_at'])[0]
                    # t1 = time.strptime(t1, "%Y-%m-%d %H:%M:%S")
                    # t1 = int(time.mktime(t1))
                    # t2 = time.strptime(t2, "%Y-%m-%d %H:%M:%S")
                    # t2 = int(time.mktime(t2))
                    t1 = prCreatedTimeMap[p1]
                    t2 = prCreatedTimeMap[p2]
                    t = math.fabs(t1 - t2) / (end_time - start_time)
                    score = score * math.exp(-t)

                    # TODO 目测计算非常的耗时间， 后面寻找优化的方案
                    scores[p2] = score
                    scoreMap[(p1, p2)] = score
                    scoreMap[(p2, p1)] = score
                elif p1 > p2:
                    score = scoreMap[(p1, p2)]
                    scores[p2] = score
            """找出K个最近的pr"""
            KNN = [x[0] for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)[0:K]]
            for p2 in KNN:
                trainPrDis[(p1, p2)] = scores[p2]  # p1,p2的顺序可能会造成影响
        return trainPrDis

    @staticmethod
    def RecommendByHG(train_data, train_data_y, test_data, test_data_y, date, project, convertDict, recommendNum=5,
                      K=20, alpha=0.8, c=1):
        """基于超图网络推荐算法
           K 超参数：考虑多少邻近的pr
           alpha 超参数： 类似正则参数
           c 超参数： 用于调节pr相似边的相对权值比例
        """
        recommendList = []
        answerList = []
        testDict = dict(list(test_data.groupby('pr_number')))
        authorList = []

        print("start building hypergraph....")
        start = datetime.now()

        """计算训练集中pr的创建时间"""
        prCreatedTimeMap = {}
        for pr, temp_df in dict(list(train_data.groupby('pr_number'))).items():
            t1 = list(temp_df['pr_created_at'])[0]
            t1 = time.strptime(t1, "%Y-%m-%d %H:%M:%S")
            t1 = int(time.mktime(t1))
            prCreatedTimeMap[pr] = t1

        """计算训练集中pr的距离"""
        tempData = train_data[['pr_number', 'filename']].copy(deep=True)
        tempData.drop_duplicates(inplace=True)
        tempData.reset_index(inplace=True, drop=True)
        pathDict = dict(list(tempData.groupby('pr_number')))
        trainPrDis = HGTrain.getTrainDataPrDistance(train_data, K, pathDict, date, prCreatedTimeMap)
        print(" pr distance cost time:", datetime.now() - start)

        """计算训练集中reviewer的 review 次数
           CN 算法把review次数小于两次的人排除在外  HG里面也可以试试
        """
        tempData = train_data[['pr_number', 'review_user_login']].copy(deep=True)
        tempData.drop_duplicates(inplace=True)
        reviewerFreqDict = {}
        for r, temp_df in dict(list(tempData.groupby('review_user_login'))).items():
            reviewerFreqDict[r] = temp_df.shape[0]

        """计算review -> request权重"""
        prToRevMat = HGTrain.buildPrToRevRelation(train_data)

        """计算author -> reviewer权重"""
        authToRevMat = HGTrain.buildAuthToRevRelation(train_data, date)

        """构建超图"""
        graph = HGTrain.createTrainDataGraph(train_data, train_data_y, trainPrDis, prToRevMat, authToRevMat,
                                             reviewerFreqDict, c)
        # HGTrain.toGephiData(project, date, convertDict, graph)
        # graph.drawGraphByNetworkX()

        prList = list(set(train_data['pr_number']))
        prList.sort()  # 从小到大排序
        prList = tuple(prList)

        startTime = datetime.now()

        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())
        # 结束时间：数据集pr最晚的创建时间
        pr_created_time_data = train_data['pr_created_at'].apply(
            lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())

        pos = 0
        now = datetime.now()
        for test_pull_number, test_df in testDict.items():
            """对每一个单独的例子  添加pr节点和K条边，以及可能添加的作者节点
               在推荐结束之后  删除pr节点和pr的边 对新添加的作者节点也给予删除
            """
            print("now:", pos, ' all:', testDict.items().__len__(), '  cost time:', datetime.now() - now)
            test_df.reset_index(drop=True, inplace=True)
            pos += 1

            """添加pr节点"""
            pr_num = list(test_df['pr_number'])[0]
            paths2 = list(set(test_df['filename']))
            node_1 = graph.add_node(nodeType=Node.STR_NODE_TYPE_PR, contentKey=pr_num, description=f"pr:{pr_num}")

            """增加K条 pr节点和其他节点连接的边"""
            scores = {}  # 记录
            for p1 in prList:
                paths1 = list(pathDict[p1]['filename'])
                score = 0
                for filename1 in paths1:
                    for filename2 in paths2:
                        score += FPSAlgorithm.LCS_2(filename1, filename2) + \
                                 FPSAlgorithm.LCSubseq_2(filename1, filename2) +\
                                 FPSAlgorithm.LCP_2(filename1, filename2) +\
                                 FPSAlgorithm.LCSubstr_2(filename1, filename2)
                        # score += FPSAlgorithm.LCP_2(filename1, filename2)
                score /= paths1.__len__() * paths2.__len__()
                # t2 = list(train_data.loc[train_data['pr_number'] == p1]['pr_created_at'])[0]
                # t2 = time.strptime(t2, "%Y-%m-%d %H:%M:%S")
                # t2 = int(time.mktime(t2))
                t2 = prCreatedTimeMap[p1]
                t = (t2 - start_time) / (end_time - start_time)
                # TODO 目测计算非常的耗时间， 后面寻找优化的方案
                scores[p1] = score * math.exp(t - 1)
                # scores[p1] = score
            """找出K个最近的pr"""
            KNN = [x[0] for x in sorted(scores.items(), key=lambda d: d[1], reverse=True)[0:K]]
            # """找出的K的最相关的pr增加边"""
            for p2 in KNN:
                node_2 = graph.get_node_by_content(Node.STR_NODE_TYPE_PR, p2)
                graph.add_edge(nodes=[node_1.id, node_2.id], edgeType=Edge.STR_EDGE_TYPE_PR_DIS,
                               weight=c * scores[p2], description=f"pr distance between {pr_num} and {p2}",
                               nodeObjects=[node_1, node_2])

            """如果还没有作者节点 添加作者"""
            author = test_df['author_user_login'][0]
            authorList.append(author)

            if author == 237:
                print(test_pull_number)
            if author == 18:
                print(test_pull_number)

            authorNode = graph.get_node_by_content(Node.STR_NODE_TYPE_AUTHOR, author)
            needAddAuthorNode = False  # 如果为True，后面需要把作者节点也删除
            if authorNode is None:
                needAddAuthorNode = True
                authorNode = graph.add_node(nodeType=Node.STR_NODE_TYPE_AUTHOR, contentKey=author, description=f"author:{author}")
            """增加作者和pr之间的边"""
            graph.add_edge(nodes=[node_1.id, authorNode.id], edgeType=Edge.STR_EDGE_TYPE_PR_AUTHOR_RELATION,
                           weight=0.00001, description=f" pr author relation between pr {pr_num} and author {author}",
                           nodeObjects=[node_1, authorNode])

            """重新计算矩阵A"""
            graph.updateMatrix()

            """新建查询向量"""
            y = np.zeros((graph.num_nodes, 1))
            """设置作者和推荐pr的位置为1 参考赋值的第三种方式"""
            nodeInverseMap = {v: k for k, v in graph.node_id_map.items()}
            # y[nodeInverseMap[node_1.id]][0] = 1
            y[nodeInverseMap[authorNode.id]][0] = 1

            """计算顺序列表f"""
            I = np.identity(graph.num_nodes)
            f = np.dot(np.linalg.inv(I - alpha * graph.A), y)

            """对计算结果排序 找到分数较高的几位"""
            if author == 18:
                print(18)
            scores = {}
            for i in range(0, graph.num_nodes):
                node_id = graph.node_id_map[i]
                node = graph.get_node_by_key(node_id)
                if node.type == Node.STR_NODE_TYPE_REVIEWER and reviewerFreqDict[node.contentKey] >= 2:
                    if node.contentKey != author:  # 对作者过滤
                        scores[node.contentKey] = f[i][0]

            answer = answer = test_data_y[pr_num]
            answerList.append(answer)
            recommendList.append([x[0] for x in sorted(scores.items(),
                                                       key=lambda d: d[1], reverse=True)[0:recommendNum]])

            """如果作者节点是最新添加的  则删除"""
            if needAddAuthorNode:
                graph.remove_node_by_key(authorNode.id)
            """删除 pr 节点"""
            graph.remove_node_by_key(node_1.id)

        print("total query cost time:", datetime.now() - startTime)
        return recommendList, answerList, authorList

    @staticmethod
    def buildPrToRevRelation(train_data):
        print("start building request -> reviewer relations....")
        start = datetime.now()
        # 开始时间：数据集pr最早的创建时间
        pr_created_time_data = train_data['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        start_time = min(pr_created_time_data.to_list())
        # 结束时间：数据集pr最晚的创建时间
        pr_created_time_data = train_data['pr_created_at'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        end_time = max(pr_created_time_data.to_list())
        prToRevMat = {}
        grouped_train_data = train_data.groupby([train_data['pr_number'], train_data['review_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = HGTrain.caculateRevToPrWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not prToRevMat.__contains__(relation[0]):
                prToRevMat[relation[0]] = {}
            prToRevMat[relation[0]][relation[1]] = weight

        # 归一化
        for pr, relations in prToRevMat.items():
            for rev, weight in relations.items():
                prToRevMat[pr][rev] = weight/max_weight
        print("finish building request -> reviewer relations. cost time: {0}s".format(datetime.now() - start))
        return prToRevMat

    @staticmethod
    def caculateRevToPrWeight(comment_records, start_time, end_time):
        """计算reviewer和pr之间的权重"""
        weight_lambda = 0.0
        weight = 0
        comment_records = comment_records.copy(deep=True)
        comment_records.drop(columns=['filename'], inplace=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)
        """遍历每条评论，计算权重"""
        for cm_idx, cm_row in comment_records.iterrows():
            cm_timestamp = time.strptime(cm_row['review_created_at'], "%Y-%m-%d %H:%M:%S")
            cm_timestamp = int(time.mktime(cm_timestamp))
            """计算t值: the element t(ij,r,n) is a time-sensitive factor """
            t = (cm_timestamp - start_time) / (end_time - start_time)
            cm_weight = math.pow(weight_lambda, cm_idx) * math.exp(t-1)
            weight += cm_weight
        return weight

    @staticmethod
    def buildAuthToRevRelation(train_data, date):
        print("start building reviewer -> reviewer relations....")
        start = datetime.now()

        # 开始时间：数据集开始时间的前一天
        start_time = time.strptime(str(date[0]) + "-" + str(date[1]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_time = int(time.mktime(start_time) - 86400)
        # 结束时间：数据集的最后一天
        end_time = time.strptime(str(date[2]) + "-" + str(date[3]) + "-" + "01 00:00:00", "%Y-%m-%d %H:%M:%S")
        end_time = int(time.mktime(end_time) - 1)

        authToRevMat = {}
        grouped_train_data = train_data.groupby([train_data['author_user_login'], train_data['review_user_login']])
        max_weight = 0
        for relation, group in grouped_train_data:
            group.reset_index(drop=True, inplace=True)
            weight = HGTrain.caculateAuthToRevWeight(group, start_time, end_time)
            max_weight = max(weight, max_weight)
            if not authToRevMat.__contains__(relation[0]):
                authToRevMat[relation[0]] = {}
            authToRevMat[relation[0]][relation[1]] = weight

        # 归一化
        for auth, relations in authToRevMat.items():
            for rev, weight in relations.items():
                authToRevMat[auth][rev] = weight/max_weight

        print("finish building comments networks! ! ! cost time: {0}s".format(datetime.now() - start))
        return authToRevMat

    @staticmethod
    def caculateAuthToRevWeight(comment_records, start_time, end_time):
        """计算author到reviewer的权重"""
        weight_lambda = 0.8
        weight = 0

        comment_records = comment_records.copy(deep=True)
        comment_records.drop(columns=['filename'], inplace=True)
        comment_records.drop_duplicates(inplace=True)
        comment_records.reset_index(inplace=True, drop=True)

        grouped_comment_records = comment_records.groupby(comment_records['pr_number'])
        for pr, comments in grouped_comment_records:
            comments.reset_index(inplace=True, drop=True)
            """遍历每条评论，计算权重"""
            for cm_idx, cm_row in comments.iterrows():
                cm_timestamp = time.strptime(cm_row['review_created_at'], "%Y-%m-%d %H:%M:%S")
                cm_timestamp = int(time.mktime(cm_timestamp))
                """计算t值: the element t(ij,r,n) is a time-sensitive factor """
                t = (cm_timestamp - start_time) / (end_time - start_time)
                cm_weight = math.pow(weight_lambda, cm_idx) * t
                weight += cm_weight
        return weight

    @staticmethod
    def toGephiData(project, date, convertDict, hyper_graph):
        file_name = f'{os.curdir}/gephi/{project}_{date[0]}_{date[1]}_{date[2]}_{date[3]}_network'

        gexf = Gexf("reviewer_recommend", file_name)
        gexf_graph = gexf.addGraph("directed", "static", file_name)
        gexf_graph.addNodeAttribute("type", 'unknown', type='string')
        gexf_graph.addNodeAttribute("description", '', type='string')
        gexf_graph.addEdgeAttribute("type", 'unknown', type='string')
        gexf_graph.addEdgeAttribute("description", '', type='string')

        tempDict = {k: v for v, k in convertDict.items()}
        edges = hyper_graph.edge_list
        for key, edge in edges.items():
            """将边上的点加入到图中"""
            nodes = edge.connectedTo
            for node in nodes:
                node = hyper_graph.get_node_by_key(node)
                if node.type == Node.STR_NODE_TYPE_PR:
                    gexf_graph.addNode(id=str(node.contentKey), label=node.description,
                                       attrs=[{'id': 0, 'value': node.type},
                                              {'id': 1, 'value': node.description}])
                if node.type == Node.STR_NODE_TYPE_REVIEWER or node.type == Node.STR_NODE_TYPE_AUTHOR:
                    gexf_graph.addNode(id=str(node.contentKey), label=tempDict[node.contentKey],
                                       attrs=[{'id': 0, 'value': node.type},
                                              {'id': 1, 'value': node.description}])
            """将边加入图中"""
            source = hyper_graph.get_node_by_key(edge.connectedTo[0])
            target = hyper_graph.get_node_by_key(edge.connectedTo[1])
            gexf_graph.addEdge(id=edge.id, source=str(source.contentKey), target=str(target.contentKey),
                               weight=edge.weight, attrs=[{'id': 2, 'value': edge.type},
                                      {'id': 3, 'value': edge.description}])
        output_file = open(file_name + ".gexf", "wb")
        gexf.write(output_file)
        output_file.close()
        return file_name + ".gexf"

if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4), (2017, 1, 2018, 5),
             (2017, 1, 2018, 6), (2017, 1, 2018, 7), (2017, 1, 2018, 8), (2017, 1, 2018, 9), (2017, 1, 2018, 10),
             (2017, 1, 2018, 11), (2017, 1, 2018, 12)]
    # projects = ['opencv', 'cakephp', 'yarn', 'akka', 'django', 'react']
    # projects = ['babel', 'xbmc']
    # dates = [(2017, 1, 2018, 1), (2017, 1, 2018, 2), (2017, 1, 2018, 3), (2017, 1, 2018, 4)]
    projects = ['opencv']
    for p in projects:
        HGTrain.TestAlgorithm(p, dates, alpha=0.98, K=20, c=0.1)
