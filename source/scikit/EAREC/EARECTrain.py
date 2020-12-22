# coding=gbk
import sys
import os

import pandas

sys.path.append("/root/zjq_rev")
import time
from datetime import datetime

from pandas import DataFrame
import numpy

from source.config.projectConfig import projectConfig
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.EAREC.Edge import Edge
from source.scikit.EAREC.Node import Node
from source.scikit.EAREC.Graph import Graph
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper
from gensim import corpora, models


class EARECTrain:
    """EAREC算法"""

    @staticmethod
    def testEARECAlgorithm(project, dates, filter_train=False, filter_test=False, a=0.5):
        """整合 训练数据"""
        recommendNum = 5  # 推荐数量
        excelName = f'outputEAREC_{project}_{filter_train}_{filter_test}.xls'
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
            recommendList, answerList, prList, convertDict, trainSize = EARECTrain.algorithmBody(date, project,
                                                                                                 recommendNum,
                                                                                                 filter_train=filter_train,
                                                                                                 filter_test=filter_test,
                                                                                                 a=a)
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
    def algorithmBody(date, project, recommendNum=5, filter_train=False, filter_test=False, a=0.5):

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
                    filename = projectConfig.getEARECDataPath() + os.sep + f'EAREC_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getEARECDataPath() + os.sep + f'EAREC_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            else:
                if filter_test:
                    filename = projectConfig.getEARECDataPath() + os.sep + f'EAREC_{project}_data_change_trigger_{y}_{m}_to_{y}_{m}.tsv'
                else:
                    filename = projectConfig.getEARECDataPath() + os.sep + f'EAREC_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """新增人名映射字典"""
        train_data, train_data_y, test_data, test_data_y, convertDict = EARECTrain.preProcess(df, date)

        prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
        # prList.sort()

        recommendList, answerList, = EARECTrain.RecommendByEAREC(train_data, train_data_y, test_data,
                                                                 test_data_y, convertDict, recommendNum=recommendNum,
                                                                 a=a)

        """保存推荐结果到本地"""
        DataProcessUtils.saveRecommendList(prList, recommendList, answerList, convertDict, key=project + str(date))

        """新增返回训练集 测试集大小"""
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
        df['pr_title'].fillna(value='', inplace=True)
        df['pr_body'].fillna(value='', inplace=True)

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == dates[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == dates[3]))
        """对reviewer名字数字化处理 存储人名映射字典做返回"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['pr_author', 'reviewer'])

        """用于收集所有文本向量分词"""
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

        """问题:lsi的过程不能在整个数据集上面做，不然会导致pr的文本重复问题"""
        df_pr = df.copy(deep=True)
        df_pr.drop_duplicates(subset=['pull_number'], keep='first', inplace=True)
        df_pr.reset_index(drop=True, inplace=True)

        # 用于记录pr中文字的数量，对于pr少于10个word的pr.直接去掉
        df_pr_word_count = []

        textList = []
        for row in df_pr.itertuples(index=False, name='Pandas'):
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
            if tempList.__len__() >= 10 or getattr(row, 'label'):
                textList.append(tempList)
            if getattr(row, 'label'):
                df_pr_word_count.append(10)  # 以便过后面的过滤
            else:
                df_pr_word_count.append(tempList.__len__())

        """去除无用的训练pr"""
        df_pr['count'] = df_pr_word_count
        df_pr = df_pr.loc[df_pr['count'] >= 10].copy(deep=True)
        df_pr.reset_index(drop=True, inplace=True)
        df_pr.drop(['count'], inplace=True, axis=1)

        """保存只有pr的列表"""
        prList = list(df_pr['pull_number'])

        """对已经有的本文特征向量和标签做训练集和测试集的拆分"""
        trainData_index = df_pr.loc[df_pr['label'] == False].index
        testData_index = df_pr.loc[df_pr['label'] == True].index

        trainDataTextList = [textList[x] for x in trainData_index]
        testDataTextList = [textList[x] for x in testData_index]

        print(textList.__len__())
        """对分词列表建立字典 并提取特征数"""
        dictionary = corpora.Dictionary(trainDataTextList)
        print('词典：', dictionary)

        """感觉有问题，tfidf模型不应该是在全数据集上面计算，而是在训练集上面计算，而测试集的向量就是
        单纯的带入模型的计算结果"""

        """根据词典建立语料库"""
        corpus = [dictionary.doc2bow(text) for text in trainDataTextList]
        # print('语料库:', corpus)
        """语料库训练TF-IDF模型"""
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        topic_num = 10
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)
        topic_list = lsi.print_topics()
        print("{0}个主题的单词分布为：\n".format(topic_num))
        for topic in topic_list:
            print(topic)

        """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
        wordVectors = []
        for i in range(0, trainDataTextList.__len__()):
            wordVectors.append(dict(lsi[dictionary.doc2bow(trainDataTextList[i])]))
        for i in range(0, testDataTextList.__len__()):
            wordVectors.append(dict(lsi[dictionary.doc2bow(testDataTextList[i])]))

        """训练集"""
        train_data = [wordVectors[x] for x in trainData_index]
        """测试集"""
        test_data = [wordVectors[x] for x in testData_index]
        """填充为向量"""
        train_v_data = DataProcessUtils.convertFeatureDictToDataFrame(train_data, featureNum=topic_num)
        test_v_data = DataProcessUtils.convertFeatureDictToDataFrame(test_data, featureNum=topic_num)

        lsi_data = pandas.concat([train_v_data, test_v_data], axis=0)  # 0 轴合并
        lsi_data['pull_number'] = prList
        lsi_data.reset_index(inplace=True, drop=True)

        train_data = df.loc[df['label'] == False]
        train_data.reset_index(drop=True, inplace=True)
        test_data = df.loc[df['label'] == True]
        test_data.reset_index(drop=True, inplace=True)

        train_data = train_data.merge(lsi_data, on="pull_number")
        train_data.drop(columns=['label'], inplace=True)

        test_data = test_data.merge(lsi_data, on="pull_number")
        test_data.drop(columns=['label'], inplace=True)

        """8ii处理NAN"""
        train_data.dropna(how='any', inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        train_data.fillna(value='', inplace=True)

        """先对tag做拆分"""
        trainDict = dict(list(train_data.groupby('pull_number')))
        testDict = dict(list(test_data.groupby('pull_number')))

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
    def RecommendByEAREC(train_data, train_data_y, test_data, test_data_y, convertDict, recommendNum=5, a=0.5):
        """EAREC推荐算法"""

        recommendList = []
        answerList = []

        print("start building reviewer<->reviewer relations....")
        start = datetime.now()

        """计算train_data的矩阵"""
        df_train = train_data.copy(deep=True)
        df_train = df_train.iloc[:, 12:22]
        """计算test_data矩阵"""
        df_test = test_data.copy(deep=True)
        df_test = df_test.iloc[:, 12:22]

        """计算距离"""
        DIS = DataFrame(numpy.dot(df_test, df_train.T))

        """计算模长"""
        train_len_dict = {}
        test_len_dict = {}

        for index, row in train_data.iterrows():
            train_len_dict[row['pull_number']] = numpy.linalg.norm(row[12:22])
        for index, row in test_data.iterrows():
            test_len_dict[row['pull_number']] = numpy.linalg.norm(row[12:22])

        graph = Graph()

        """增加候选人的顶点"""
        candidates = list(set(train_data['reviewer']))
        for candidate in candidates:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_REVIEWER, contentKey=candidate,
                           description=f"reviewer:{candidate}")

        # 用于计算两个评审者之间的相关程度
        scoreMap = {}
        """构造reviewer, reviewer关系"""
        grouped_train_data = train_data.groupby(train_data['pull_number'])
        for pr, group in grouped_train_data:
            reviewers = list(set(group['reviewer'].to_list()))
            reviewers = sorted(reviewers)
            for i in range(0, reviewers.__len__()):
                for j in range(i + 1, reviewers.__len__()):
                    if scoreMap.get((reviewers[i], reviewers[j]), None) is None:
                        scoreMap[(reviewers[i], reviewers[j])] = 0
                    scoreMap[(reviewers[i], reviewers[j])] += 1
                    # reviewer_i = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, reviewers[i])
                    # reviewer_j = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, reviewers[j])
                    # # 边权累加
                    # graph.add_edge(nodes=[reviewer_i.id, reviewer_j.id],
                    #                edgeType=Edge.STR_EDGE_TYPE_REVIEWER_REVIEWER,
                    #                weight=1,
                    #                description=f" pr review relation between reviewer {reviewers[i]} and reviewer {reviewers[j]}")
        for reviewers, weight in scoreMap.items():
            i, j = reviewers
            reviewer_i = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, i)
            reviewer_j = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, j)
            graph.add_edge(nodes=[reviewer_i.id, reviewer_j.id],
                           edgeType=Edge.STR_EDGE_TYPE_REVIEWER_REVIEWER,
                           weight=weight,
                           description=f" pr review relation between reviewer {i} and reviewer {j}")
        print("finish building reviewer<->reviewer relations!  cost time: {0}s".format(datetime.now() - start))

        print("start building reviewer<->ipr relations....")
        test_pr_list = tuple(test_data['pull_number'])  # 用set压缩会导致后面dis读取错位
        train_pr_list = tuple(train_data['pull_number'])

        prList = list(test_data.drop_duplicates(['pull_number'])['pull_number'])
        cur = 1
        for pr_num in prList:

            print(cur, "  all:", prList.__len__())
            cur += 1

            """添加pr节点"""
            pr_node = graph.add_node(nodeType=Node.STR_NODE_TYPE_PR, contentKey=pr_num, description=f"pr:{pr_num}")

            """初始化p向量"""
            p = numpy.zeros((graph.num_nodes, 1))

            for candidate in candidates:
                candidateNode = graph.get_node_by_content(Node.STR_NODE_TYPE_REVIEWER, candidate)
                """找到候选者评审过的pr"""
                commented_pr_df = train_data[train_data['reviewer'] == candidate]
                max_score = commented_pr_df.shape[0]
                commented_pr_df_grouped = commented_pr_df.groupby(commented_pr_df['pull_number'])
                score = 0
                for pr, comments in commented_pr_df_grouped:
                    index_train = train_pr_list.index(pr)
                    index_test = test_pr_list.index(pr_num)
                    score += comments.shape[0] * DIS.iloc[index_test][index_train] / (
                                train_len_dict[pr] * test_len_dict[pr_num])
                score /= max_score

                """更新p向量"""
                p[candidateNode.id] = score

                """增加reviewer->ipr边"""
                graph.add_edge(nodes=[candidateNode.id, pr_node.id],
                               edgeType=Edge.STR_EDGE_TYPE_PR_REVIEW_RELATION,
                               weight=score,
                               description=f" pr review relation between reviewer {candidate} and ipr {pr_num}")
            """更新w矩阵"""
            graph.updateW()
            """设置q向量"""
            q = numpy.zeros((graph.num_nodes, 1))
            q[pr_node.id][0] = 1

            """迭代六次"""
            for c in range(0, 6):  # 迭代6次
                tmp = numpy.dot(graph.W, p)
                p = (1 - a) * tmp + a * q

            """最后得出的p，看谁的分数在前五"""
            score_dict = {}
            for i in range(0, p.__len__() - 1):
                node = graph.get_node_by_key(i)
                score_dict[node.contentKey] = p[i]

            recommendList.append(
                [x[0] for x in sorted(score_dict.items(), key=lambda d: d[1], reverse=True)[0:recommendNum]])
            answerList.append(test_data_y[pr_num])

            """删除 pr 节点"""
            graph.remove_node_by_key(pr_node.id)

        return recommendList, answerList


if __name__ == '__main__':
    dates = [(2017, 1, 2018, 1)]
    projects = ['scikit-learn']
    for p in projects:
        projectName = p
        """论文里λ用0.1-0.9测试的，每个项目选了最好的topk作为结果，没有统一λ，这里折中取了0.5"""
        EARECTrain.testEARECAlgorithm(projectName, dates, filter_train=False, filter_test=False, a=0.5)
