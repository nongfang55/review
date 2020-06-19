# coding=gbk
import os
from datetime import datetime
import heapq
import time
from math import ceil

import graphviz
import numpy
import pandas
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import export_graphviz

from source.config.projectConfig import projectConfig
from source.data.service.DataSourceHelper import processFilePathVectorByGensim, appendTextualFeatureVector, \
    appendFilePathFeatureVector
from source.scikit.ML.MultipleLabelAlgorithm import MultipleLabelAlgorithm
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.scikit.service.MLGraphHelper import MLGraphHelper
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper

from sklearn.impute import SimpleImputer


class MLTrain:

    @staticmethod
    def testMLAlgorithms(project, dates, algorithm):
        """
           测试算法接口，把流程相似的算法统一
           algorithm : svm, dt, rf
        """

        recommendNum = 5  # 推荐数量
        excelName = f'output{algorithm}.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

        for date in dates:
            startTime = datetime.now()

            """直接读取不带路径的信息"""
            filename = projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train' + os.sep + \
                       f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}.tsv'
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
            print("raw df:", df.shape)

            # """读取带路径的文件信息"""
            # filename = projectConfig.getRootPath() + os.sep + r'data' + os.sep + 'train' + os.sep + \
            #            f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}_include_filepath.csv'
            # df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD,
            #                               sep=StringKeyUtils.STR_SPLIT_SEP_CSV)

            """df做预处理"""
            train_data, train_data_y, test_data, test_data_y = MLTrain.preProcessForSingleLabel(df, date, project,
                                                                                                isNOR=True)
            recommendList = None
            answerList = None
            """根据算法获得推荐列表"""
            if algorithm == StringKeyUtils.STR_ALGORITHM_SVM:  # 支持向量机
                recommendList, answerList = MLTrain.RecommendBySVM(train_data, train_data_y, test_data,
                                                                   test_data_y, recommendNum=recommendNum)
            elif algorithm == StringKeyUtils.STR_ALGORITHM_DT:  # 决策树
                recommendList, answerList = MLTrain.RecommendByDecisionTree(train_data, train_data_y, test_data,
                                                                            test_data_y, recommendNum=recommendNum)
            elif algorithm == StringKeyUtils.STR_ALGORITHM_RF:  # 随机森林
                recommendList, answerList = MLTrain.RecommendByRandomForest(train_data, train_data_y, test_data,
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
    def testBayesAlgorithms(project, dates):  # 输入测试日期和对应文件序列  输出一整个算法的表现

        recommendNum = 5  # 推荐数量
        excelName = 'outputNB.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

        for i in range(1, 4):  # Bayes 有三个模型
            for date in dates:
                filename = projectConfig.getRootPath() + r'\data\train' + r'\\' \
                           + f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}.tsv'
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
                """df做预处理"""
                isNOR = True
                if i == 1 or i == 3:
                    isNOR = False  # 对伯努利不做归一
                train_data, train_data_y, test_data, test_data_y = MLTrain.preProcessForSingleLabel(df, date, project,
                                                                                                    isNOR=isNOR)

                """根据算法获得推荐列表"""
                recommendList, answerList = MLTrain.RecommendByNativeBayes(train_data, train_data_y, test_data,
                                                                           test_data_y, recommendNum, i)

                """根据推荐列表做评价"""
                topk, mrr = DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

                """结果写入excel"""
                DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def RecommendByNativeBayes(train_data, train_data_y, test_data, test_data_y, recommendNum=5, bayesType=1):
        """使用NB
           recommendNum : 推荐数量
           bayesType : 1 Bernoulli
                       2 Gaussian
                       3 Multionmial

        """
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
        clf = None
        if bayesType == 2:
            clf = GaussianNB()
        elif bayesType == 3:
            clf = MultinomialNB()
            param = {"alpha": [0.2 * x for x in range(0, 10)], "fit_prior": [False, True]}
            clf = GridSearchCV(clf, param_grid=param)
        elif bayesType == 1:
            clf = BernoulliNB()

        clf.fit(X=train_data, y=train_data_y)
        if bayesType == 3:
            print(clf.best_params_, clf.best_score_)

        """查看算法的学习曲线"""
        MLGraphHelper.plot_learning_curve(clf, 'Bayes', train_data, train_data_y).show()

        pre = clf.predict_proba(test_data)
        # print(clf.classes_)
        pre_class = clf.classes_

        recommendList = DataProcessUtils.getListFromProbable(pre, pre_class, recommendNum)
        # print(recommendList)
        answer = [[x] for x in test_data_y]
        # print(answer)
        return [recommendList, answer]

    @staticmethod
    def RecommendBySVM(train_data, train_data_y, test_data, test_data_y, recommendNum=5, CoreType='rbf', C=1,
                       gamma='auto',
                       decisionShip='ovo'):
        """使用SVM
           recommendNum : 推荐数量
           CoreType : 'linear' 线性
                      'rbf' 高斯
           C： 惩罚系数
           gamma： 核参数lambda
           decisionShip: 分类策略
        """

        """设定判断参数"""

        """训练集按照3 7开分成训练集和交叉验证集"""

        """自定义验证集 而不是使用交叉验证"""

        """这里使用交叉验证还是自定义验证需要再研究一下  3.31"""
        test_fold = numpy.zeros(train_data.shape[0])
        test_fold[:ceil(train_data.shape[0] * 0.7)] = -1
        ps = PredefinedSplit(test_fold=test_fold)

        grid_parameters = [
            {'kernel': ['rbf'], 'gamma': [0.0005, 0.00075, 0.0001],
             'C': [100, 105, 108, 110], 'decision_function_shape': ['ovr']}]
        # {'kernel': ['linear'], 'C': [90, 95, 100],
        #  'decision_function_shape': ['ovr', 'ovo'],
        #  'class_weight': ['balanced', None]}]  # 调节参数

        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        clf = svm.SVC(C=C, kernel=CoreType, probability=True, gamma=gamma, decision_function_shape=decisionShip)
        """
          因为REVIEW中有特征是时间相关的  所以讲道理nfold不能使用
          需要自定义验证集 如果使用自定义验证集   GridSearchCVA(CV=ps)

        """
        # clf = GridSearchCV(clf, param_grid=grid_parameters, cv=ps)  # 网格搜索参数
        clf.fit(X=train_data, y=train_data_y)
        # clf.fit(X=train_features, y=train_label)

        # print(clf.best_params_)

        # clf = svm.SVC(C=100, kernel='linear', probability=True)
        # clf.fit(train_data, train_data_y)

        pre = clf.predict_proba(test_data)
        pre_class = clf.classes_
        # print(pre)
        # print(pre_class)
        """查看算法的学习曲线"""
        MLGraphHelper.plot_learning_curve(clf, 'SVM', train_data, train_data_y).show()

        recommendList = DataProcessUtils.getListFromProbable(pre, pre_class, recommendNum)
        # print(recommendList.__len__())
        answer = [[x] for x in test_data_y]
        # print(answer.__len__())
        return [recommendList, answer]

    @staticmethod
    def preProcessForSingleLabel(df, date, project, isSTD=False, isNOR=False):
        """参数说明
         df：读取的dataframe对象
         testDate:作为测试的年月 (year,month)
         isSTD:对数据是否标准化
         isNOR:对数据是否归一化

         之前的单标签问题处理
        """

        # """计算filepath的tf-idf"""
        # df = processFilePathVectorByGensim(df=df)
        # print("filepath df:", df.shape)

        # """在现在的dataframe的基础上面追加review相关的文本的信息特征"""
        # df = appendTextualFeatureVector(df, project, date)

        columnName = ['reviewer_reviewer', 'pr_number', 'review_id', 'commit_sha', 'author', 'pr_created_at',
                      'pr_commits', 'pr_additions', 'pr_deletions', 'pr_head_label', 'pr_base_label',
                      'review_submitted_at', 'commit_status_total', 'commit_status_additions',
                      'commit_status_deletions', 'commit_files', 'author_review_count',
                      'author_push_count', 'author_submit_gap']
        df.columns = columnName

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == date[2] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == date[3]))
        """对人名字做数字处理"""
        MLTrain.changeStringToNumber(df, ['reviewer_reviewer', 'author'])
        print(df.shape)

        """"去除除了时间间隔之外的NAN数据"""
        df = df[~df['pr_head_label'].isna()]
        df = df[~df['pr_created_at'].isna()]
        df = df[~df['review_submitted_at'].isna()]
        df.reset_index(drop=True, inplace=True)
        print(df.shape)

        """对branch做处理  舍弃base,head做拆分 并数字化"""
        df.drop(axis=1, columns=['pr_base_label'], inplace=True)  # inplace 代表直接数据上面
        df['pr_head_tail'] = df['pr_head_label']
        df['pr_head_tail'] = df['pr_head_tail'].apply(lambda x: x.split(':')[1])
        df['pr_head_label'] = df['pr_head_label'].apply(lambda x: x.split(':')[0])

        MLTrain.changeStringToNumber(df, ['pr_head_tail'])
        MLTrain.changeStringToNumber(df, ['pr_head_label'])

        """时间转时间戳处理"""
        df['pr_created_at'] = df['pr_created_at'].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        df['review_submitted_at'] = df['review_submitted_at'].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))

        """去除无用的 commit_sha, review_id 和 pr_number 和review_submitted_at"""
        df.drop(axis=1, columns=['commit_sha', 'review_id', 'pr_number', 'review_submitted_at'], inplace=True)
        # inplace 代表直接数据上面

        """参数处理缺省值"""
        df.fillna(value=999999999999, inplace=True)
        # print(df)

        """测试集和训练集分开"""
        test_data = df.loc[df['label']].copy(deep=True)

        print("test:", test_data.shape)
        train_data = df[df['label'] == False].copy(deep=True)
        print("train:", train_data.shape)

        test_data.drop(axis=1, columns=['label'], inplace=True)
        train_data.drop(axis=1, columns=['label'], inplace=True)

        """分割 tag和feature"""

        test_data_y = test_data['reviewer_reviewer'].copy(deep=True)
        test_data.drop(axis=1, columns=['reviewer_reviewer'], inplace=True)

        train_data_y = train_data['reviewer_reviewer'].copy(deep=True)
        train_data.drop(axis=1, columns=['reviewer_reviewer'], inplace=True)

        # """主成分分析"""
        # pca = PCA()
        # train_data = pca.fit_transform(train_data)
        # print("after pca train:", train_data.shape)
        # print(pca.explained_variance_ratio_)
        # test_data = pca.transform(test_data)
        # print("after pca test:", test_data.shape)

        """参数规范化"""
        if isSTD:
            stdsc = StandardScaler()
            train_data_std = stdsc.fit_transform(train_data)
            test_data_std = stdsc.transform(test_data)
            # print(train_data_std)
            # print(test_data_std.shape)
            return train_data_std, train_data_y, test_data_std, test_data_y
        elif isNOR:
            maxminsc = MinMaxScaler()
            train_data_std = maxminsc.fit_transform(train_data)
            test_data_std = maxminsc.transform(test_data)
            return train_data_std, train_data_y, test_data_std, test_data_y
        else:
            return train_data, train_data_y, test_data, test_data_y

    @staticmethod
    def preProcess(df, date, project, featureType, isSTD=False, isNOR=False):
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

        # """在现有的特征中添加文本路径特征"""
        """更正说明：由于PCA不能训练集和测试集同时降维，否则相当于使用了后面的信息
           所以添加之前必须两者分别处理 4.13 
           append 函数必须在表明label后面使用"""

        if featureType == 1 or featureType == 3:
            df = appendFilePathFeatureVector(df, project, date, 'pr_number')
        """在现有的特征中添加pr标题和内容文本特征"""
        if featureType == 2 or featureType == 3:
            df = appendTextualFeatureVector(df, project, date, 'pr_number')

        """频率统计每一个reviewer的次数，排除数量过少的reviewer"""
        freq = {}
        for data in df.itertuples(index=False):
            name = data[list(df.columns).index('review_user_login')]
            if freq.get(name, None) is None:
                freq[name] = 0
            """训练集用户次数加一  测试集直接保留 """
            if not data[list(df.columns).index('label')]:
                freq[name] += 1
            else:
                freq[name] += 1

        num = 5
        df['freq'] = df['review_user_login'].apply(lambda x: freq[x])
        df = df.loc[df['freq'] > num].copy(deep=True)
        df.drop(columns=['freq'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("after lifter unexperienced user:", df.shape)

        # # # 画出参与人数的频度图
        # MLTrain.getSeriesBarPlot(df['review_user_login'])

        """对人名字做数字处理"""
        """频率不过的评审者在编号之前就已经过滤了，不用考虑分类不连续的情况"""
        """这里reviewer_user_login 放在 第一个否则会影响candicateNum这个变量在后面的引用"""
        convertDict = DataProcessUtils.changeStringToNumber(df, ['review_user_login', 'pr_user_login'])
        print(df.shape)
        candicateNum = max(df.loc[df['label'] == 0]['review_user_login'])
        print("candicate Num:", candicateNum)

        """对branch做处理  舍弃base,head做拆分 并数字化"""
        df.drop(axis=1, columns=['pr_base_label'], inplace=True)  # inplace 代表直接数据上面
        df['pr_head_tail'] = df['pr_head_label']
        df['pr_head_tail'] = df['pr_head_tail'].apply(lambda x: x.split(':')[1])
        df['pr_head_label'] = df['pr_head_label'].apply(lambda x: x.split(':')[0])

        df.drop(axis=1, columns=['pr_head_tail'], inplace=True)

        # MLTrain.changeStringToNumber(df, ['pr_head_tail'])
        DataProcessUtils.changeStringToNumber(df, ['pr_head_label'])

        """时间转时间戳处理"""
        df['pr_created_at'] = df['pr_created_at'].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))

        """先对tag做拆分"""
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
        """训练集 结果做出多标签分类通用的模式"""
        train_data_y = DataProcessUtils.convertLabelListToDataFrame(train_data_y, pull_number_list, candicateNum)

        test_data_y = {}
        pull_number_list = test_data.drop_duplicates(['pr_number']).copy(deep=True)['pr_number']
        for pull_number in test_data.drop_duplicates(['pr_number'])['pr_number']:
            reviewers = list(tagDict[pull_number].drop_duplicates(['review_user_login'])['review_user_login'])
            test_data_y[pull_number] = reviewers

        test_data.drop(columns=['review_user_login'], inplace=True)
        test_data.drop_duplicates(inplace=True)
        # test_data_y = DataProcessUtils.convertLabelListToDataFrame(test_data_y, pull_number_list, candicateNum)
        test_data_y = DataProcessUtils.convertLabelListToListArray(test_data_y, pull_number_list)

        """获得pr list"""
        prList = list(test_data['pr_number'])

        """去除pr number"""
        test_data.drop(columns=['pr_number'], inplace=True)
        train_data.drop(columns=['pr_number'], inplace=True)

        """参数规范化"""
        if isSTD:
            stdsc = StandardScaler()
            train_data_std = stdsc.fit_transform(train_data)
            test_data_std = stdsc.transform(test_data)
            # print(train_data_std)
            # print(test_data_std.shape)
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
    def RecommendByDecisionTree(train_data, train_data_y, test_data, test_data_y, recommendNum=5):
        """使用决策树
           recommendNum : 推荐数量
           max_depth 决策树最大深度
           min_samples_split 内部节点划分所需最小样本数
           min_samples_leaf 叶子节点最小样本数
           class_weight 分类权重
        """

        """设定判断参数"""

        """训练集按照3 7开分成训练集和交叉验证集"""

        """自定义验证集 而不是使用交叉验证"""
        test_fold = numpy.zeros(train_data.shape[0])
        test_fold[:ceil(train_data.shape[0] * 0.7)] = -1
        ps = PredefinedSplit(test_fold=test_fold)

        grid_parameters = [
            {'min_samples_leaf': [2, 4, 8, 16, 32, 64], 'max_depth': [2, 4, 6, 8],
             'class_weight': [None]}]  # 调节参数

        # # scores = ['precision', 'recall']  # 判断依据

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        clf = DecisionTreeClassifier()
        clf = GridSearchCV(clf, param_grid=grid_parameters, cv=ps, n_jobs=-1)
        clf.fit(train_data, train_data_y)

        print(clf.best_params_)
        # dot_data = export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("DTree")

        pre = clf.predict_proba(test_data)
        pre_class = clf.classes_
        # print(pre)
        # print(pre_class)

        recommendList = DataProcessUtils.getListFromProbable(pre, pre_class, recommendNum)
        # print(recommendList)
        answer = [[x] for x in test_data_y]
        # print(answer)
        return [recommendList, answer]

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
    def RecommendByRandomForest(train_data, train_data_y, test_data, test_data_y, recommendNum=5):
        """使用随机森林
           n_estimators : 最大弱学习器个数
           recommendNum : 推荐数量
           max_depth 决策树最大深度
           min_samples_split 内部节点划分所需最小样本数
           min_samples_leaf 叶子节点最小样本数
           class_weight 分类权重
        """

        """设定判断参数"""

        """自定义验证集 而不是使用交叉验证"""
        test_fold = numpy.zeros(train_data.shape[0])
        test_fold[:ceil(train_data.shape[0] * 0.7)] = -1
        ps = PredefinedSplit(test_fold=test_fold)

        """导入模型"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        clf = RandomForestClassifier(min_samples_split=100,
                                     min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10)
        # clf = GridSearchCV(clf, param_grid=grid_parameters, cv=ps, n_jobs=-1)
        # clf.fit(train_data, train_data_y)
        #
        # print("OOB SCORE:", clf.oob_score_)

        """对弱分类器数量做调参数量"""
        # param_test1 = {'n_estimators': range(10, 200, 10)}
        # clf = GridSearchCV(estimator=clf, param_grid=param_test1)
        # clf.fit(train_data, train_data_y)
        # print(clf.best_params_, clf.best_score_)

        """对决策树的参数做调参"""
        param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
        clf = GridSearchCV(estimator=clf, param_grid=param_test2, iid=False, cv=5)
        clf.fit(train_data, train_data_y)
        # gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

        """查看算法的学习曲线"""
        MLGraphHelper.plot_learning_curve(clf, 'RF', train_data, train_data_y).show()

        pre = clf.predict_proba(test_data)
        pre_class = clf.classes_
        # print(pre)
        # print(pre_class)

        recommendList = DataProcessUtils.getListFromProbable(pre, pre_class, recommendNum)
        # print(recommendList)
        answer = [[x] for x in test_data_y]
        # print(answer)
        return [recommendList, answer]

    @staticmethod
    def testMLAlgorithmsByMultipleLabels(projects, dates, algorithms=None):
        """
           多标签测试算法接口，把流程相似的算法统一
        """
        startTime = datetime.now()

        for algorithmType in algorithms:
            for project in projects:
                excelName = f'output{algorithmType}_{project}_ML.xlsx'
                recommendNum = 5  # 推荐数量
                sheetName = 'result'
                """初始化excel文件"""
                ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])
                for featureType in range(0, 1):
                    """初始化项目抬头"""
                    content = ["项目名称：", project]
                    ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
                    content = ['特征类型：', str(featureType)]
                    ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

                    """计算累积数据"""
                    topks = []
                    mrrs = []
                    precisionks = []
                    recallks = []
                    fmeasureks = []

                    for date in dates:
                        recommendList, answerList, prList, convertDict, trainSize = MLTrain.algorithmBody(date, project,
                                                                                               algorithmType,
                                                                                               recommendNum,
                                                                                               featureType)
                        """根据推荐列表做评价"""
                        topk, mrr, precisionk, recallk, fmeasurek = \
                            DataProcessUtils.judgeRecommend(recommendList, answerList, recommendNum)

                        topks.append(topk)
                        mrrs.append(mrr)
                        precisionks.append(precisionk)
                        recallks.append(recallk)
                        fmeasureks.append(fmeasurek)

                        """结果写入excel"""
                        DataProcessUtils.saveResult(excelName, sheetName, topk, mrr, precisionk, recallk, fmeasurek,
                                                    date)

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
    def algorithmBody(date, project, algorithmType, recommendNum=5, featureType=3):
        df = None
        """对需求文件做合并 """
        for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):  # 拆分的数据做拼接
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1

            print(y, m)
            filename = projectConfig.getMLDataPath() + os.sep + f'ML_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
            """数据自带head"""
            if df is None:
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
            else:
                temp = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD)
                df = df.append(temp)  # 合并

        df.reset_index(inplace=True, drop=True)
        """df做预处理"""
        """获取测试的 pull number列表"""
        train_data, train_data_y, test_data, test_data_y, convertDict, prList = MLTrain.preProcess(df, date, project,
                                                                                                   featureType,
                                                                                                   isNOR=True)
        print("train data:", train_data.shape)
        print("test data:", test_data.shape)

        recommendList, answerList = MultipleLabelAlgorithm. \
            RecommendByAlgorithm(train_data, train_data_y, test_data, test_data_y, algorithmType)

        trainSize = (train_data.shape[0], test_data.shape[0])

        return recommendList, answerList, prList, convertDict, trainSize


if __name__ == '__main__':
    # dates = [(2018, 1, 2019, 4), (2018, 1, 2019, 5), (2018, 1, 2019, 6), (2018, 1, 2019, 7), (2018, 1, 2019, 7),
    #          (2018, 1, 2019, 8)
    #     , (2018, 1, 2019, 9), (2018, 1, 2019, 10), (2018, 1, 2019, 11), (2018, 1, 2019, 12)]
    # dates = [(2018, 1, 2019, 5), (2018, 1, 2019, 6), (2018, 1, 2019, 7), (2018, 1, 2019, 8), (2018, 1, 2019, 9)
    #     , (2018, 1, 2019, 10)]
    # dates = [(2018, 1, 2019, 12)]
    # MLTrain.testMLAlgorithms('rails', dates, StringKeyUtils.STR_ALGORITHM_DT)
    # MLTrain.testBayesAlgorithms('rails', dates)
    # projects = ['rails', 'scala', 'akka', 'bitcoin']
    dates = [(2018, 1, 2019, 1)]
    projects = ['cakephp']
    MLTrain.testMLAlgorithmsByMultipleLabels(projects, dates, [0])
