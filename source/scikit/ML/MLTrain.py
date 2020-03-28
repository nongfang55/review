# coding=gbk
from datetime import datetime
import heapq
import time

import numpy
import pandas
from pandas import DataFrame
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from source.config.projectConfig import projectConfig
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.pandas.pandasHelper import pandasHelper

from sklearn.impute import SimpleImputer


class MLTrain:

    @staticmethod
    def testSVMAlgorithms(project, dates):

        recommendNum = 5  # 推荐数量
        excelName = 'outputSVM.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

        for date in dates:
            startTime = datetime.now()
            filename = projectConfig.getRootPath() + r'\data' + r'\\' + \
                       f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}.tsv'
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
            """df做预处理"""
            train_data, train_data_y, test_data, test_data_y = MLTrain.preProcess(df, (date[2], date[3]), isNOR=True)
            """根据算法获得推荐列表"""
            recommendList, answerList = MLTrain.RecommendBySVM(train_data, train_data_y, test_data,
                                                               test_data_y, recommendNum=recommendNum)

            """根据推荐列表做评价"""
            topk, mrr = MLTrain.judgeRecommend(recommendList, answerList, recommendNum)

            """结果写入excel"""
            MLTrain.saveResult(excelName, sheetName, topk, mrr, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

            print("cost time:", datetime.now() - startTime)

    @staticmethod
    def testBayesAlgorithms(project, dates):  # 输入测试日期和对应文件序列  输出一整个算法的表现

        recommendNum = 5  # 推荐数量
        excelName = 'output.xlsx'
        sheetName = 'result'

        """初始化excel文件"""
        ExcelHelper().initExcelFile(fileName=excelName, sheetName=sheetName, excel_key_list=['训练集', '测试集'])

        for i in range(1, 4):  # Bayes 有三个模型
            for date in dates:
                filename = projectConfig.getRootPath() + r'\data' + r'\\' \
                           + f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}.tsv'
                df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
                """df做预处理"""
                isNOR = True
                if i == 1:
                    isNOR = False  # 对伯努利不做归一
                train_data, train_data_y, test_data, test_data_y = MLTrain.preProcess(df, (date[2], date[3]),
                                                                                      isNOR=isNOR)

                """根据算法获得推荐列表"""
                recommendList, answerList = MLTrain.RecommendByNativeBayes(train_data, train_data_y, test_data,
                                                                           test_data_y, recommendNum, i)

                """根据推荐列表做评价"""
                topk, mrr = MLTrain.judgeRecommend(recommendList, answerList, recommendNum)

                """结果写入excel"""
                MLTrain.saveResult(excelName, sheetName, topk, mrr, date)

            """文件分割"""
            content = ['']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())
            content = ['训练集', '测试集']
            ExcelHelper().appendExcelRow(excelName, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def saveResult(filename, sheetName, topk, mrr, date):
        """时间和准确率"""
        content = None
        if date[3] == 1:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2] - 1}.{12}", "TopKAccuracy"]
        else:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2]}.{date[3] - 1}", "TopKAccuracy"]

        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + topk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'MRR']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + mrr
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

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
        if bayesType == 3:
            clf = GaussianNB()
        elif bayesType == 2:
            clf = MultinomialNB()
        elif bayesType == 1:
            clf = BernoulliNB()

        clf.fit(X=train_data, y=train_data_y)
        pre = clf.predict_proba(test_data)
        # print(clf.classes_)
        pre_class = clf.classes_

        recommendList = MLTrain.getListFromProbable(pre, pre_class, recommendNum)
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

        """自定义验证集 而不是使用交叉验证"""
        train_features = numpy.concatenate((train_data, test_data), axis=0)
        train_label = numpy.concatenate((train_data_y, test_data_y), axis=0)
        test_fold = numpy.zeros(train_features.shape[0])
        test_fold[:train_data.shape[0]] = -1
        ps = PredefinedSplit(test_fold=test_fold)

        grid_parameters = [
            # {'kernel': ['rbf'], 'gamma': [0.00075, 0.0001, 0.0002],
            #                 'C': [105, 108, 110, 112, 115], 'decision_function_shape': ['ovr']},
                           {'kernel': ['linear'], 'C': [90, 95, 100],
                            'decision_function_shape': ['ovr']}]  # 调节参数

        # # scores = ['precision', 'recall']  # 判断依据

        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        clf = svm.SVC(C=C, kernel=CoreType, probability=True, gamma=gamma, decision_function_shape=decisionShip)
        clf = GridSearchCV(clf, param_grid=grid_parameters, cv=ps, n_jobs=-1)
        clf.fit(X=train_features, y=train_label)

        print(clf.best_params_)

        # clf = svm.SVC(C=100, kernel='linear', probability=True)
        # clf.fit(train_data, train_data_y)

        pre = clf.predict_proba(test_data)
        pre_class = clf.classes_
        # print(pre)
        # print(pre_class)

        recommendList = MLTrain.getListFromProbable(pre, pre_class, recommendNum)
        # print(recommendList)
        answer = [[x] for x in test_data_y]
        # print(answer)
        return [recommendList, answer]

    @staticmethod
    def preProcess(df, testDate, isSTD=False, isNOR=False):
        """参数说明
         df：读取的dataframe对象
         testDate:作为测试的年月 (year,month)
         isSTD:对数据是否标准化
         isNOR:对数据是否归一化
        """
        columnName = ['reviewer_reviewer', 'pr_number', 'review_id', 'commit_sha', 'author', 'pr_created_at',
                      'pr_commits', 'pr_additions', 'pr_deletions', 'pr_head_label', 'pr_base_label',
                      'review_submitted_at', 'commit_status_total', 'commit_status_additions',
                      'commit_status_deletions', 'commit_files', 'author_review_count',
                      'author_push_count', 'author_submit_gap']
        df.columns = columnName

        """对df添加一列标识训练集和测试集"""
        df['label'] = df['pr_created_at'].apply(
            lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == testDate[0] and
                       time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == testDate[1]))
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
        df.drop(axis=1, columns=['commit_sha', 'review_id', 'pr_number', 'review_submitted_at'
                                 ], inplace=True)
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
    def judgeRecommend(recommendList, answer, recommendNum):

        """评价推荐表现"""
        topk = RecommendMetricUtils.topKAccuracy(recommendList, answer, recommendNum)
        print(topk)
        mrr = RecommendMetricUtils.MRR(recommendList, answer, recommendNum)
        print(mrr)

        return topk, mrr

    @staticmethod
    def getListFromProbable(probable, classList, k):  # 推荐k个
        recommendList = []
        for case in probable:
            max_index_list = list(map(lambda x: numpy.argwhere(case == x), heapq.nlargest(k, case)))
            caseList = []
            for item in max_index_list:
                caseList.append(classList[item[0][0]])
            recommendList.append(caseList)
        return recommendList

    @staticmethod
    def changeStringToNumber(data, columns):  # 对dataframe的一些特征做文本转数字  input: dataFrame，需要处理的某些列
        if isinstance(data, DataFrame):
            count = 0
            convertDict = {}  # 用于转换的字典  开始为1
            for column in columns:
                pos = 0
                for item in data[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    data.at[pos, column] = convertDict[item]
                    pos += 1


if __name__ == '__main__':
    # inputPath = projectConfig.getRootPath() + r'\data\train\ML_rails_data_2018_4_to_2019_4.tsv'
    # featureProcess.preProcess(inputPath)
    # dates = [(2019, 3, 2019, 4), (2019, 1, 2019, 4), (2018, 10, 2019, 4), (2018, 7, 2019, 4), (2018, 4, 2019, 4)]
    dates = [(2019, 1, 2019, 4)]
    MLTrain.testSVMAlgorithms('rails', dates)
