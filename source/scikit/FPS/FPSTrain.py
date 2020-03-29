# coding=gbk
from datetime import datetime

import pandas

from source.config.projectConfig import projectConfig
from source.data.bean.PullRequest import PullRequest
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class FPSTrain:

    @staticmethod
    def TestAlgorithm(trainDataPath, testDataPath):
        """整合 训练数据"""
        # trainData = None
        # for path in trainDataPath:
        #     trainMonthData = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        #     if trainData is None:
        #         trainData = trainMonthData
        #     else:
        #         trainData = trainData.append(trainMonthData)
        # trainData.columns = DataFrameColumnUtils.COLUMN_REVIEW_FPS_ALL
        # print(trainData)
        pos = 1
        for path in trainDataPath:
            trainData = pandasHelper.readTSVFile(path, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
            trainData.columns = DataFrameColumnUtils.COLUMN_REVIEW_FPS_ALL

            testData = pandasHelper.readTSVFile(testDataPath, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
            testData.columns = DataFrameColumnUtils.COLUMN_REVIEW_FPS_ALL
            # print(testData)

            reviews = testData.copy(deep=True)
            reviews = reviews[StringKeyUtils.STR_KEY_ID]
            reviews = reviews.unique()

            """训练集review涉及文件预处理"""
            review_size = {}
            for data in trainData.itertuples():
                review_id = getattr(data, StringKeyUtils.STR_KEY_ID)
                if review_size.get(review_id) is None:
                    review_size[review_id] = 1
                else:
                    review_size[review_id] += 1

            startTime = datetime.now()

            resultList = []  # 是推荐和正确答案的二元组的列表
            for review_id in reviews:
                # print("review_id:", review_id)
                reviewData = testData.loc[testData[StringKeyUtils.STR_KEY_ID] == review_id].reset_index(drop=True)
                # print(reviewData)
                recommendList, answerList = FPSAlgorithm.reviewerRecommendByNumpy(trainData, reviewData, review_size,
                                                                                  10)
                # print(recommendList)
                # print(answerList)
                resultList.append([recommendList, answerList])

            topk = RecommendMetricUtils.topKAccuracy([x[0] for x in resultList], [x[1] for x in resultList], 5)
            print(topk)
            mrr = RecommendMetricUtils.MRR([x[0] for x in resultList], [x[1] for x in resultList], 5)
            print(mrr)

            endTime = datetime.now()
            print("cost:", endTime - startTime)

            """结果写入excel"""
            writer = pandas.ExcelWriter(f'output{pos}.xlsx')
            df1 = pandas.DataFrame(data=topk).T
            df2 = pandas.DataFrame(data=mrr).T
            print(df1)
            print(df2)
            df1.to_excel(writer, sheet_name='topk', )
            df2.to_excel(writer, sheet_name='mrr')
            writer.save()
            pos = pos + 1


if __name__ == '__main__':
    # trainDataPath = [projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_3.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_2.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_1.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_12.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_11.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_10.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_9.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_8.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_7.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_6.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_5.tsv',
    #                  projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_4.tsv']

    trainDataPath = [projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_3_to_2019_3.tsv',
                     projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_1_to_2019_3.tsv',
                     projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2018_10_to_2019_3.tsv']
    # projectConfig.getRootPath() + r'\data\train\FPS_scala_data_2018_7_to_2019_3.tsv',
    # projectConfig.getRootPath() + r'\data\train\FPS_scala_data_2018_4_to_2019_3.tsv']

    testDataPath = projectConfig.getRootPath() + r'\data\train\FPS_rails_data_2019_4_to_2019_4.tsv'
    FPSTrain.TestAlgorithm(trainDataPath, testDataPath)
