# coding=gbk
from datetime import datetime

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig
from source.data.bean.Commit import Commit
from source.data.bean.File import File
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Review import Review
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class FPSDemo:
    """用于演示 FPS算法的demo类"""

    @staticmethod
    def demo():
        data = pandasHelper.readTSVFile(projectConfig.getFPSTestData(), pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        print("input data:", data.shape)
        startTime = datetime.now()
        # print(DataFrameColumnUtils.COLUMN_REVIEW_FPS)

        """导入pullrequest, review，file，commit数据"""
        pullrequests, pullrequestsIndex = \
            BeanNumpyHelper.getBeansFromDataFrame(PullRequest(),
                                                  DataFrameColumnUtils.COLUMN_REVIEW_FPS_PULL_REQUEST,
                                                  data)
        if configPraser.getPrintMode():
            print(pullrequests.__len__())
            print(pullrequestsIndex)

        time2 = datetime.now()
        print("pull request cost time:", time2 - startTime)

        reviews, reviewsIndex = BeanNumpyHelper.getBeansFromDataFrame(Review(),
                                                                      DataFrameColumnUtils.COLUMN_REVIEW_FPS_REVIEW,
                                                                      data)

        time3 = datetime.now()
        print("review cost time:", time3 - time2)

        if configPraser.getPrintMode():
            print(reviews)
            print(reviewsIndex)
        commits, commitsIndex = BeanNumpyHelper.getBeansFromDataFrame(Commit(),
                                                                      DataFrameColumnUtils.COLUMN_REVIEW_FPS_COMMIT,
                                                                      data)
        time4 = datetime.now()
        print("commits cost time:", time4 - time3)
        if configPraser.getPrintMode():
            print(commits)
            print(commitsIndex)
        files, filesIndex = BeanNumpyHelper.getBeansFromDataFrame(File(),
                                                                  DataFrameColumnUtils.COLUMN_REVIEW_FPS_FILE,
                                                                  data)

        time5 = datetime.now()
        print("file cost time:", time5 - time4)
        if configPraser.getPrintMode():
            print(files)
            print(filesIndex)

        pullrequestReviewIndex = BeanNumpyHelper.beanAssociate(pullrequests, [StringKeyUtils.STR_KEY_REPO_FULL_NAME,
                                                                              StringKeyUtils.STR_KEY_NUMBER],
                                                               reviews, [StringKeyUtils.STR_KEY_REPO_FULL_NAME,
                                                                         StringKeyUtils.STR_KEY_PULL_NUMBER])
        time6 = datetime.now()
        print("pull request index time:", time6 - time5)

        if configPraser.getPrintMode():
            print(pullrequestReviewIndex)

        reviewCommitIndex = BeanNumpyHelper.beanAssociate(reviews, [StringKeyUtils.STR_KEY_COMMIT_ID],
                                                          commits, [StringKeyUtils.STR_KEY_SHA])
        time7 = datetime.now()
        print("commits index cost time:", time7 - time6)

        if configPraser.getPrintMode():
            print(reviewCommitIndex)

        commitFileIndex = BeanNumpyHelper.beanAssociate(commits, [StringKeyUtils.STR_KEY_SHA],
                                                        files, [StringKeyUtils.STR_KEY_COMMIT_SHA])

        time8 = datetime.now()
        print("files index cost time:", time8 - time7)

        if configPraser.getPrintMode():
            print(commitFileIndex)

        receiveTime = datetime.now()
        print("load cost time:", receiveTime - startTime)

        """用于做评价的结果收集"""
        recommendList = []
        answerList = []

        testNumber = configPraser.getTestNumber()
        for pos in range(0, testNumber):
            """通过review算法获取推荐名单"""
            candicateList, authorList = FPSAlgorithm.reviewerRecommend(pullrequests, pullrequestsIndex,
                                                           reviews, reviewsIndex, commits, commitsIndex, files, filesIndex,
                                                           pullrequestReviewIndex,
                                                           reviewCommitIndex, commitFileIndex,
                                                           pos, configPraser.getReviewerNumber())

            print("candicateList", candicateList)

            endTime = datetime.now()
            print("total cost time:", endTime - startTime, " recommend cost time:", endTime - receiveTime)

            recommendList.append(candicateList)
            answerList.append(authorList)
        print(RecommendMetricUtils.topKAccuracy(recommendList, answerList, configPraser.getTopK()))
        print(RecommendMetricUtils.MRR(recommendList, answerList))


if __name__ == '__main__':
    FPSDemo.demo()
