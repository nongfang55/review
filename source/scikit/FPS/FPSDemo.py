# coding=gbk
from ctypes import *
from datetime import datetime, time

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig
from source.data.bean.Commit import Commit
from source.data.bean.File import File
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Review import Review
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.FPS.FPSClassCovert import *
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

        if configPraser.getFPSCtypes():
            """调用dll库实现增加运行速度"""
            dll = CDLL("cFPS.dll")
            dll.addf.restype = c_float
            dll.addf.argtypes = [c_float, c_float]
            print(dll.addf(10, 30))

            c_prs = FPSClassCovert.convertPullRequest(pullrequests)
            c_reviews = FPSClassCovert.convertReview(reviews)
            c_commits = FPSClassCovert.convertCommit(commits)
            c_files = FPSClassCovert.convertFile(files)

            c_result = c_fps_result()
            print(c_prs)
            print(c_reviews)
            print(c_commits)
            print(c_files)

            dll.FPS.restype = None
            dll.FPS.argtypes = (POINTER(c_fps_pr), c_int, POINTER(c_fps_review), c_int,
                                POINTER(c_fps_commit), c_int, POINTER(c_fps_file), c_int,
                                POINTER(c_fps_result), c_int, c_int)

            prs_num = c_prs.__len__()
            p_c_prs = (c_fps_pr * prs_num)(*c_prs)
            reviews_num = c_reviews.__len__()
            p_c_reviews = (c_fps_review * reviews_num)(*c_reviews)
            commits_num = c_commits.__len__()
            p_c_commits = (c_fps_commit * commits_num)(*c_commits)
            files_num = c_files.__len__()
            p_c_files = (c_fps_file * files_num)(*c_files)

            dll.FPS(p_c_prs, prs_num, p_c_reviews, reviews_num, p_c_commits,
                    commits_num, p_c_files, files_num, pointer(c_result), 0, 10, True)

            endTime = datetime.now()
            print("total cost time:", endTime - startTime, " recommend cost time:", endTime - receiveTime)

            print("answer:", str(c_result.answer, encoding='utf-8'))
            print("recommend:", str(c_result.recommend, encoding='utf-8'))

        else:
            """使用Python实现算法 但是很慢"""
            for pos in range(0, testNumber):
                """通过review算法获取推荐名单"""
                candicateList, authorList = FPSAlgorithm.reviewerRecommend(pullrequests, pullrequestsIndex,
                                                                           reviews, reviewsIndex, commits, commitsIndex,
                                                                           files, filesIndex,
                                                                           pullrequestReviewIndex,
                                                                           reviewCommitIndex, commitFileIndex,
                                                                           pos, configPraser.getReviewerNumber())

                print("candicateList", candicateList)
                endTime = datetime.now()
                print("total cost time:", endTime - startTime, " recommend cost time:", endTime - receiveTime)

                recommendList.append(candicateList)
                answerList.append(authorList)

        # print(RecommendMetricUtils.topKAccuracy(recommendList, answerList, configPraser.getTopK()))
        # print(RecommendMetricUtils.MRR(recommendList, answerList))


if __name__ == '__main__':
    FPSDemo.demo()
