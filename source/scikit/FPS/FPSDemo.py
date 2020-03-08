# coding=gbk
from source.config.projectConfig import projectConfig
from source.data.bean.Commit import Commit
from source.data.bean.File import File
from source.data.bean.Review import Review
from source.scikit.FPS.FPSAlgorithm import FPSAlgorithm
from source.scikit.service.BeanNumpyHelper import BeanNumpyHelper
from source.scikit.service.DataFrameColumnUtils import DataFrameColumnUtils
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper


class FPSDemo:
    """用于演示 FPS算法的demo类"""

    @staticmethod
    def demo():
        data = pandasHelper.readTSVFile(projectConfig.getFPSTestData(), pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        print(data.shape)
        # print(DataFrameColumnUtils.COLUMN_REVIEW_FPS)

        """导入review，file，commit数据"""

        reviews, reviewsIndex = BeanNumpyHelper.getBeansFromDataFrame(Review(),
                                                                      DataFrameColumnUtils.COLUMN_REVIEW_FPS_REVIEW,
                                                                      data)
        print(reviews)
        print(reviewsIndex)
        commits, commitsIndex = BeanNumpyHelper.getBeansFromDataFrame(Commit(),
                                                                      DataFrameColumnUtils.COLUMN_REVIEW_FPS_COMMIT,
                                                                      data)
        print(commits)
        print(commitsIndex)
        files, filesIndex = BeanNumpyHelper.getBeansFromDataFrame(File(),
                                                                  DataFrameColumnUtils.COLUMN_REVIEW_FPS_FILE,
                                                                  data)
        print(files)
        print(filesIndex)

        reviewCommitIndex = BeanNumpyHelper.beanAssociate(reviews, [StringKeyUtils.STR_KEY_COMMIT_ID],
                                                          commits, [StringKeyUtils.STR_KEY_SHA])
        print(reviewCommitIndex)

        commitFileIndex = BeanNumpyHelper.beanAssociate(commits, [StringKeyUtils.STR_KEY_SHA],
                                                        files, [StringKeyUtils.STR_KEY_COMMIT_SHA])
        print(commitFileIndex)

        """通过review算法获取推荐名单"""
        FPSAlgorithm.reviewerRecommend(reviews, reviewsIndex, commits, commitsIndex, files, filesIndex,
                                       reviewCommitIndex, commitFileIndex, 0, 2)




if __name__ == '__main__':
    FPSDemo.demo()