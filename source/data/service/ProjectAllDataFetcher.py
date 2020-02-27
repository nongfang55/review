# coding=gbk
from concurrent.futures._base import ALL_COMPLETED, wait
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
import time

from retrying import retry

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig
from source.data.service.ApiHelper import ApiHelper
from source.database.SqlExecuteHelper import SqlExecuteHelper
from source.database.SqlUtils import SqlUtils
from source.utils.statisticsHelper import statisticsHelper
from source.database.SqlServerInterceptor import SqlServerInterceptor


class ProjectAllDataFetcher:
    """用于获取项目所有信息的类"""

    @staticmethod
    def getAllDataForProject(owner, repo):

        helper = ApiHelper(owner=owner, repo=repo)
        helper.setAuthorization(True)

        statistic = statisticsHelper()
        statistic.startTime = datetime.now()

        '''提取项目的信息以及项目的owner信息'''
        ProjectAllDataFetcher.getDataForRepository(helper)
        '''提取项目的pull request信息'''
        ProjectAllDataFetcher.getPullRequestForRepositoryUseConcurrent(helper, limit=configPraser.getLimit(),
                                                          statistic=statistic, start=37600)

        statistic.endTime = datetime.now()

        print("useful pull request:", statistic.usefulRequestNumber,
              " useful review:", statistic.usefulReviewNumber,
              " useful review comment:", statistic.usefulReviewCommentNumber,
              " useful issue comment:", statistic.usefulIssueCommentNumber,
              " useful commit:", statistic.usefulCommitNumber,
              " cost time:", (statistic.endTime - statistic.startTime).seconds)

    @staticmethod
    def getDataForRepository(helper):

        exceptionTime = 0
        project = None

        while exceptionTime < configPraser.getRetryTime():
            try:
                project = helper.getInformationForProject()
                break
            except Exception as e:
                if exceptionTime < 5:
                    time.sleep(5)
                else:
                    time.sleep(20)
                exceptionTime += 1
                print(e)

        if exceptionTime == configPraser.getRetryTime():
            raise Exception("error out the limit!")

        if project is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REPOS
                                                   , project.getItemKeyList()
                                                   , project.getValueDict()
                                                   , project.getIdentifyKeys())
        # 存储项目的owner信息
        if project.owner is not None and project.owner.login is not None:
            ProjectAllDataFetcher.saveUserInformationToDB(helper, project.owner)
            # user = helper.getInformationForUser(project.owner.login)
            # #             user = SqlServerInterceptor.convertFromBeanbaseToOutput(user)
            #
            # print(user.getValueDict())
            #
            # SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                        , user.getItemKeyList()
            #                                        , user.getValueDict()
            #                                        , user.getIdentifyKeys())

    @staticmethod
    def getPullRequestForRepository(helper, statistic, limit=-1, start=-1):

        if start == -1:
            # 获取项目pull request的数量
            # requestNumber = helper.getTotalPullRequestNumberForProject()
            requestNumber = helper.getMaxSolvedPullRequestNumberForProject()

            print("total pull request number:", requestNumber)

            resNumber = requestNumber
        else:
            resNumber = start
        rr = 0

        # usefulRequestNumber = 0
        # commentNumber = 0
        # usefulReviewNumber = 0  # review的提取数量
        # usefulReviewCommentNumber = 0  # review comment的提取数量
        # usefulIssueCommentNumber = 0  # issue comment 的提取数量
        # usefulCommitNumber = 0  # commit的提取数量
        # usefulCommitCommentNumber = 0  # commit comment的提取数量

        while resNumber > 0:
            print("pull request:", resNumber, " now:", rr)
            ProjectAllDataFetcher.getSinglePullRequestWithExceptionCatch(helper, statistic, resNumber)
            resNumber = resNumber - 1
            rr = rr + 1
            if 0 < limit < rr:
                break

    @staticmethod
    def getPullRequestForRepositoryUseConcurrent(helper, statistic, limit=-1, start=-1):
        if start == -1:
            # 获取项目pull request的数量
            # requestNumber = helper.getTotalPullRequestNumberForProject()
            requestNumber = helper.getMaxSolvedPullRequestNumberForProject()

            print("total pull request number:", requestNumber)

            resNumber = requestNumber
        else:
            resNumber = start

        executor = ThreadPoolExecutor(max_workers=20)
        future_tasks = [executor.submit(ProjectAllDataFetcher.getSinglePullRequestWithExceptionCatch,
                                        helper, statistic,
                                        pull_number) for pull_number in range(resNumber, max(0, resNumber - limit), -1)]
        wait(future_tasks, return_when=ALL_COMPLETED)

    @staticmethod
    def getSinglePullRequestWithExceptionCatch(helper, statistic, pull_number):
        # ProjectAllDataFetcher.getSinglePullRequest(helper, statistic, pull_number)
        print('pull_number:', pull_number)
        exceptionTime = 0
        while exceptionTime < configPraser.getRetryTime():
            try:
                ProjectAllDataFetcher.getSinglePullRequest(helper, statistic, pull_number)
                break
            except Exception as e:
                time.sleep(20)
                exceptionTime += 1
                print(e)

        if exceptionTime == configPraser.getRetryTime():
            raise Exception("error out the limit!")

    @staticmethod
    def getSinglePullRequest(helper, statistic, pull_number):  # 获取某个编号pull request的信息
        pullRequest = helper.getInformationForPullRequest(pull_number)
        if pullRequest is not None:  # pull request存在就储存
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_PULL_REQUEST
                                                   , pullRequest.getItemKeyList()
                                                   , pullRequest.getValueDict()
                                                   , pullRequest.getIdentifyKeys())
            head = pullRequest.head
            if head is not None:
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_BRANCH
                                                       , head.getItemKeyList()
                                                       , head.getValueDict()
                                                       , head.getIdentifyKeys())

            base = pullRequest.base
            if base is not None:
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_BRANCH
                                                       , base.getItemKeyList()
                                                       , base.getValueDict()
                                                       , base.getIdentifyKeys())
            # statistic.usefulRequestNumber += 1

            usefulReviewNumber = 0
            ''' 获取 pull request对应的review信息'''
            reviews = helper.getInformationForReviewWithPullRequest(pullRequest.number)
            for review in reviews:
                if review is not None:
                    ProjectAllDataFetcher.saveReviewInformationToDB(helper, review)
                    # statistic.usefulReviewNumber += 1
                    usefulReviewNumber += 1

            usefulReviewCommentNumber = 0
            '''获取 pull request对应的review comment信息'''
            reviewComments = helper.getInformationForReviewCommentWithPullRequest(pullRequest.number)
            for comment in reviewComments:
                if comment is not None:
                    ProjectAllDataFetcher.saveReviewCommentInformationToDB(helper, comment)
                    # statistic.usefulReviewCommentNumber += 1
                    usefulReviewCommentNumber += 1

            usefulIssueCommentNumber = 0
            '''获取 pull request对应的issue comment信息'''
            issueComments = helper.getInformationForIssueCommentWithIssue(pullRequest.number)
            for comment in issueComments:
                if comment is not None:
                    ProjectAllDataFetcher.saveIssueCommentInformationToDB(helper, comment)
                    # statistic.usefulIssueCommentNumber += 1
                    usefulIssueCommentNumber += 1

            usefulCommitNumber = 0
            usefulCommitCommentNumber = 0
            '''获取 pull request对应的commit信息'''
            commits, relations = helper.getInformationForCommitWithPullRequest(pullRequest.number)
            for commit in commits:
                if commit is not None:
                    commit = helper.getInformationCommit(commit.sha)  # 对status和file信息的补偿
                    ProjectAllDataFetcher.saveCommitInformationToDB(helper, commit)
                    # statistic.usefulCommitNumber += 1
                    usefulCommitNumber += 1

                    '''获取 commit对应的commit comment'''
                    """讲道理commit comment是应该通过遍历项目所有的commit
                       来寻找的  但是现在主要通过pull request为主题提取信息  可以通过这
                       条线的数量来判断是否重要  如果重要后面再做进一步的处理

                       举个例子  rails项目commit 2万+ 遍历实在是浪费资源"""

                    commit_comments = helper.getInformationForCommitCommentsWithCommit(commit.sha)
                    if commit_comments is not None:
                        for commit_comment in commit_comments:
                            ProjectAllDataFetcher.saveCommitCommentInformationToDB(helper, commit_comment)
                            # statistic.usefulCommitCommentNumber += 1
                            usefulCommitCommentNumber += 1

            '''存储 pull request和commit的关系'''
            for relation in relations:
                if relation is not None:
                    ProjectAllDataFetcher.saveCommitPRRelationInformationToDB(helper, relation)

            # 做了同步处理
            statistic.lock.acquire()
            statistic.usefulRequestNumber += 1
            statistic.usefulReviewNumber += usefulReviewNumber
            statistic.usefulReviewCommentNumber += usefulReviewCommentNumber
            statistic.usefulIssueCommentNumber += usefulIssueCommentNumber
            statistic.usefulCommitNumber += usefulCommitNumber
            statistic.usefulCommitCommentNumber = usefulCommitCommentNumber
            print("useful pull request:", statistic.usefulRequestNumber,
                  " useful review:", statistic.usefulReviewNumber,
                  " useful review comment:", statistic.usefulReviewCommentNumber,
                  " useful issue comment:", statistic.usefulIssueCommentNumber,
                  " useful commit:", statistic.usefulCommitNumber)
            statistic.lock.release()

    @staticmethod
    def saveReviewInformationToDB(helper, review):  # review信息录入数据库
        if review is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REVIEW
                                                   , review.getItemKeyList()
                                                   , review.getValueDict()
                                                   , review.getIdentifyKeys())

            # if review.user is not None:
            #     user = helper.getInformationForUser(review.user.login)  # 获取完善的用户信息
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, review.user)

    @staticmethod
    def saveReviewCommentInformationToDB(helper, reviewComment):  # review comment信息录入数据库
        if reviewComment is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REVIEW_COMMENT
                                                   , reviewComment.getItemKeyList()
                                                   , reviewComment.getValueDict()
                                                   , reviewComment.getIdentifyKeys())

            # if reviewComment.user is not None:
            #     user = helper.getInformationForUser(reviewComment.user.login)  # 获取完善的用户信息
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, reviewComment.user)

    @staticmethod
    def saveIssueCommentInformationToDB(helper, issueComment):  # issue comment信息录入数据库
        if issueComment is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_ISSUE_COMMENT
                                                   , issueComment.getItemKeyList()
                                                   , issueComment.getValueDict()
                                                   , issueComment.getIdentifyKeys())

            # if issueComment.user is not None:
            #     user = helper.getInformationForUser(issueComment.user.login)  # 获取完善的用户信息
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, issueComment.user)

    @staticmethod
    def saveCommitInformationToDB(helper, commit):  # commit信息录入数据库
        if commit is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_COMMIT
                                                   , commit.getItemKeyList()
                                                   , commit.getValueDict()
                                                   , commit.getIdentifyKeys())

            # if commit.author is not None:
            #     user = helper.getInformationForUser(commit.author.login)  # 完善作者信息
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, commit.author)

            # if commit.committer is not None:
            #     user = helper.getInformationForUser(commit.committer.login)  # 完善提交者信息
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, commit.committer)

            if commit.files is not None:
                for file in commit.files:
                    SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_FILE
                                                           , file.getItemKeyList()
                                                           , file.getValueDict()
                                                           , file.getIdentifyKeys())
            if commit.parents is not None:
                for parent in commit.parents:
                    SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_COMMIT_RELATION
                                                           , parent.getItemKeyList()
                                                           , parent.getValueDict()
                                                           , parent.getIdentifyKeys())

    @staticmethod
    def saveCommitPRRelationInformationToDB(helper, relation):  # commit信息录入数据库
        if relation is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_COMMIT_PR_RELATION
                                                   , relation.getItemKeyList()
                                                   , relation.getValueDict()
                                                   , relation.getIdentifyKeys())

    @staticmethod
    def saveCommitCommentInformationToDB(helper, comment):  # commit信息录入数据库
        if comment is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_COMMIT_COMMENT
                                                   , comment.getItemKeyList()
                                                   , comment.getValueDict()
                                                   , comment.getIdentifyKeys())

            # if comment.user is not None:
            #     user = helper.getInformationForUser(comment.user.login)
            #     SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
            #                                            , user.getItemKeyList()
            #                                            , user.getValueDict()
            #                                            , user.getIdentifyKeys())
            ProjectAllDataFetcher.saveUserInformationToDB(helper, comment.user)

    @staticmethod
    def saveUserInformationToDB(helper, user):  # user信息录入数据库  先查询数据库再，如果信息不完整再请求
        if user is not None and user.login is not None:
            res = SqlExecuteHelper.queryValuesFromTable(SqlUtils.STR_TABLE_NAME_USER,
                                                        user.getIdentifyKeys(), user.getValueDict())
            if res is None or res.__len__() == 0:
                if configPraser.getPrintMode():
                    print('新用户  从git中获取信息')
                user = helper.getInformationForUser(user.login)
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())
            else:
                if configPraser.getPrintMode():
                    print(type(configPraser.getPrintMode()))
                    print('老用户  不必获取')


if __name__ == '__main__':
    ProjectAllDataFetcher.getAllDataForProject(configPraser.getOwner(), configPraser.getRepo())
    # ProjectAllDataFetcher.getAllDataForProject('ctripcorp', 'apollo')
    # ProjectAllDataFetcher.getAllDataForProject('kytrinyx', 'rails')
    # print(configPraser.getLimit())
