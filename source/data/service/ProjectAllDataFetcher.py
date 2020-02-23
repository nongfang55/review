# coding=gbk
from source.data.service.ApiHelper import ApiHelper
from source.database.SqlExecuteHelper import SqlExecuteHelper
from source.database.SqlUtils import SqlUtils
from source.database.SqlServerInterceptor import SqlServerInterceptor


class ProjectAllDataFetcher:
    """用于获取项目所有信息的类"""

    @staticmethod
    def getAllDataForProject(owner, repo):

        helper = ApiHelper(owner=owner, repo=repo)
        helper.setAuthorization(True)

        '''提取项目的信息以及项目的owner信息'''
        # ProjectAllDataFetcher.getDataForRepository(helper)
        '''提取项目的pull request信息'''
        ProjectAllDataFetcher.getPullRequestForRepository(helper, 5)

    @staticmethod
    def getDataForRepository(helper):

        project = helper.getInformationForProject()
        print(project)
        if project is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REPOS
                                                   , project.getItemKeyList()
                                                   , project.getValueDict()
                                                   , project.getIdentifyKeys())
        # 存储项目的owner信息
        if project.owner is not None and project.owner.login is not None:
            user = helper.getInformationForUser(project.owner.login)
            #             user = SqlServerInterceptor.convertFromBeanbaseToOutput(user)

            print(user.getValueDict())

            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                   , user.getItemKeyList()
                                                   , user.getValueDict()
                                                   , user.getIdentifyKeys())

    @staticmethod
    def getPullRequestForRepository(helper, limit=-1):

        # 获取项目pull request的数量
        # requestNumber = helper.getTotalPullRequestNumberForProject()
        requestNumber = helper.getMaxSolvedPullRequestNumberForProject()

        print("total pull request number:", requestNumber)

        resNumber = requestNumber
        rr = 0

        usefulRequestNumber = 0
        commentNumber = 0
        usefulReviewNumber = 0  # review的提取数量
        usefulReviewCommentNumber = 0  # review comment的提取数量
        usefulIssueCommentNumber = 0  # issue comment 的提取数量
        usefulCommitNumber = 0  # commit的提取数量

        while resNumber > 0:
            print("pull request:", resNumber, " now:", rr)
            pullRequest = helper.getInformationForPullRequest(resNumber)
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
                usefulRequestNumber += 1

                # ''' 获取 pull request对应的review信息'''
                # reviews = helper.getInformationForReviewWithPullRequest(pullRequest.number)
                # for review in reviews:
                #     if review is not None:
                #         ProjectAllDataFetcher.saveReviewInformationToDB(helper, review)
                #         usefulReviewNumber += 1
                #
                # '''获取 pull request对应的review comment信息'''
                # reviewComments = helper.getInformationForReviewCommentWithPullRequest(pullRequest.number)
                # for comment in reviewComments:
                #     if comment is not None:
                #         ProjectAllDataFetcher.saveReviewCommentInformationToDB(helper, comment)
                #         usefulReviewCommentNumber += 1

                # '''获取 pull request对应的issue comment信息'''
                # issueComments = helper.getInformationForIssueCommentWithIssue(pullRequest.number)
                # for comment in issueComments:
                #     if comment is not None:
                #         ProjectAllDataFetcher.saveIssueCommentInformationToDB(helper, comment)
                #         usefulIssueCommentNumber += 1

                '''获取 pull request对应的commit信息'''
                commits, relations = helper.getInformationForCommitWithPullRequest(pullRequest.number)
                for commit in commits:
                    if commit is not None:
                        commit = helper.getInformationCommit(commit.sha)  # 对status和file信息的补偿
                        ProjectAllDataFetcher.saveCommitInformationToDB(helper, commit)
                        usefulCommitNumber += 1

                '''存储 pull request和commit的关系'''
                for relation in relations:
                    if relation is not None:
                        ProjectAllDataFetcher.saveCommitPRRelationInformationToDB(helper, relation)

            resNumber = resNumber - 1
            rr = rr + 1
            if 0 < limit < rr:
                break

        print("useful pull request:", usefulRequestNumber,
              " useful review:", usefulReviewNumber,
              " useful review comment:", usefulReviewCommentNumber,
              " useful issue comment:", usefulIssueCommentNumber,
              " useful commit:", usefulCommitNumber)

    @staticmethod
    def saveReviewInformationToDB(helper, review):  # review信息录入数据库
        if review is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REVIEW
                                                   , review.getItemKeyList()
                                                   , review.getValueDict()
                                                   , review.getIdentifyKeys())

            if review.user is not None:
                user = helper.getInformationForUser(review.user.login)  # 获取完善的用户信息
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())

    @staticmethod
    def saveReviewCommentInformationToDB(helper, reviewComment):  # review comment信息录入数据库
        if reviewComment is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_REVIEW_COMMENT
                                                   , reviewComment.getItemKeyList()
                                                   , reviewComment.getValueDict()
                                                   , reviewComment.getIdentifyKeys())

            if reviewComment.user is not None:
                user = helper.getInformationForUser(reviewComment.user.login)  # 获取完善的用户信息
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())

    @staticmethod
    def saveIssueCommentInformationToDB(helper, issueComment):  # issue comment信息录入数据库
        if issueComment is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_ISSUE_COMMENT
                                                   , issueComment.getItemKeyList()
                                                   , issueComment.getValueDict()
                                                   , issueComment.getIdentifyKeys())

            if issueComment.user is not None:
                user = helper.getInformationForUser(issueComment.user.login)  # 获取完善的用户信息
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())

    @staticmethod
    def saveCommitInformationToDB(helper, commit):  # commit信息录入数据库
        if commit is not None:
            SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_COMMIT
                                                   , commit.getItemKeyList()
                                                   , commit.getValueDict()
                                                   , commit.getIdentifyKeys())

            if commit.author is not None:
                user = helper.getInformationForUser(commit.author.login)  # 完善作者信息
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())

            if commit.committer is not None:
                user = helper.getInformationForUser(commit.committer.login)  # 完善提交者信息
                SqlExecuteHelper.insertValuesIntoTable(SqlUtils.STR_TABLE_NAME_USER
                                                       , user.getItemKeyList()
                                                       , user.getValueDict()
                                                       , user.getIdentifyKeys())
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


if __name__ == '__main__':
    ProjectAllDataFetcher.getAllDataForProject('rails', 'rails')
    # ProjectAllDataFetcher.getAllDataForProject('ctripcorp', 'apollo')
    # ProjectAllDataFetcher.getAllDataForProject('kytrinyx', 'rails')
