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

        while resNumber > 0:
            print("pull request:", resNumber, " now:", rr)
            pullRequest = helper.getInformationForPullRequest(resNumber)
            if pullRequest is not None: # pull request存在就储存
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

            resNumber = resNumber - 1
            rr = rr + 1
            if 0 < limit < rr:
                break

        print("useful pull request:", usefulRequestNumber, "  total comment:", commentNumber)


if __name__ == '__main__':
    ProjectAllDataFetcher.getAllDataForProject('rails', 'rails')
    # ProjectAllDataFetcher.getAllDataForProject('ctripcorp', 'apollo')
    # ProjectAllDataFetcher.getAllDataForProject('kytrinyx', 'rails')


