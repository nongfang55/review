# coding=gbk
import asyncio
import time
from datetime import datetime

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig
from source.data.service.ApiHelper import ApiHelper
from source.data.service.AsyncApiHelper import AsyncApiHelper
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.database.AsyncSqlExecuteHelper import getMysqlObj
from source.database.SqlUtils import SqlUtils
from source.utils.pandas.pandasHelper import pandasHelper
from source.utils.statisticsHelper import statisticsHelper


class AsyncProjectAllDataFetcher:
    # 获取项目的所有信息 主题信息采用异步获取

    @staticmethod
    def getPullRequestTimeLine(owner, repo, nodes):
        # 获取多个个pull request的时间线上面的信息 并对上面的comment做拼接
        AsyncApiHelper.setRepo(owner, repo)
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcessTimeLine(loop, nodes, statistic))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print('cost time:', datetime.now() - t1)

    @staticmethod
    async def preProcessTimeLine(loop, node, statistic):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")

        tasks = [asyncio.ensure_future(AsyncApiHelper.downloadRPTimeLine(nodeIds, semaphore, mysql, statistic))
                 for nodeIds in node]  # 可以通过nodes 过多次嵌套节省请求数量
        await asyncio.wait(tasks)

    @staticmethod
    def getDataForRepository(owner, repo, limit=-1, start=-1):
        """指定目标owner/repo 获取start到  start - limit编号的pull-request相关评审信息"""

        """设定start 和 limit"""
        if start == -1:
            # 获取项目pull request的数量 这里使用同步方法获取
            requestNumber = ApiHelper(owner, repo).getMaxSolvedPullRequestNumberForProject()
            print("total pull request number:", requestNumber)

            startNumber = requestNumber
        else:
            startNumber = start

        if limit == -1:
            limit = startNumber

        """获取repo信息"""
        AsyncApiHelper.setRepo(owner, repo)
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        """异步多协程爬虫爬取pull-request信息"""
        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcess(loop, limit, start, statistic))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print("useful pull request:", statistic.usefulRequestNumber,
              " useful review:", statistic.usefulReviewNumber,
              " useful review comment:", statistic.usefulReviewCommentNumber,
              " useful issue comment:", statistic.usefulIssueCommentNumber,
              " useful commit:", statistic.usefulCommitNumber,
              " cost time:", datetime.now() - statistic.startTime)

    @staticmethod
    async def preProcess(loop, limit, start, statistic):
        """准备工作"""
        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        """初始化数据库"""
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")

        """多协程"""
        tasks = [asyncio.ensure_future(AsyncApiHelper.downloadInformation(pull_number, semaphore, mysql, statistic))
                 for pull_number in range(start, max(start - limit, 0), -1)]
        await asyncio.wait(tasks)

    @staticmethod
    def getUnmatchedCommits():
        # 获取 数据库中没有获得的commit点，一次最多2000个
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcessUnmatchCommits(loop, statistic))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print('cost time:', datetime.now() - t1)

    @staticmethod
    async def preProcessUnmatchCommits(loop, statistic):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")

        res = await AsyncSqlHelper.query(mysql, SqlUtils.STR_SQL_QUERY_UNMATCH_COMMITS, None)
        print(res)

        tasks = [asyncio.ensure_future(AsyncApiHelper.downloadCommits(item[0], item[1], semaphore, mysql, statistic))
                 for item in res]  # 可以通过nodes 过多次嵌套节省请求数量
        await asyncio.wait(tasks)


if __name__ == '__main__':
    # AsyncProjectAllDataFetcher.getDataForRepository(owner=configPraser.getOwner(), repo=configPraser.getRepo()
    #                                                 , start=configPraser.getStart(), limit=configPraser.getLimit())

    # data = pandasHelper.readTSVFile(projectConfig.getChangeTriggerPRPath(), pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
    # print(data.as_matrix().shape)
    # node_to = configPraser.getStart()
    # node_from = max(configPraser.getStart() - configPraser.getLimit(), 0)
    # pr_nodes = data.as_matrix()[node_from:node_to, 3]
    # print(pr_nodes.__len__())
    #
    # AsyncProjectAllDataFetcher.getPullRequestTimeLine(owner=configPraser.getOwner(), repo=configPraser.getRepo(),
    #                                                   nodes=[[x] for x in pr_nodes])

    AsyncProjectAllDataFetcher.getUnmatchedCommits()