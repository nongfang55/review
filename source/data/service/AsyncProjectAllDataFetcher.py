# coding=gbk
import asyncio
import json
import os
import time
from datetime import datetime
import random
import numpy as np
import math

from pandas import DataFrame

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.data.bean.ReviewChangeRelation import ReviewChangeRelation
from source.data.service.ApiHelper import ApiHelper
from source.data.service.AsyncApiHelper import AsyncApiHelper
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.data.service.PRTimeLineUtils import PRTimeLineUtils
from source.database.AsyncSqlExecuteHelper import getMysqlObj
from source.database.SqlUtils import SqlUtils
from source.utils.Logger import Logger
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper
from source.utils.statisticsHelper import statisticsHelper


class AsyncProjectAllDataFetcher:
    # 获取项目的所有信息 主题信息采用异步获取

    @staticmethod
    def getPullRequestNodes(repo_full_name):
        loop = asyncio.get_event_loop()
        coro = AsyncProjectAllDataFetcher.fetchPullRequestNodes(loop, repo_full_name)
        task = loop.create_task(coro)
        loop.run_until_complete(task)
        return task.result()

    @staticmethod
    async def fetchPullRequestNodes(loop, repo_full_name):
        mysql = await getMysqlObj(loop)
        print("mysql init success")

        sql = SqlUtils.STR_SQL_QUERY_PR_FOR_TIME_LINE
        results = await AsyncSqlHelper.query(mysql, sql, [repo_full_name])

        return results

    @staticmethod
    def getPullRequestTimeLine(owner, repo, nodes):
        # 获取多个个pull request的时间线上面的信息 并对上面的comment做拼接
        AsyncApiHelper.setRepo(owner, repo)
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        loop = asyncio.get_event_loop()
        coro = getMysqlObj(loop)
        task = loop.create_task(coro)
        loop.run_until_complete(task)
        mysql = task.result()

        # nodes = np.array(nodes).reshape(10, -1)
        nodesGroup = []
        if nodes.__len__() % 10 == 0:
            nodesGroup = np.array(nodes).reshape(10, -1)
        else:
            offset = 10 - nodes.__len__() % 10
            nodes.extend([None for i in range(0, offset)])
            nodesGroup = np.array(nodes).reshape(20, -1)
            # for index in range(0, nodes.__len__(), 10):
            #     if index + 10 < nodes.__len__():
            #         nodesGroup.append(nodes[index:index + 10])
            #     else:
            #         nodesGroup.append(nodes[index:nodes.__len__()])

        tasks = [
            asyncio.ensure_future(AsyncApiHelper.downloadRPTimeLine([x for x in nodegroup.tolist() if x is not None],
                                                                    semaphore, mysql, statistic))
            for nodegroup in nodesGroup]  # 可以通过nodes 过多次嵌套节省请求数量

        tasks = asyncio.gather(*tasks)
        loop.run_until_complete(tasks)
        print('cost time:', datetime.now() - t1)
        return tasks.result()

    @staticmethod
    async def fetchPullRequestTimelineNodesFromDB(mysql, pr_nodes):
        sql = "select * from PRTimeLine " \
              "where pullrequest_node in %s"
        results = await AsyncSqlHelper.query(mysql, sql, [pr_nodes])
        return results

    @staticmethod
    def analyzePullRequestReview(pr_timeline_item_groups, pr_author_map):
        """一次解析多个pr的change_trigger信息"""
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        coro = getMysqlObj(loop)
        task = loop.create_task(coro)
        loop.run_until_complete(task)
        mysql = task.result()

        loop = asyncio.get_event_loop()
        tasks = [asyncio.ensure_future(
            AsyncProjectAllDataFetcher.analyzePRTimeline(mysql, pr_timeline_item_group, statistic, pr_author_map))
            for pr_timeline_item_group in pr_timeline_item_groups]
        tasks = asyncio.gather(*tasks)
        loop.run_until_complete(tasks)
        print('cost time:', datetime.now() - t1)
        return tasks.result()

    @staticmethod
    async def analyzePRTimeline(mysql, prOriginTimeLineItems, statistic, pr_author_map):
        """分析pr时间线，找出触发过change_trigger的comment
           返回五元组（reviewer, comment_node, comment_type, file, change_trigger）
           对于issue_comment，无需返回file，默认会trigger该pr的所有文件
        """

        """将原始数据转化为prTimeLineItem列表"""
        if prOriginTimeLineItems is None:
            return None
        pr_node_id = None
        prTimeLineItems = []

        for item in prOriginTimeLineItems:
            """从某一条item里获取pr_node_id"""
            if pr_node_id is None:
                pr_node_id = item.get(StringKeyUtils.STR_KEY_PULL_REQUEST_NODE)
            origin = item.get(StringKeyUtils.STR_KEY_ORIGIN)
            prTimeLineRelation = PRTimeLineRelation.Parser.parser(origin)
            if prTimeLineRelation is None:
                continue
            prTimeLineRelation.position = item.get(StringKeyUtils.STR_KEY_POSITION, None)
            prTimeLineRelation.pull_request_node = item.get(StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, None)
            prTimeLineItems.append(prTimeLineRelation)

        """将timeline items逆序"""
        prTimeLineItems.reverse()

        """获取作者"""
        pr_author = pr_author_map[pr_node_id]

        """解析pr时间线，找出和changes相关联的comment"""
        changeTriggerComments = []
        ReviewChangeRelations = []
        pairs = PRTimeLineUtils.splitTimeLine(prTimeLineItems)
        for pair in pairs:
            changes = pair[0]
            reviews = pair[1]

            # 数据存储的暂时不做
            # reviewChangeRelationList = ReviewChangeRelation.parserV4.parser(pair)
            # ReviewChangeRelations.extend(reviewChangeRelationList)
            is_in_gap = False
            """先对这个pair是否在gap做判断"""
            for change in changes:
                if change.typename == StringKeyUtils.STR_KEY_REOPENED_EVENT:
                    is_in_gap = True

            """注意：现在的hasUsefulComment是issue commenth和 review_no_comment共享
                2020.10.31 @张逸凡
            """

            # 改成检测到的第几条issue comment或者 review with no comment,中间不算作者的评论，从0开始
            # 由于子序列是时间倒序的，数值增长即可
            UsefulIssueCommentPos = 0
            for review in reviews:
                """对于出现在pair中的第一条issue comment, 直接认为它是有效的
                   如果都认为有效,issue_comment就全都有效了
                """

                if review.typename == StringKeyUtils.STR_KEY_ISSUE_COMMENT:
                    """对issue comment进行处理"""
                    UsefulIssueCommentPos = AsyncProjectAllDataFetcher.analyzeIssueComment(review, changes, pr_author,
                                                                                           pr_node_id,
                                                                                           changeTriggerComments,
                                                                                           UsefulIssueCommentPos,
                                                                                           is_in_gap)
                else:
                    """对review comment进行处理"""
                    """对于 review comment 先获取 review对应的comment 数量  2020.10.30"""
                    comments = await AsyncApiHelper.getReviewCommentsByNodeFromStore(review.timeline_item_node, mysql)

                    if comments is None or comments.__len__() == 0:
                        """对于没有comment的情况，认为种类是review_with_on_comment"""
                        """对issue comment进行处理"""
                        UsefulIssueCommentPos = AsyncProjectAllDataFetcher.analyzeReviewWithNoComment(review, changes,
                                                                                                      pr_author,
                                                                                                      pr_node_id,
                                                                                                      changeTriggerComments,
                                                                                                      UsefulIssueCommentPos,
                                                                                                      is_in_gap)
                    else:
                        """这种就是正常的review comment的情况"""
                        change_trigger_review_comments = await AsyncApiHelper.analyzeReviewChangeTriggerByBlob(
                            pr_node_id,
                            changes, review,
                            mysql, statistic,
                            comments, pr_author,
                            is_in_gap)
                        if change_trigger_review_comments is not None:
                            changeTriggerComments.extend(change_trigger_review_comments)
                        # if reviewChangeRelationList.__len__() > 0:
                        #     ReviewChangeRelations.extend(reviewChangeRelationList)
        await AsyncSqlHelper.updateBeanDateList(ReviewChangeRelations, mysql)

        """计数"""
        statistic.lock.acquire()
        statistic.usefulRequestNumber += 1
        print("analyzed pull request:", statistic.usefulRequestNumber,
              " cost time:", datetime.now() - statistic.startTime)
        statistic.lock.release()

        return changeTriggerComments

    @staticmethod
    def analyzeIssueComment(review, changes, pr_author, pr_node_id, changeTriggerComments,
                            UsefulIssueCommentPos, is_in_gap):
        """把部分判断issue的逻辑放在这里，不然主函数有点拥挤了
           返回值为最新的 IssueCommentPos

           is_in_gap 用于判断是否在closed和 reopened的间隙

        """
        change_trigger_issue_comment = {
            "pullrequest_node": pr_node_id,
            "user_login": review.user_login,
            "comment_node": review.timeline_item_node,
            "comment_type": StringKeyUtils.STR_LABEL_ISSUE_COMMENT,
            "change_trigger": None,
            "filepath": None
        }

        """识别作者的情况"""
        if review.user_login == pr_author:
            change_trigger_issue_comment['change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_ISSUE_COMMENT_AUTHOR
            changeTriggerComments.append(change_trigger_issue_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """识别间隙的情况"""
        if is_in_gap:
            change_trigger_issue_comment[
                'change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_ISSUE_COMMENT_BETWEEN_REOPEN
            changeTriggerComments.append(change_trigger_issue_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """识别正常pr流程外面的情况
           问题点： changes = 0 是否说明是在pr流程外面的comment？
        """
        if changes.__len__() == 0:
            change_trigger_issue_comment['change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_ISSUE_COMMENT_OUT_PR
            changeTriggerComments.append(change_trigger_issue_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """除去上面的情况，就是正常的issue comment"""
        # if hasUsefulIssueComment:
        #     change_trigger_issue_comment['change_trigger'] = -1
        # else:
        #     change_trigger_issue_comment['change_trigger'] = 0
        #     hasUsefulIssueComment = True
        change_trigger_issue_comment['change_trigger'] = - UsefulIssueCommentPos  # 注意负号
        UsefulIssueCommentPos += 1

        changeTriggerComments.append(change_trigger_issue_comment)
        return UsefulIssueCommentPos

    @staticmethod
    def analyzeReviewWithNoComment(review, changes, pr_author, pr_node_id, changeTriggerComments,
                                   UsefulIssueCommentPos, is_in_gap):
        """把部分判断的逻辑放在这里，不然主函数有点拥挤了
           返回值为最新的 UsefulIssueCommentPos

           is_in_gap 用于判断是否在closed和 reopened的间隙

           对于这一类的review with no comment，把review的node_id作为标识
        """
        change_trigger_review_with_no_comment = {
            "pullrequest_node": pr_node_id,
            "user_login": review.user_login,
            "comment_node": review.timeline_item_node,
            "comment_type": StringKeyUtils.STR_LABEL_REVIEW_WITH_NO_COMMENT,
            "change_trigger": None,
            "filepath": None
        }

        """识别作者的情况"""
        if review.user_login == pr_author:
            change_trigger_review_with_no_comment[
                'change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_NO_COMMENT_AUTHOR
            changeTriggerComments.append(change_trigger_review_with_no_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """识别间隙的情况"""
        if is_in_gap:
            change_trigger_review_with_no_comment[
                'change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_NO_COMMENT_BETWEEN_REOPEN
            changeTriggerComments.append(change_trigger_review_with_no_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """识别正常pr流程外面的情况
           问题点： changes = 0 是否说明是在pr流程外面的comment？
        """
        if changes.__len__() == 0:
            change_trigger_review_with_no_comment[
                'change_trigger'] = StringKeyUtils.STR_CHANGE_TRIGGER_REVIEW_NO_COMMENT_OUT_PR
            changeTriggerComments.append(change_trigger_review_with_no_comment)
            """原样返回"""
            return UsefulIssueCommentPos

        """除去上面的情况，就是正常的issue comment"""
        # if hasUsefulIssueComment:
        #     change_trigger_review_with_no_comment['change_trigger'] = -1
        # else:
        #     change_trigger_review_with_no_comment['change_trigger'] = 0
        #     hasUsefulIssueComment = True
        """2020.11.5 对于review not comment也用类似的手段处理，直接认为无效"""
        # change_trigger_review_with_no_comment['change_trigger'] = - UsefulIssueCommentPos  # 注意负号
        # UsefulIssueCommentPos += 1
        change_trigger_review_with_no_comment['change_trigger'] = -1

        changeTriggerComments.append(change_trigger_review_with_no_comment)
        return UsefulIssueCommentPos

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
        if configPraser.getApiVersion() == StringKeyUtils.API_VERSION_RESET:
            tasks = [asyncio.ensure_future(AsyncApiHelper.downloadInformation(pull_number, semaphore, mysql, statistic))
                     for pull_number in range(start, max(start - limit, 0), -1)]
        elif configPraser.getApiVersion() == StringKeyUtils.API_VERSION_GRAPHQL:
            tasks = [
                asyncio.ensure_future(AsyncApiHelper.downloadInformationByV4(pull_number, semaphore, mysql, statistic))
                for pull_number in range(start, max(start - limit, 0), -1)]
        await asyncio.wait(tasks)

    @staticmethod
    def getUserFollowList(userList):
        # 获取 给定的用户列表中的用户的 follow 列表
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcessUserFollowList(loop, statistic, userList))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print('cost time:', datetime.now() - t1)

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
    def getUnmatchedCommitFile():
        # 获取 数据库中没有获得file的 commit点，一次最多2000个
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcessUnmatchCommitFile(loop, statistic))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print('cost time:', datetime.now() - t1)

    @staticmethod
    def getNoOriginLineReviewComment(owner, repo, min_num, max_num):
        # 获取 数据库中没有获得review comment，一次最多2000个
        t1 = datetime.now()

        statistic = statisticsHelper()
        statistic.startTime = t1

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcessNoOriginLineReviewComment(loop, statistic,
                                                                                                     owner, repo,
                                                                                                     min_num, max_num))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        print('cost time:', datetime.now() - t1)

    @staticmethod
    async def preProcessNoOriginLineReviewComment(loop, statistic, owner, repo, min_num, max_num):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")
        print("mysql init success")

        repoName = owner + '/' + repo
        values = [repoName, repoName, min_num, max_num]
        total = await AsyncSqlHelper.query(mysql, SqlUtils.STR_SQL_QUERY_NO_ORIGINAL_LINE_REVIEW_COMMENT_COUNT,
                                           values)
        fetch_loop = math.ceil(total[0][0] / 2000)
        for i in range(0, fetch_loop):
            res = await AsyncSqlHelper.query(mysql, SqlUtils.STR_SQL_QUERY_NO_ORIGINAL_LINE_REVIEW_COMMENT
                                             , values)
            print("fetched size:", res.__len__())

            tasks = [asyncio.ensure_future(
                AsyncApiHelper.downloadSingleReviewComment(repoName, item[0], semaphore, mysql, statistic))
                for item in res]  # 可以通过nodes 过多次嵌套节省请求数量
            await asyncio.wait(tasks)

    @staticmethod
    async def preProcessUnmatchCommitFile(loop, statistic):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")
        print("mysql init success")
        fetch_size = 2000
        total = await AsyncSqlHelper.query(mysql, SqlUtils.STR_SQL_QUERY_UNMATCH_COMMIT_FILE_COUNT_BY_HAS_FETCHED_FILE,
                                           None)
        fetch_loop = int(total[0][0] / fetch_size)
        for i in range(0, fetch_loop):
            start = random.randint(0, fetch_loop - 1)
            res = await AsyncSqlHelper.query(mysql, SqlUtils.STR_SQL_QUERY_UNMATCH_COMMIT_FILE_BY_HAS_FETCHED_FILE,
                                             [start * fetch_size])
            print(res)

            tasks = [
                asyncio.ensure_future(AsyncApiHelper.downloadCommits(item[0], item[1], semaphore, mysql, statistic))
                for item in res]  # 可以通过nodes 过多次嵌套节省请求数量
            await asyncio.wait(tasks)

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

    @staticmethod
    async def preProcessUserFollowList(loop, statistic, userList):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")

        tasks = [asyncio.ensure_future(AsyncApiHelper.downloadUserFollowList(login, semaphore, mysql, statistic))
                 for login in userList]  # 可以通过nodes 过多次嵌套节省请求数量
        await asyncio.wait(tasks)

    @staticmethod
    def getPRTimeLine(owner, repo):
        """1. 获取该仓库所有的pr_node"""
        repo_fullname = owner + "/" + repo
        pr_nodes = AsyncProjectAllDataFetcher.getPullRequestNodes(repo_fullname)
        pr_nodes = list(pr_nodes)
        pr_nodes = [node[0] for node in pr_nodes]
        # 起始位置
        pos = 4000
        # 每次获取的数量限制
        """加速"""
        fetchLimit = 400
        size = pr_nodes.__len__()
        Logger.logi("--------------begin to fetch {0} {1}--------------".format(owner, repo))
        print("start fetch")
        """2. 分割获取pr_timeline"""
        while pos < size:
            print("total:", size, "  now:", pos)
            loop_begin_time = datetime.now()
            Logger.logi("start: {0}, end: {1}, all: {2}".format(pos, pos + fetchLimit, size))
            pr_sub_nodes = pr_nodes[pos:pos + fetchLimit]
            results = AsyncProjectAllDataFetcher.getPullRequestTimeLine(owner=owner,
                                                                        repo=repo, nodes=pr_sub_nodes)
            if results is None:
                Logger.loge("start: {0}, end: {1} meet error".format(pos, pos + fetchLimit))
                pos += fetchLimit
                continue
            for result in results:
                pr_timelines = result
                if pr_timelines is None:
                    continue
                Logger.logi("fetched, cost time: {1}".format(pr_timelines.__len__(), datetime.now() - loop_begin_time))
            pos += fetchLimit
        Logger.logi("--------------end---------------")

    @staticmethod
    def checkPRTimeLineResult(owner, repo, limit=5):
        """检查PRTimeline数据是否完整爬取"""
        """1. 获取该仓库所有的pr_node"""
        repo_fullname = owner + "/" + repo
        pr_nodes = AsyncProjectAllDataFetcher.getPullRequestNodes(repo_fullname)
        pr_nodes = list(pr_nodes)
        pr_nodes = [node[0] for node in pr_nodes]
        """2. 读取prtimeline文件，对比pr"""
        target_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        df = pandasHelper.readTSVFile(fileName=target_filename, header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        """3. 获取需要fetch的PR"""
        fetched_prs = list(df['pullrequest_node'])
        need_fetch_prs = list(set(pr_nodes).difference(set(fetched_prs)))
        Logger.logi("there are {0} pr_timeline need to fetch".format(need_fetch_prs.__len__()))

        """设置fetch参数"""
        pos = 0
        fetchLimit = 200
        size = need_fetch_prs.__len__()
        while pos < size:
            sub_need_fetch_prs = need_fetch_prs[pos:pos + fetchLimit]
            Logger.logi("start: {0}, end: {1}, all: {2}".format(pos, pos + fetchLimit, size))
            """4. 开始爬取"""
            results = AsyncProjectAllDataFetcher.getPullRequestTimeLine(owner=owner,
                                                                        repo=repo, nodes=sub_need_fetch_prs)
            Logger.logi("successfully fetched {0} pr! ".format(pos + fetchLimit))
            pos += fetchLimit

    @staticmethod
    def checkChangeTriggerResult(owner, repo):
        """检查PRChangeTrigger是否计算完整"""
        """在切换代理的时候，数据库连接会断开，导致comments信息查不到，会遗漏review comment的情况"""
        """这里检查一遍pr的change_trigger里是否有review_comment数据，如果没有，重新获取一次"""

        """PRTimeLine表头"""
        PR_CHANGE_TRIGGER_COLUMNS = ["pullrequest_node", "user_login", "comment_node",
                                     "comment_type", "change_trigger", "filepath"]
        """初始化目标文件"""
        target_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{configPraser.getRepo()}_data_pr_change_trigger.tsv'

        """1. 获取该仓库所有的pr_node"""
        # repo_fullname = configPraser.getOwner() + "/" + configPraser.getRepo()
        # pr_nodes = AsyncProjectAllDataFetcher.getPullRequestNodes(repo_fullname)
        # pr_nodes = list(pr_nodes)
        # pr_nodes = [node[0] for node in pr_nodes]
        """需要获取的prs改为有issue 额 review的timeline的pr"""
        timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        timeline_df = pandasHelper.readTSVFile(fileName=timeline_filename, header=0)
        timeline_df = timeline_df.loc[(timeline_df['typename'] == 'IssueComment') \
                                      | (timeline_df['typename'] == 'PullRequestReview')].copy(deep=True)
        pr_nodes = list(set(timeline_df['pullrequest_node']))

        """2. 读取pr_change_trigger文件"""
        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)
        change_nodes = list(set(change_trigger_df['pullrequest_node']))

        # """3. 读取pr_timeline文件"""
        # timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{configPraser.getRepo()}_data_prtimeline.tsv'
        # timeline_df = pandasHelper.readTSVFile(fileName=timeline_filename, header=0)

        """4. 将change_trigger按照pull_request_node分组"""
        grouped_timeline = change_trigger_df.groupby((['pullrequest_node']))
        """5. 分析pullrequest_node的change_trigger信息是否完整，整理出需要重新获取的pr信息"""
        re_analyze_prs = [x for x in pr_nodes if x not in change_nodes]
        # for pr, group in grouped_timeline:
        #     if pr not in pr_nodes:
        #         re_analyze_prs.append(pr)
        #     else:
        #         review_comment_trigger = group.loc[(group['comment_type'] == StringKeyUtils.STR_LABEL_REVIEW_COMMENT) & (group['change_trigger'] >= 0)]
        #         if review_comment_trigger is None or review_comment_trigger.empty:
        #             re_analyze_prs.append(pr)
        # Logger.logi("there are {0} prs need to re analyze".format(re_analyze_prs.__len__()))

        """设置fetch参数"""
        pos = 0
        fetchLimit = 200
        size = re_analyze_prs.__len__()
        while pos < size:
            Logger.logi("start: {0}, end: {1}, all: {2}".format(pos, pos + fetchLimit, size))
            sub_re_analyze_prs = re_analyze_prs[pos:pos + fetchLimit]
            """6. 重新获取这些pr的timeline"""
            re_analyze_prs_timeline_df = timeline_df[timeline_df['pullrequest_node'].isin(sub_re_analyze_prs)]
            grouped_timeline = re_analyze_prs_timeline_df.groupby((['pullrequest_node']))
            formated_data = []
            for pr, group in grouped_timeline:
                formated_data.append(group.to_dict(orient='records'))

            """7. 开始分析"""
            pr_change_trigger_comments = AsyncProjectAllDataFetcher.analyzePullRequestReview(formated_data)
            pr_change_trigger_comments = [x for y in pr_change_trigger_comments for x in y]

            """8. 将分析结果去重并追加到change_trigger表中"""
            if pr_change_trigger_comments is not None and pr_change_trigger_comments.__len__() > 0:
                target_content = DataFrame()
                target_content = target_content.append(pr_change_trigger_comments, ignore_index=True)
                target_content = target_content[PR_CHANGE_TRIGGER_COLUMNS].copy(deep=True)
                target_content.drop_duplicates(subset=['pullrequest_node', 'comment_node'], inplace=True, keep='first')
                if not target_content.empty:
                    pandasHelper.writeTSVFile(target_filename, target_content, pandasHelper.STR_WRITE_STYLE_APPEND_NEW,
                                              header=pandasHelper.INT_WRITE_WITHOUT_HEADER)
            Logger.logi("successfully analyzed {0} prs".format(re_analyze_prs.__len__()))
            pos += fetchLimit

    @staticmethod
    def testChangeTriggerAnalyzer(owner, repo, pull_request_node):
        AsyncApiHelper.setRepo(owner, repo)

        """读取PRTimeline，获取需要分析change_trigger的pr列表"""
        pr_timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        pr_timeline_df = pandasHelper.readTSVFile(fileName=pr_timeline_filename,
                                                  header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        pr_nodes = list(set(list(pr_timeline_df['pullrequest_node'])))
        pr_nodes.sort()

        """按照爬取限制取子集"""
        pr_timeline_items = pr_timeline_df[pr_timeline_df['pullrequest_node'] == pull_request_node]
        """对子集按照pull_request_node分组"""
        grouped_timeline = pr_timeline_items.groupby((['pullrequest_node']))
        """将分组结果保存为字典{pr->pr_timeline_items}"""
        formated_data = []
        for pr, group in grouped_timeline:
            record = group.to_dict(orient='records')
            record = sorted(record, key=lambda x: int(x.get(StringKeyUtils.STR_KEY_POSITION)))
            formated_data.append(record)

        """分析这些pr的timeline"""
        pr_change_trigger_comments = AsyncProjectAllDataFetcher.analyzePullRequestReview(formated_data)
        print("finish!")

    @staticmethod
    def getPRChangeTriggerData(owner, repo):
        """ 根据
            ALL_{repo}_data_prtimeline.tsv
            获取pr change_trigger数据
        """
        AsyncApiHelper.setRepo(owner, repo)
        """PRTimeLine表头"""
        PR_CHANGE_TRIGGER_COLUMNS = ["pullrequest_node", "user_login", "comment_node",
                                     "comment_type", "change_trigger", "filepath"]
        """初始化目标文件"""
        target_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        target_content = DataFrame(columns=PR_CHANGE_TRIGGER_COLUMNS)
        # pandasHelper.writeTSVFile(target_filename, target_content, pandasHelper.STR_WRITE_STYLE_APPEND_NEW,
        #                           header=pandasHelper.INT_WRITE_WITH_HEADER)

        """读取PRTimeline，获取需要分析change_trigger的pr列表"""
        pr_timeline_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_prtimeline.tsv'
        pr_timeline_df = pandasHelper.readTSVFile(fileName=pr_timeline_filename,
                                                  header=pandasHelper.INT_READ_FILE_WITH_HEAD)

        """读取PullRequestData，获取pr所对应的作者"""
        pr_data_filename = projectConfig.getPullRequestPath() + os.sep + f'ALL_{repo}_data_pullrequest.tsv'
        pr_data_df = pandasHelper.readTSVFile(fileName=pr_data_filename, header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        """收集pr已经对应的作者  用于后面过滤属于作者评论"""
        pr_author_map = {}
        for index, row in pr_data_df.iterrows():
            pr_author_map[row['node_id']] = row['user_login']

        pr_nodes = list(set(list(pr_timeline_df['pullrequest_node'])))
        pr_nodes.sort()
        # pr_nodes = ['MDExOlB1bGxSZXF1ZXN0MjE5MjEzOTc5']  # 3次reopend
        # pr_nodes = ['MDExOlB1bGxSZXF1ZXN0MjA0MTk5ODkw']
        # pr_nodes = ['MDExOlB1bGxSZXF1ZXN0NDQwOTAxMzk0']
        # pr_nodes = ['MDExOlB1bGxSZXF1ZXN0MzE1OTU0NDgw']  # pr外review
        pr_nodes = ['MDExOlB1bGxSZXF1ZXN0MTQ3NDczNTIx']  # 普通用例
        # pr_nodes = ['MDExOlB1bGxSZXF1ZXN0NDM4NjAzMjk2']  # 超多review

        """设置fetch参数"""
        pos = 0
        fetchLimit = 400
        size = pr_nodes.__len__()
        Logger.logi("there are {0} prs need to analyze".format(pr_nodes.__len__()))
        t1 = datetime.now()

        while pos < size:
            print("now:", pos, ' total:', size, 'cost time:', datetime.now() - t1)
            Logger.logi("start: {0}, end: {1}, all: {2}".format(pos, pos + fetchLimit, size))

            """按照爬取限制取子集"""
            sub_prs = pr_nodes[pos:pos + fetchLimit]
            pr_timeline_items = pr_timeline_df[pr_timeline_df['pullrequest_node'].isin(sub_prs)]
            """对子集按照pull_request_node分组"""
            grouped_timeline = pr_timeline_items.groupby((['pullrequest_node']))
            """将分组结果保存为字典{pr->pr_timeline_items}"""
            formated_data = []
            for pr, group in grouped_timeline:
                record = group.to_dict(orient='records')
                record = sorted(record, key=lambda x: int(x.get(StringKeyUtils.STR_KEY_POSITION)))
                formated_data.append(record)

            """分析这些pr的timeline"""
            pr_change_trigger_comments = AsyncProjectAllDataFetcher.analyzePullRequestReview(formated_data,
                                                                                             pr_author_map)
            pr_change_trigger_comments = [x for y in pr_change_trigger_comments for x in y]

            """将分析结果去重并追加到change_trigger表中"""
            if pr_change_trigger_comments.__len__() > 0:
                target_content = DataFrame()
                target_content = target_content.append(pr_change_trigger_comments, ignore_index=True)
                target_content = target_content[PR_CHANGE_TRIGGER_COLUMNS].copy(deep=True)
                target_content.drop_duplicates(subset=['pullrequest_node', 'comment_node'], inplace=True, keep='first')
                if not target_content.empty:
                    pandasHelper.writeTSVFile(target_filename, target_content, pandasHelper.STR_WRITE_STYLE_APPEND_NEW,
                                              header=pandasHelper.INT_WRITE_WITHOUT_HEADER)
                Logger.logi("successfully analyzed {0} prs".format(pos))
            pos += fetchLimit


if __name__ == '__main__':
    # """1. 获取基础数据"""
    # # 格式说明: owner, repo, 需要爬取的pr数量, pr的结束编号
    # # eg. 数据表上pr序号是14000-17800
    # # 这里应该填 opencv, opencv, 3800(17800减14000), 17800
    # projects = [("opencv", "opencv", 3800, 17800),
    #             ("facebook", "react", 4800, 19300)]
    # for p in projects:
    #     AsyncProjectAllDataFetcher.getDataForRepository(p[0], p[1], p[2], p[3])
    #
    # """2. 获取reviewer comment original_line数据"""
    # # 格式说明: owner, repo, pr的开始编号, pr的结束编号
    # projects = [("opencv", "opencv", 14000, 17800),
    #             ("facebook", "react", 14500, 19300)]
    # for p in projects:
    #     AsyncProjectAllDataFetcher.getNoOriginLineReviewComment(p[0], p[1], p[2], p[3])
    #
    # """3. 获取pr时间线信息"""
    # # 格式说明：owner, repo
    # projects = [("opencv", "opencv")]
    # for project in projects:
    #     AsyncProjectAllDataFetcher.getPRTimeLine(project[0], project[1])

    # """4. 获取commit_file"""
    # AsyncProjectAllDataFetcher.getUnmatchedCommitFile()

    """5. 获取change_trigger"""
    # TODO 注意在爬取之前需要先将PRTimeLine数据保存到本地，见文档
    # 格式说明：owner, repo
    projects = [("opencv", "opencv")]
    for project in projects:
        AsyncProjectAllDataFetcher.getPRChangeTriggerData(project[0], project[1])

    # """5. 获取commit_file"""
    # AsyncProjectAllDataFetcher.getUnmatchedCommitFile()
