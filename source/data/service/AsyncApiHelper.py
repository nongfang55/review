# coding=gbk
import asyncio
import json
import random
import time
import traceback
from datetime import datetime

import aiohttp

from source.config.configPraser import configPraser
from source.data.bean.Comment import Comment
from source.data.bean.CommentRelation import CommitRelation
from source.data.bean.Commit import Commit
from source.data.bean.CommitPRRelation import CommitPRRelation
from source.data.bean.File import File
from source.data.bean.IssueComment import IssueComment
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Review import Review
from source.data.bean.ReviewComment import ReviewComment
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.data.service.BeanParserHelper import BeanParserHelper
from source.data.service.GraphqlHelper import GraphqlHelper
from source.data.service.PRTimeLineUtils import PRTimeLineUtils
from source.data.service.ProxyHelper import ProxyHelper
from source.data.service.TextCompareUtils import TextCompareUtils
from source.utils.StringKeyUtils import StringKeyUtils


class AsyncApiHelper:
    """使用aiohttp异步通讯"""

    owner = None
    repo = None

    @staticmethod
    def setRepo(owner, repo):  # 使用之前设置项目名和所有者
        AsyncApiHelper.owner = owner
        AsyncApiHelper.repo = repo

    @staticmethod
    def getAuthorizationHeaders(header):
        """设置Github 的Token用于验证"""
        if header is not None and isinstance(header, dict):
            if configPraser.getAuthorizationToken():
                header[StringKeyUtils.STR_HEADER_AUTHORIZAITON] = (StringKeyUtils.STR_HEADER_TOKEN
                                                                   + configPraser.getAuthorizationToken())
        return header

    @staticmethod
    def getUserAgentHeaders(header):
        """爬虫策略： 随机请求的agent"""
        if header is not None and isinstance(header, dict):
            # header[self.STR_HEADER_USER_AGENT] = self.STR_HEADER_USER_AGENT_SET
            header[StringKeyUtils.STR_HEADER_USER_AGENT] = random.choice(StringKeyUtils.USER_AGENTS)
        return header

    @staticmethod
    def getMediaTypeHeaders(header):
        if header is not None and isinstance(header, dict):
            header[StringKeyUtils.STR_HEADER_ACCEPT] = StringKeyUtils.STR_HEADER_MEDIA_TYPE
        return header

    @staticmethod
    async def getProxy():
        """获取代理ip池中的ip  详细看 ProxyHelper"""
        if configPraser.getProxy():
            proxy = await ProxyHelper.getAsyncSingleProxy()
            if configPraser.getPrintMode():
                print(proxy)
            if proxy is not None:
                return StringKeyUtils.STR_PROXY_HTTP_FORMAT.format(proxy)
        return None

    @staticmethod
    async def parserPullRequest(resultJson):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                res = PullRequest.parser.parser(resultJson)
                if res is not None and res.base is not None:
                    res.repo_full_name = res.base.repo_full_name  # 对pull_request的repo_full_name 做一个补全
                return res
        except Exception as e:
            print(e)

    @staticmethod
    def judgeNotFind(resultJson):
        if resultJson is not None and isinstance(json, dict):
            if resultJson.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_NOT_FIND:
                return True
            if resultJson.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_FAILED_FETCH:
                return True
        return False

    @staticmethod
    async def downloadInformation(pull_number, semaphore, mysql, statistic):
        """获取一个项目 单个pull-request 相关的信息"""

        """增加issue  需要仿写downloadInformation函数 
           只是pull-request的获取转换为issue
        """
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    beanList = []  # 用来收集需要存储的bean类
                    """先获取pull request信息"""
                    api = AsyncApiHelper.getPullRequestApi(pull_number)
                    json = await AsyncApiHelper.fetchBeanData(session, api)
                    pull_request = await AsyncApiHelper.parserPullRequest(json)
                    print(pull_request)
                    usefulPullRequestsCount = 0
                    usefulReviewsCount = 0
                    usefulReviewCommentsCount = 0
                    usefulIssueCommentsCount = 0
                    usefulCommitsCount = 0

                    if pull_request is not None:
                        usefulPullRequestsCount = 1
                        beanList.append(pull_request)

                        if pull_request.head is not None:
                            beanList.append(pull_request.head)
                        if pull_request.base is not None:
                            beanList.append(pull_request.base)
                        if pull_request.user is not None:
                            beanList.append(pull_request.user)

                        reviewCommits = []  # review中涉及的Commit的点

                        """获取review信息"""
                        api = AsyncApiHelper.getReviewForPullRequestApi(pull_number)
                        json = await AsyncApiHelper.fetchBeanData(session, api)
                        reviews = await AsyncApiHelper.parserReview(json, pull_number)
                        if configPraser.getPrintMode():
                            print(reviews)

                        usefulReviewsCount = 0
                        if reviews is not None:
                            for review in reviews:
                                usefulReviewsCount += 1
                                beanList.append(review)
                                if review.user is not None:
                                    beanList.append(review.user)
                                if review.commit_id not in reviewCommits:
                                    reviewCommits.append(review.commit_id)

                        """获取review comment信息"""
                        api = AsyncApiHelper.getReviewCommentForPullRequestApi(pull_number)
                        json = await AsyncApiHelper.fetchBeanData(session, api, isMediaType=True)
                        reviewComments = await AsyncApiHelper.parserReviewComment(json)

                        if configPraser.getPrintMode():
                            print(reviewComments)
                        usefulReviewCommentsCount = 0
                        if reviewComments is not None:
                            for reviewComment in reviewComments:
                                usefulReviewCommentsCount += 1
                                beanList.append(reviewComment)
                                if reviewComment.user is not None:
                                    beanList.append(reviewComment.user)

                        '''获取 pull request对应的issue comment信息'''
                        api = AsyncApiHelper.getIssueCommentForPullRequestApi(pull_number)
                        json = await AsyncApiHelper.fetchBeanData(session, api, isMediaType=True)
                        issueComments = await  AsyncApiHelper.parserIssueComment(json, pull_number)
                        usefulIssueCommentsCount = 0
                        if issueComments is not None:
                            for issueComment in issueComments:
                                usefulIssueCommentsCount += 1
                                beanList.append(issueComment)
                                if issueComment.user is not None:
                                    beanList.append(issueComment.user)

                        '''获取 pull request对应的commit信息'''
                        api = AsyncApiHelper.getCommitForPullRequestApi(pull_number)
                        json = await AsyncApiHelper.fetchBeanData(session, api, isMediaType=True)
                        Commits, Relations = await AsyncApiHelper.parserCommitAndRelation(json, pull_number)

                        for commit in Commits:
                            if commit.sha in reviewCommits:
                                reviewCommits.remove(commit.sha)

                        """有些review涉及的commit的点没有在PR线中收集到 这些点主要是中间存在最后
                        没有的点 但是最后需要在特征提取中用到 所以也需要收集"""

                        """剩下的点需要依次获取"""
                        for commit_id in reviewCommits:
                            api = AsyncApiHelper.getCommitApi(commit_id)
                            json = await AsyncApiHelper.fetchBeanData(session, api)
                            commit = await AsyncApiHelper.parserCommit(json)
                            Commits.append(commit)

                        usefulCommitsCount = 0
                        for commit in Commits:
                            if commit is not None:
                                usefulCommitsCount += 1
                                api = AsyncApiHelper.getCommitApi(commit.sha)
                                json = await AsyncApiHelper.fetchBeanData(session, api)
                                commit = await AsyncApiHelper.parserCommit(json)
                                beanList.append(commit)

                                if commit.committer is not None:
                                    beanList.append(commit.committer)
                                if commit.author is not None:
                                    beanList.append(commit.author)
                                if commit.files is not None:
                                    for file in commit.files:
                                        beanList.append(file)
                                if commit.parents is not None:
                                    for parent in commit.parents:
                                        beanList.append(parent)
                                """作为资源节约   commit comment不做采集"""

                        for relation in Relations:
                            beanList.append(relation)

                        print(beanList)

                    """数据库存储"""
                    await AsyncSqlHelper.storeBeanDateList(beanList, mysql)

                    # 做了同步处理
                    statistic.lock.acquire()
                    statistic.usefulRequestNumber += usefulPullRequestsCount
                    statistic.usefulReviewNumber += usefulReviewsCount
                    statistic.usefulReviewCommentNumber += usefulReviewCommentsCount
                    statistic.usefulIssueCommentNumber += usefulIssueCommentsCount
                    statistic.usefulCommitNumber += usefulCommitsCount
                    print("useful pull request:", statistic.usefulRequestNumber,
                          " useful review:", statistic.usefulReviewNumber,
                          " useful review comment:", statistic.usefulReviewCommentNumber,
                          " useful issue comment:", statistic.usefulIssueCommentNumber,
                          " useful commit:", statistic.usefulCommitNumber,
                          " cost time:", datetime.now() - statistic.startTime)
                    statistic.lock.release()
                except Exception as e:
                    print(e)

    @staticmethod
    async def parserReview(resultJson, pull_number):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                items = []
                for item in resultJson:
                    res = Review.parser.parser(item)
                    res.repo_full_name = AsyncApiHelper.owner + '/' + AsyncApiHelper.repo  # 对repo_full_name 做一个补全
                    res.pull_number = pull_number
                    items.append(res)
                return items
        except Exception as e:
            print(e)

    @staticmethod
    def getPullRequestApi(pull_number):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        return api

    @staticmethod
    def getReviewCommentForPullRequestApi(pull_number):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMENTS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        return api

    @staticmethod
    def getReviewForPullRequestApi(pull_number):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_REVIEWS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        return api

    @staticmethod
    def getIssueCommentForPullRequestApi(issue_number):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_ISSUE_COMMENT_FOR_ISSUE
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_ISSUE_NUMBER, str(issue_number))
        return api

    @staticmethod
    def getCommitForPullRequestApi(pull_number):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMITS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        return api

    @staticmethod
    def getGraphQLApi():
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_GRAPHQL
        return api

    @staticmethod
    def getCommitApi(commit_sha):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMIT
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_COMMIT_SHA, str(commit_sha))
        return api

    @staticmethod
    async def fetchBeanData(session, api, isMediaType=False):
        """异步获取数据通用接口（重要）"""

        """初始化请求头"""
        headers = {}
        headers = AsyncApiHelper.getUserAgentHeaders(headers)
        headers = AsyncApiHelper.getAuthorizationHeaders(headers)
        if isMediaType:
            headers = AsyncApiHelper.getMediaTypeHeaders(headers)
        while True:
            """对单个请求循环判断 直到请求成功或者错误"""

            """获取代理ip  ip获取需要运行代理池"""
            proxy = await AsyncApiHelper.getProxy()
            if configPraser.getProxy() and proxy is None:  # 对代理池没有ip的情况做考虑
                print('no proxy and sleep!')
                await asyncio.sleep(20)
            else:
                break

        try:
            async with session.get(api, ssl=False, proxy=proxy
                    , headers=headers, timeout=configPraser.getTimeout()) as response:
                print("rate:", response.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_REMIAN))
                print("status:", response.status)
                if response.status == 403:
                    await ProxyHelper.judgeProxy(proxy.split('//')[1], ProxyHelper.INT_KILL_POINT)
                    raise 403
                elif proxy is not None:
                    await ProxyHelper.judgeProxy(proxy.split('//')[1], ProxyHelper.INT_POSITIVE_POINT)
                return await response.json()
        except Exception as e:
            """非 403的网络请求出错  循环重试"""
            print(e)
            if proxy is not None:
                proxy = proxy.split('//')[1]
                await ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_NEGATIVE_POINT)
            # print("judge end")
            """循环重试"""
            return await AsyncApiHelper.fetchBeanData(session, api, isMediaType=isMediaType)

    @staticmethod
    async def postGraphqlData(session, api, args=None):
        """通过 github graphhql接口 通过post请求"""
        headers = {}
        headers = AsyncApiHelper.getUserAgentHeaders(headers)
        headers = AsyncApiHelper.getAuthorizationHeaders(headers)

        body = {}
        body = GraphqlHelper.getTimeLineQueryByNodes(body)
        body = GraphqlHelper.getGraphqlVariables(body, args)
        bodyJson = json.dumps(body)
        # print("bodyjson:", bodyJson)

        while True:
            proxy = await AsyncApiHelper.getProxy()
            if configPraser.getProxy() and proxy is None:  # 对代理池没有ip的情况做考虑
                print('no proxy and sleep!')
                await asyncio.sleep(20)
            else:
                break

        try:
            async with session.post(api, ssl=False, proxy=proxy,
                                    headers=headers, timeout=configPraser.getTimeout(),
                                    data=bodyJson) as response:
                print("rate:", response.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_REMIAN))
                print("status:", response.status)
                if response.status == 403:
                    await ProxyHelper.judgeProxy(proxy.split('//')[1], ProxyHelper.INT_KILL_POINT)
                    raise 403
                elif proxy is not None:
                    await ProxyHelper.judgeProxy(proxy.split('//')[1], ProxyHelper.INT_POSITIVE_POINT)
                return await response.json()
        except Exception as e:
            print(e)
            if proxy is not None:
                proxy = proxy.split('//')[1]
                await ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_NEGATIVE_POINT)
            # print("judge end")
            return await AsyncApiHelper.postGraphqlData(session, api, args)

    @staticmethod
    async def parserReviewComment(resultJson):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                items = []
                for item in resultJson:
                    res = ReviewComment.parser.parser(item)
                    items.append(res)
                return items
        except Exception as e:
            print(e)

    @staticmethod
    async def parserIssueComment(resultJson, issue_number):
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                items = []
                for item in resultJson:
                    res = IssueComment.parser.parser(item)
                    """信息补全"""
                    res.repo_full_name = AsyncApiHelper.owner + '/' + AsyncApiHelper.repo  # 对repo_full_name 做一个补全
                    res.pull_number = issue_number

                    items.append(res)
                return items
        except Exception as e:
            print(e)

    @staticmethod
    async def parserCommitAndRelation(resultJson, pull_number):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                items = []
                relations = []
                for item in resultJson:
                    res = Commit.parser.parser(item)
                    relation = CommitPRRelation()
                    relation.sha = res.sha
                    relation.pull_number = pull_number
                    relation.repo_full_name = AsyncApiHelper.owner + '/' + AsyncApiHelper.repo
                    relations.append(relation)
                    items.append(res)
                return items, relations
        except Exception as e:
            print(e)

    @staticmethod
    async def parserCommit(resultJson):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                res = Commit.parser.parser(resultJson)
                return res
        except Exception as e:
            print(e)

    @staticmethod
    async def parserPRItemLine(resultJson):
        try:
            if not AsyncApiHelper.judgeNotFind(resultJson):
                res, items = PRTimeLineRelation.parser.parser(resultJson)
                return res, items
        except Exception as e:
            print(e)

    @staticmethod
    async def downloadRPTimeLine(nodeIds, semaphore, mysql, statistic):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    args = {"ids": nodeIds}
                    """先获取pull request信息"""
                    api = AsyncApiHelper.getGraphQLApi()
                    resultJson = await AsyncApiHelper.postGraphqlData(session, api, args)
                    beanList = []
                    # pull_request = await AsyncApiHelper.parserPullRequest(json)
                    # print(pull_request)
                    print(type(resultJson))
                    print("post json:", resultJson)
                    timeLineRelations, timeLineItems = await  AsyncApiHelper.parserPRItemLine(resultJson)

                    usefulTimeLineItemCount = 0
                    usefulTimeLineCount = 0

                    beanList.extend(timeLineRelations)
                    beanList.extend(timeLineItems)
                    await AsyncSqlHelper.storeBeanDateList(beanList, mysql)

                    """完善获取关联的commit 信息"""
                    pairs = PRTimeLineUtils.splitTimeLine(timeLineRelations)
                    for pair in pairs:
                        print(pair)
                        await AsyncApiHelper.completeCommitInformation(pair, mysql, session, statistic)

                    # 做了同步处理
                    statistic.lock.acquire()
                    statistic.usefulTimeLineCount += 1
                    print(f" usefulTimeLineCount:{statistic.usefulTimeLineCount}",
                          f" change trigger count:{statistic.usefulChangeTrigger}",
                          f" twoParents case:{statistic.twoParentsNodeCase}",
                          f" outOfLoop case:{statistic.outOfLoopCase}")
                    statistic.lock.release()
                except Exception as e:
                    print(e)

    @staticmethod
    async def completeCommitInformation(pair, mysql, session, statistic):
        """完善 review和随后事件相关的commit"""
        review = pair[0]
        changes = pair[1]

        """review 的 comments 获取一次即可"""

        """获得review comments"""
        comments = await AsyncApiHelper.getReviewCommentsByNodeFromStore(review.timelineitem_node, mysql)

        twoParentsBadCase = 0
        outOfLoopCase = 0

        for change in changes:  # 对后面连续改动依次遍历
            commit1 = review.pullrequestReviewCommit
            commit2 = None
            if change.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT:
                commit2 = change.pullrequestCommitCommit
            elif change.typename == StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT:
                commit2 = change.headRefForcePushedEventAfterCommit

            loop = 0
            if commit2 is not None and commit1 is not None:

                class CommitNode:
                    willFetch = None
                    oid = None
                    parents = None  # [sha1, sha2 ...]

                """为了统计使用的工具类"""

                def findNodes(nodes, oid):
                    for node in nodes:
                        if node.oid == oid:
                            return node

                def isExist(nodes, oid):
                    for node in nodes:
                        if node.oid == oid:
                            return True
                    return False

                def isNodesContains(nodes1, nodes2):  # nodes2 的所有未探索的点被nodes1 包含
                    isContain = True
                    for node in nodes2:
                        isFind = False
                        for node1 in nodes1:
                            if node1.oid == node.oid:
                                isFind = True
                                break
                        if not isFind and node.willFetch:
                            isContain = False
                            break
                    return isContain

                def printNodes(nodes1, nodes2):
                    print('node1')
                    for node in nodes1:
                        print(node.oid, node.willFetch, node.parents)
                    print('node2')
                    for node in nodes2:
                        print(node.oid, node.willFetch, node.parents)

                async def fetNotFetchedNodes(nodes, mysql, session):
                    fetchList = [node.oid for node in nodes if node.willFetch]
                    """先尝试从数据库中读取"""
                    localExistList, localRelationList = await AsyncApiHelper.getCommitsFromStore(fetchList, mysql)
                    fetchList = [oid for oid in fetchList if oid not in localExistList]
                    # print("res fetch list:", fetchList)
                    webRelationList = await AsyncApiHelper.getCommitsFromApi(fetchList, mysql, session)

                    for node in nodes:
                        node.willFetch = False

                    # for node in nodes:
                    #     print("after fetched 1: " + f"{node.oid}  {node.willFetch}")

                    relationList = []
                    relationList.extend(localRelationList)
                    relationList.extend(webRelationList)
                    # print("relationList:", relationList)
                    # for relation in relationList:
                    #     print(relation.child, "    ", relation.parent)

                    """原有的node 补全parents"""
                    for relation in relationList:
                        node = findNodes(nodes, relation.child)
                        if node is not None:
                            if relation.parent not in node.parents:
                                node.parents.append(relation.parent)

                    addNode = []
                    for relation in relationList:
                        isFind = False
                        """确保在两个地方都不存在"""
                        for node in nodes:
                            if relation.parent == node.oid:
                                isFind = True
                                break
                        for node in addNode:
                            if relation.parent == node.oid:
                                isFind = True
                                break

                        if not isFind:
                            newNode = CommitNode()
                            newNode.willFetch = True
                            newNode.oid = relation.parent
                            newNode.parents = []
                            addNode.append(newNode)
                    nodes.extend(addNode)

                    # for node in nodes:
                    #     print("after fetched  2: " + f"{node.oid}  {node.willFetch}")

                    return nodes

                try:
                    commit1Nodes = []
                    commit2Nodes = []

                    node1 = CommitNode()
                    node1.oid = commit1
                    node1.willFetch = True
                    node1.parents = []
                    commit1Nodes.append(node1)
                    node2 = CommitNode()
                    node2.oid = commit2
                    commit2Nodes.append(node2)
                    node2.willFetch = True
                    node2.parents = []

                    completeFetch = 0
                    while loop < configPraser.getCommitFetchLoop():

                        loop += 1

                        print("loop:", loop, " 1")
                        printNodes(commit1Nodes, commit2Nodes)

                        if isNodesContains(commit1Nodes, commit2Nodes):
                            completeFetch = 2
                            break

                        if isNodesContains(commit2Nodes, commit1Nodes):
                            completeFetch = 1
                            break

                        await fetNotFetchedNodes(commit2Nodes, mysql, session)
                        print("loop:", loop, " 2")
                        printNodes(commit1Nodes, commit2Nodes)

                        if isNodesContains(commit1Nodes, commit2Nodes):
                            completeFetch = 2
                            break
                        if isNodesContains(commit2Nodes, commit1Nodes):
                            completeFetch = 1
                            break

                        await fetNotFetchedNodes(commit1Nodes, mysql, session)

                        print("loop:", loop, " 3")
                        printNodes(commit1Nodes, commit2Nodes)

                    if completeFetch == 0:
                        outOfLoopCase += 1
                        raise Exception('out of the loop !')

                    """找出两组不同的node进行比较"""

                    """被包含的那里开始行走测试 找出两者差异的点  并筛选出一些特殊情况做舍弃"""
                    finishNodes1 = None
                    finishNodes2 = None
                    startNode1 = None
                    startNode2 = None

                    """依据包含关系 确认1，2位对象"""
                    if completeFetch == 2:
                        finishNodes1 = commit1Nodes
                        finishNodes2 = commit2Nodes  # 2号位为被包含
                        startNode1 = node1.oid
                        startNode2 = node2.oid
                    if completeFetch == 1:
                        finishNodes1 = commit2Nodes
                        finishNodes2 = commit1Nodes
                        startNode1 = node2.oid
                        startNode2 = node1.oid

                    diff_nodes1 = []  # 用于存储两边不同差异的点
                    diff_nodes2 = [x for x in finishNodes2 if not findNodes(finishNodes1, x.oid)]

                    # diff_nodes1 先包含所有点，然后找出从2出发到达不了的点

                    diff_nodes1 = finishNodes1.copy()
                    for node in finishNodes2:
                        if not findNodes(finishNodes1, node.oid):  # 去除
                            diff_nodes1.append(node)

                    temp = [startNode2]
                    while temp.__len__() > 0:
                        oid = temp.pop(0)
                        node = findNodes(diff_nodes1, oid)
                        if node is not None:
                            temp.extend(node.parents)
                        diff_nodes1.remove(node)

                    for node in diff_nodes1:
                        if node.willFetch:
                            twoParentsBadCase += 1
                            raise Exception('will fetch node in set 1 !')  # 去除分叉节点未经之前遍历的情况

                    """diff_node1 和 diff_node2 分别存储两边都各异的点"""
                    printNodes(diff_nodes1, diff_nodes2)

                    """除去特异的点中有merge 节点的存在"""
                    for node in diff_nodes1:
                        if node.parents.__len__() >= 2:
                            twoParentsBadCase += 1
                            raise Exception('merge node find in set1 !')
                    for node in diff_nodes2:
                        if node.parents.__len__() >= 2:
                            twoParentsBadCase += 1
                            raise Exception('merge node find in set 2!')

                    if comments is not None:

                        """获得commit 所有的change file"""
                        file1s = await AsyncApiHelper.getFilesFromStore([x.oid for x in diff_nodes1], mysql)
                        file2s = await AsyncApiHelper.getFilesFromStore([x.oid for x in diff_nodes2], mysql)

                        for comment in comments:  # 对每一个comment统计change trigger
                            commentFile = comment.path
                            commentLine = comment.original_line

                            diff_patch1 = []  # 两边不同的patch patch就是不同文本
                            diff_patch2 = []

                            startNode = [startNode1]  # 从commit源头找到根中每一个commit的涉及文件名的patch
                            while startNode.__len__() > 0:
                                node_oid = startNode.pop(0)
                                for node in diff_nodes1:
                                    if node.oid == node_oid:
                                        for file in file1s:
                                            if file.filename == commentFile and file.commit_sha == node.oid:
                                                diff_patch1.insert(0, file.patch)
                                        startNode.extend(node.parents)

                            startNode = [startNode2]
                            while startNode.__len__() > 0:
                                node_oid = startNode.pop(0)
                                for node in diff_nodes2:
                                    if node.oid == node_oid:
                                        for file in file2s:
                                            if file.filename == commentFile and file.commit_sha == node.oid:
                                                diff_patch2.insert(0, file.patch)
                                        startNode.extend(node.parents)

                            closedChange = TextCompareUtils.getClosedFileChange(diff_patch1, diff_patch2, commentLine)
                            print("closedChange :", closedChange)
                            if comment.change_trigger is None:
                                comment.change_trigger = closedChange
                            else:
                                comment.change_trigger = min(comment.change_trigger, closedChange)
                except Exception as e:
                    print(e)
                    continue

            statistic.lock.acquire()
            statistic.outOfLoopCase += outOfLoopCase
            statistic.usefulChangeTrigger += [x for x in comments if x.change_trigger is not None].__len__()
            statistic.lock.release()

        await AsyncSqlHelper.updateBeanDateList(comments, mysql)

    @staticmethod
    async def getReviewCommentsByNodeFromStore(node_id, mysql):
        """从数据库中读取review id 到时候更新只要从数据库中增加就ok了"""

        review = Review()
        review.node_id = node_id

        reviews = await AsyncSqlHelper.queryBeanData([review], mysql, [[StringKeyUtils.STR_KEY_NODE_ID]])
        print("reviews:", reviews)
        if reviews[0].__len__() > 0:
            review_id = reviews[0][0][2]
            print("review_id:", review_id)
            comment = ReviewComment()
            comment.pull_request_review_id = review_id

            result = await AsyncSqlHelper.queryBeanData([comment], mysql,
                                                        [[StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID]])
            print(result)
            if result[0].__len__() > 0:
                comments = BeanParserHelper.getBeansFromTuple(ReviewComment(), ReviewComment.getItemKeyList(),
                                                              result[0])

                """获取comment 以及对应的sha 和nodeId 和行数,fileName"""
                for comment in comments:
                    print(comment.getValueDict())
                return comments

    @staticmethod
    async def getFilesFromStore(oids, mysql):
        """从数据库中读取多个oid的file changes"""

        print("query file oids:", oids)

        queryFiles = []
        for oid in oids:
            file = File()
            file.commit_sha = oid
            queryFiles.append(file)

        gitFiles = []

        if queryFiles.__len__() > 0:
            results = await AsyncSqlHelper.queryBeanData(queryFiles, mysql,
                                                         [[StringKeyUtils.STR_KEY_COMMIT_SHA]] * queryFiles.__len__())
            print("files:", results)
            for result in results:
                if result.__len__() > 0:
                    files = BeanParserHelper.getBeansFromTuple(File(), File.getItemKeyList(), result)
                    gitFiles.extend(files)

        return gitFiles

    @staticmethod
    async def getCommitsFromStore(oids, mysql):

        beans = []

        existList = []  # 存在列表
        relationList = []  # 查询得到的关系列表 在子列表出现代表了系统有存储

        """先从sha(oid)转化为commit对象"""
        for oid in oids:
            bean = CommitRelation()
            bean.child = oid
            beans.append(bean)

        results = await AsyncSqlHelper.queryBeanData(beans, mysql, [[StringKeyUtils.STR_KEY_CHILD]] * beans.__len__())
        print("result:", results)

        """从本地返回的结果做解析"""
        for relationTuple in results:
            if relationTuple.__len__() > 0:
                existList.append(relationTuple[0][0])
                for relation in relationTuple:
                    r = CommitRelation()
                    r.child = relation[0]
                    r.parent = relation[1]
                    relationList.append(r)
        """去重处理"""
        existList = list(set(existList))
        relationList = list(set(relationList))
        return existList, relationList

    @staticmethod
    async def getCommitsFromApi(oids, mysql, session):

        beanList = []
        relationList = []  # 查询得到的关系列表

        for oid in oids:
            api = AsyncApiHelper.getCommitApi(oid)
            json = await AsyncApiHelper.fetchBeanData(session, api)
            # print(json)
            commit = await AsyncApiHelper.parserCommit(json)

            if commit.parents is not None:
                relationList.extend(commit.parents)
            if commit.files is not None:
                beanList.extend(commit.files)

            beanList.append(commit)
        beanList.extend(relationList)
        await AsyncSqlHelper.storeBeanDateList(beanList, mysql)
        return relationList
