# coding=gbk
import asyncio
import json
import random
import time
import traceback
from datetime import datetime

import aiohttp

from source.config.configPraser import configPraser
from source.data.bean.Commit import Commit
from source.data.bean.CommitPRRelation import CommitPRRelation
from source.data.bean.IssueComment import IssueComment
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Review import Review
from source.data.bean.ReviewComment import ReviewComment
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.data.service.GraphqlHelper import GraphqlHelper
from source.data.service.ProxyHelper import ProxyHelper
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
    async def parserPullRequest(json):
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                res = PullRequest.parser.parser(json)
                if res is not None and res.base is not None:
                    res.repo_full_name = res.base.repo_full_name  # 对pull_request的repo_full_name 做一个补全
                return res
        except Exception as e:
            print(e)

    @staticmethod
    def judgeNotFind(json):
        """判断404的场景，用于按照某些编号检索Github实体失败"""
        if json is not None and isinstance(json, dict):
            if json.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_NOT_FIND:
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
    async def parserReview(json, pull_number):
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                items = []
                for item in json:
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
    async def parserReviewComment(json):
        """解析json 获取Review Comment信息"""
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                items = []
                for item in json:
                    res = ReviewComment.parser.parser(item)
                    items.append(res)
                return items
        except Exception as e:
            print(e)

    @staticmethod
    async def parserIssueComment(json, issue_number):
        """解析json 获取Issue Comment信息"""
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                items = []
                for item in json:
                    res = IssueComment.parser.parser(item)
                    """信息补全"""
                    res.repo_full_name = AsyncApiHelper.owner + '/' + AsyncApiHelper.repo  # 对repo_full_name 做一个补全
                    res.pull_number = issue_number

                    items.append(res)
                return items
        except Exception as e:
            print(e)

    @staticmethod
    async def parserCommitAndRelation(json, pull_number):
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                items = []
                relations = []
                for item in json:
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
    async def parserCommit(json):
        """解析json 获取Commit信息"""
        try:
            if not AsyncApiHelper.judgeNotFind(json):
                res = Commit.parser.parser(json)
                return res
        except Exception as e:
            print(e)

    @staticmethod
    async def parserPRItemLine(json):
        try:
            # if not AsyncApiHelper.judgeNotFind(json):
            res = PRTimeLineRelation.parser.parser(json)
            return res
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
                    # pull_request = await AsyncApiHelper.parserPullRequest(json)
                    # print(pull_request)
                    print(type(resultJson))
                    print("post json:", resultJson)
                    beanList = await  AsyncApiHelper.parserPRItemLine(resultJson)
                    usefulTimeLineItemCount = 0
                    usefulTimeLineCount = 0

                    await AsyncSqlHelper.storeBeanDateList(beanList, mysql)

                    # 做了同步处理
                    statistic.lock.acquire()
                    statistic.usefulTimeLineCount += usefulTimeLineCount
                    print(f"usefulTimeLineItemCount: {usefulTimeLineCount}",
                          f" usefulTimeLineCount:{usefulTimeLineItemCount}")
                    statistic.lock.release()
                except Exception as e:
                    print(e)
