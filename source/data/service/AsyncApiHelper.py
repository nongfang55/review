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
from source.data.bean.PRChangeFile import PRChangeFile
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Review import Review
from source.data.bean.ReviewComment import ReviewComment
from source.data.bean.User import User
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
    async def parserPullRequest(resultJson, pull_number=None, rawData=None):
        try:
            res = None
            if configPraser.getApiVersion() == StringKeyUtils.API_VERSION_RESET:
                if not AsyncApiHelper.judgeNotFind(resultJson):
                    res = PullRequest.parser.parser(resultJson)
            elif configPraser.getApiVersion() == StringKeyUtils.API_VERSION_GRAPHQL:
                res = PullRequest.parserV4.parser(resultJson)
                """对于v4接口 pr获取不到的情况，如果确认不存在，则是视为等issue的情况"""
                """读取errors 信息"""
                if res is None:
                    errorMessage = rawData.get(StringKeyUtils.STR_KEY_ERRORS)[0]. \
                        get(StringKeyUtils.STR_KEY_MESSAGE)
                    if errorMessage.find(StringKeyUtils.STR_KEY_ERRORS_PR_NOT_FOUND) != -1:
                        res = PullRequest()
                        res.repo_full_name = AsyncApiHelper.owner + '/' + AsyncApiHelper.repo
                        res.number = pull_number
                        res.is_pr = False
            if res is not None and res.base is not None:
                res.repo_full_name = res.base.repo_full_name  # 对pull_request的repo_full_name 做一个补全
            return res
        except Exception as e:
            print(e)

    @staticmethod
    def judgeNotFind(resultJson):
        if configPraser.getApiVersion() == StringKeyUtils.API_VERSION_RESET:
            if resultJson is not None and isinstance(json, dict):
                if resultJson.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_NOT_FIND:
                    return True
                if resultJson.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_FAILED_FETCH:
                    return True
            return False
        elif configPraser.getApiVersion() == StringKeyUtils.API_VERSION_GRAPHQL:
            if resultJson is not None and isinstance(json, dict):
                if resultJson.get(StringKeyUtils.STR_KEY_ERRORS) is not None:
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
    async def downloadInformationByV4(pull_number, semaphore, mysql, statistic):
        """获取一个项目 单个pull-request 相关的信息
           主要接口请求迁移到GraphQl的v4接口上   这样可以一次性获取pr信息
           保证了pr信息的完整性
           但是commit的具体信息无法获取  这个准备单独开函数获取

           即gitFile的信息和其他信息获取分离
        """
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    beanList = []  # 用来收集需要存储的bean类
                    """先获取pull request信息"""
                    args = {"number": pull_number, "owner": AsyncApiHelper.owner, "name": AsyncApiHelper.repo}
                    api = AsyncApiHelper.getGraphQLApi()
                    query = GraphqlHelper.getPrInformationByNumber()
                    resultJson = await AsyncApiHelper.postGraphqlData(session, api, query, args)
                    print(resultJson)

                    """解析pull request"""
                    allData = resultJson.get(StringKeyUtils.STR_KEY_DATA, None)
                    if allData is not None and isinstance(allData, dict):
                        repoData = allData.get(StringKeyUtils.STR_KEY_REPOSITORY, None)
                        if repoData is not None and isinstance(repoData, dict):
                            prData = repoData.get(StringKeyUtils.STR_KEY_ISSUE_OR_PULL_REQUEST, None)

                            pull_request = await AsyncApiHelper.parserPullRequest(prData, pull_number, resultJson)

                            usefulPullRequestsCount = 0
                            usefulReviewsCount = 0
                            usefulReviewCommentsCount = 0
                            usefulIssueCommentsCount = 0
                            usefulCommitsCount = 0

                            """添加pul request 和 branch"""
                            if pull_request is not None:
                                usefulPullRequestsCount = 1
                                beanList.append(pull_request)
                                if pull_request.head is not None:
                                    beanList.append(pull_request.head)
                                if pull_request.base is not None:
                                    beanList.append(pull_request.base)

                            if pull_request is not None and pull_request.is_pr:
                                users = []
                                """解析 user 直接从pr的participate获取"""
                                user_list = prData.get(StringKeyUtils.STR_KEY_PARTICIPANTS, None)
                                if user_list is not None and isinstance(user_list, dict):
                                    user_list_nodes = user_list.get(StringKeyUtils.STR_KEY_NODES, None)
                                    if user_list_nodes is not None and isinstance(user_list_nodes, list):
                                        for userData in user_list_nodes:
                                            user = User.parserV4.parser(userData)
                                            if user is not None:
                                                users.append(user)
                                """添加用户"""
                                beanList.extend(users)

                                """解析 review, review comment, review 涉及的 commit 信息"""
                                reviews = []
                                reviewComments = []
                                commits = []
                                review_list = prData.get(StringKeyUtils.STR_KEY_REVIEWS, None)
                                if review_list is not None and isinstance(review_list, dict):
                                    review_list_nodes = review_list.get(StringKeyUtils.STR_KEY_NODES, None)
                                    if review_list_nodes is not None and isinstance(review_list_nodes, list):
                                        for reviewData in review_list_nodes:
                                            review = Review.parserV4.parser(reviewData)
                                            if review is not None:
                                                review.repo_full_name = pull_request.repo_full_name
                                                review.pull_number = pull_number
                                                reviews.append(review)

                                            if reviewData is not None and isinstance(reviewData, dict):
                                                comment_list = reviewData.get(StringKeyUtils.STR_KEY_COMMENTS, None)
                                                if comment_list is not None and isinstance(comment_list, dict):
                                                    comment_list_nodes = comment_list.get(StringKeyUtils.STR_KEY_NODES
                                                                                          , None)
                                                    if comment_list_nodes is not None and isinstance(comment_list_nodes
                                                            , list):
                                                        for commentData in comment_list_nodes:
                                                            comment = ReviewComment.parserV4.parser(commentData)
                                                            comment.pull_request_review_id = review.id
                                                            reviewComments.append(comment)

                                                commitData = reviewData.get(StringKeyUtils.STR_KEY_COMMIT, None)
                                                if commitData is not None and isinstance(commitData, dict):
                                                    commit = Commit.parserV4.parser(commitData)
                                                    isFind = False
                                                    for c in commits:
                                                        if c.sha == commit.sha:
                                                            isFind = True
                                                            break
                                                    if not isFind:
                                                        commits.append(commit)

                                """对于2016年的数据  没有review数据项，而PullRequestReviewThread
                                   可以获取对应 review、review comment和 commit
                                """
                                itemLineItem_list = prData.get(StringKeyUtils.STR_KEY_TIME_LINE_ITEMS, None)
                                if itemLineItem_list is not None and isinstance(itemLineItem_list, dict):
                                    itemLineItem_list_edges = itemLineItem_list.get(StringKeyUtils.STR_KEY_EDGES, None)
                                    if itemLineItem_list_edges is not None and isinstance(itemLineItem_list_edges,
                                                                                          list):
                                        for itemLineItem_list_edge_node in itemLineItem_list_edges:
                                            if itemLineItem_list_edge_node is not None and \
                                                    isinstance(itemLineItem_list_edge_node, dict):
                                                itemLineItem_list_edge_node = itemLineItem_list_edge_node.\
                                                    get(StringKeyUtils.STR_KEY_NODE, None)
                                                typename = itemLineItem_list_edge_node.get(
                                                    StringKeyUtils.STR_KEY_TYPE_NAME_JSON, None)
                                                if typename == StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_THREAD:
                                                    """ReviewThread 作为Review 存储到数据库中  但是只有node_id 信息"""
                                                    review = Review()
                                                    review.pull_number = pull_request
                                                    review.repo_full_name = pull_request.repo_full_name
                                                    review.node_id = itemLineItem_list_edge_node.get(StringKeyUtils.STR_KEY_ID, None)
                                                    reviews.append(review)

                                                    """解析 review 涉及的review comment"""
                                                    comment_list = itemLineItem_list_edge_node.get(StringKeyUtils.STR_KEY_COMMENTS, None)
                                                    if comment_list is not None and isinstance(comment_list, dict):
                                                        comment_list_nodes = comment_list.get(
                                                            StringKeyUtils.STR_KEY_NODES
                                                            , None)
                                                        if comment_list_nodes is not None and isinstance(
                                                                comment_list_nodes
                                                                , list):
                                                            for commentData in comment_list_nodes:
                                                                comment = ReviewComment.parserV4.parser(commentData)
                                                                comment.pull_request_review_id = review.id
                                                                reviewComments.append(comment)

                                                                """"从commentData 解析 original commit"""
                                                                commitData = commentData.get(
                                                                    StringKeyUtils.STR_KEY_ORIGINAL_COMMIT, None)
                                                                if commitData is not None and isinstance(commitData,
                                                                                                         dict):
                                                                    commit = Commit.parserV4.parser(commitData)
                                                                    isFind = False
                                                                    for c in commits:
                                                                        if c.sha == commit.sha:
                                                                            isFind = True
                                                                            break
                                                                    if not isFind:
                                                                        commits.append(commit)

                                if configPraser.getPrintMode():
                                    print(reviews)
                                    print(reviewComments)

                                usefulReviewsCount += reviews.__len__()
                                usefulReviewCommentsCount += reviewComments.__len__()

                                """添加review reviewComments"""
                                beanList.extend(reviews)
                                beanList.extend(reviewComments)

                                """issue comment 信息获取"""
                                issueComments = []
                                issue_comment_list = prData.get(StringKeyUtils.STR_KEY_COMMENTS, None)
                                if issue_comment_list is not None and isinstance(issue_comment_list, dict):
                                    issue_comment_list_nodes = issue_comment_list.get(StringKeyUtils.STR_KEY_NODES,
                                                                                      None)
                                    if issue_comment_list_nodes is not None and isinstance(issue_comment_list_nodes,
                                                                                           list):
                                        for commentData in issue_comment_list_nodes:
                                            issueComment = IssueComment.parserV4.parser(commentData)
                                            issueComment.pull_number = pull_number
                                            issueComment.repo_full_name = pull_request.repo_full_name
                                            issueComments.append(issueComment)

                                if configPraser.getPrintMode():
                                    print(issueComments)
                                usefulIssueCommentsCount += issueComments.__len__()
                                beanList.extend(issueComments)

                                """获取 pr 中直接关联的 commit 信息"""
                                commit_list = prData.get(StringKeyUtils.STR_KEY_COMMITS, None)
                                if commit_list is not None and isinstance(commit_list, dict):
                                    commit_list_nodes = commit_list.get(StringKeyUtils.STR_KEY_NODES, None)
                                    if commit_list_nodes is not None and isinstance(commit_list_nodes, list):
                                        for commitData in commit_list_nodes:
                                            commitData = commitData.get(StringKeyUtils.STR_KEY_COMMIT, None)
                                            commit = Commit.parserV4.parser(commitData)
                                            isFind = False
                                            for c in commits:
                                                if c.sha == commit.sha:
                                                    isFind = True
                                                    break
                                            if not isFind:
                                                commits.append(commit)

                                """整合 commitPrRelation 和 commitRelation"""
                                CommitPrRelations = []
                                CommitRelations = []
                                for commit in commits:
                                    relation = CommitPRRelation()
                                    relation.repo_full_name = pull_request.repo_full_name
                                    relation.pull_number = pull_number
                                    relation.sha = commit.sha
                                    CommitPrRelations.append(relation)
                                    CommitRelations.extend(commit.parents)

                                usefulCommitsCount += commits.__len__()
                                beanList.extend(CommitPrRelations)
                                beanList.extend(CommitRelations)
                                beanList.extend(commits)

                                """新增 pull request 涉及的文件变动，而不是commit文件变动的累加"""
                                files = []
                                files_list = prData.get(StringKeyUtils.STR_KEY_FILES, None)
                                if files_list is not None and isinstance(files_list, dict):
                                    files_list_nodes = files_list.get(StringKeyUtils.STR_KEY_NODES, None)
                                    if files_list_nodes is not None and isinstance(files_list_nodes, list):
                                        for fileData in files_list_nodes:
                                            file = PRChangeFile.parserV4.parser(fileData)
                                            file.pull_number = pull_number
                                            file.repo_full_name = pull_request.repo_full_name
                                            files.append(file)

                                if configPraser.getPrintMode():
                                    print(files)

                                beanList.extend(files)

                            """beanList 添加各个数据项"""

                            """数据库存储"""
                            if beanList.__len__() > 0:
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
    def getCommitApiWithProjectName(owner, repo, commit_sha):
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMIT
        api = api.replace(StringKeyUtils.STR_OWNER, owner)
        api = api.replace(StringKeyUtils.STR_REPO, repo)
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
    async def postGraphqlData(session, api, query=None, args=None):
        """通过 github graphhql接口 通过post请求"""
        headers = {}
        headers = AsyncApiHelper.getUserAgentHeaders(headers)
        headers = AsyncApiHelper.getAuthorizationHeaders(headers)

        body = {}
        body = GraphqlHelper.getGraphlQuery(body, query)
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
            return await AsyncApiHelper.postGraphqlData(session, api, query, args)

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
                    """从GitHub v4 API 中获取 某个pull-request的TimeLine对象"""
                    api = AsyncApiHelper.getGraphQLApi()
                    query = GraphqlHelper.getTimeLineQueryByNodes()
                    resultJson = await AsyncApiHelper.postGraphqlData(session, api, query, args)
                    beanList = []
                    print(type(resultJson))
                    print("post json:", resultJson)
                    """从回应结果解析 timeLineItems 和 Relations"""
                    timeLineRelations, timeLineItems = await  AsyncApiHelper.parserPRItemLine(resultJson)

                    usefulTimeLineItemCount = 0
                    usefulTimeLineCount = 0

                    beanList.extend(timeLineRelations)
                    beanList.extend(timeLineItems)
                    """存储数据库中"""
                    await AsyncSqlHelper.storeBeanDateList(beanList, mysql)

                    """完善获取关联的commit 信息"""
                    pairs = PRTimeLineUtils.splitTimeLine(timeLineRelations)
                    for pair in pairs:
                        print(pair)
                        """依照每一对review和后面关联的commit来判断这个reivew中comment的有效性"""
                        """completeCommitInformation 这个函数名取得不好"""
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
        """依照commit来判断review中comment的有效性"""
        review = pair[0]
        changes = pair[1]

        """review 的 comments 获取一次即可"""

        """获得review comments"""
        """一个review 可能会关联多个comment  
        每个comment会指定一个文件和对应代码行"""
        comments = await AsyncApiHelper.getReviewCommentsByNodeFromStore(review.timelineitem_node, mysql)

        twoParentsBadCase = 0  # 记录一个commit有两个根节点的情况 发现这个情况直接停止
        outOfLoopCase = 0  # 记录寻找两个commit点的最近祖宗节点 使用上级追溯的次数超过限制情况

        """changes 代指review后面关联的commit 依次就changeTrigger的情况进行判断"""
        for change in changes:  # 对后面连续改动依次遍历
            """通过commit1和commit2 来比较两者之间的代码差异"""
            """commit1 : review的涉及的commit
               commit2 : reivew后面作者改动的commit
            """
            commit1 = review.pullrequestReviewCommit
            commit2 = None
            if change.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT:
                commit2 = change.pullrequestCommitCommit
            elif change.typename == StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT:
                commit2 = change.headRefForcePushedEventAfterCommit

            """后面算法就是  通过commit1和commit2 来比较两者之间的代码差异"""
            """大体思路：  commit1和它的祖宗节点组成一个commit的集合Group1
                          commit2同样组成了Group2
                          
                          在Group中每一个commit点都有以下信息：
                          1. oid (commit-sha)
                          2. 父节点的 oid
                          3. 这个commit涉及的文件改动
                          
                          Group中包含两种类型节点，一种是信息已经获取，还有一种是信息尚未获取。
                          信息已经获取代表了这个commit上面三个信息都知道，而未获取代表了这个commit
                          只有oid信息。
                          
                          Group一次迭代是指，每次获取类型为为获取信息的commit点，点作为获取信息的节点
                          加入Group中，而commit指向的父节点作为未获取信息节点加入Group中。
                          
                          两个commit作为起点不断做迭代操作，直到某个Group中未包含的点集合包含在了
                          另外一个Group的总体节点中
                          
                          迭代结束之后分别找到两个Group独特的commit点集合，作为后续算法的输入
            """

            """缺点： 现在算法无法处理commit点有两个父类的情况，如merge操作出现的点
                      现在算法感觉怪怪的，效率可能不是很高
                      
                      这个问题应该是LCA问题的变种
            """

            """算法限制： 1、commit点获取次数越少越好
                         2、两个commit点版本差异过过大的时候可以检测，并妥善处理 
            """

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
                    """获取commit点信息 包括数据库获取的GitHub API获取 nodes就是一个Group"""
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
                            """新加入为获取的点"""
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
                    """两个Group的迭代过程"""
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
                        """迭代次数有限制"""

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
                            """comment 对应的文件和文件行"""
                            commentFile = comment.path
                            commentLine = comment.original_line

                            diff_patch1 = []  # 两边不同的patch patch就是不同文本集合
                            diff_patch2 = []

                            startNode = [startNode1]  # 从commit源头找到根中每一个commit的涉及文件名的patch
                            while startNode.__len__() > 0:
                                """类似DFS算法"""
                                node_oid = startNode.pop(0)
                                for node in diff_nodes1:
                                    if node.oid == node_oid:
                                        for file in file1s:
                                            if file.filename == commentFile and file.commit_sha == node.oid:
                                                """patch是一个含有某些行数变化的文本，需要后面单独的解析"""
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

                            """通过比较commit集合来计算距离comment最近的文件变化"""
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

    @staticmethod
    async def downloadCommits(projectName, oid, semaphore, mysql, statistic):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    beanList = []
                    owner, repo = projectName.split('/')
                    api = AsyncApiHelper.getCommitApiWithProjectName(owner, repo, oid)
                    json = await AsyncApiHelper.fetchBeanData(session, api)
                    # print(json)
                    commit = await AsyncApiHelper.parserCommit(json)

                    if commit.parents is not None:
                        beanList.extend(commit.parents)
                    if commit.files is not None:
                        beanList.extend(commit.files)

                    beanList.append(commit)
                    await AsyncSqlHelper.storeBeanDateList(beanList, mysql)

                    # 做了同步处理
                    statistic.lock.acquire()
                    statistic.usefulCommitNumber += 1
                    print(f" usefulCommitCount:{statistic.usefulCommitNumber}")
                    statistic.lock.release()
                except Exception as e:
                    print(e)
