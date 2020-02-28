# coding=gbk
import random

import requests
import sys
import json
import io
import time

import os

import urllib3
from urllib3.exceptions import InsecureRequestWarning

from source.config import projectConfig
from source.config import configPraser
from source.data.bean.CommentRelation import CommitRelation
from source.data.bean.CommitComment import CommitComment
from source.data.bean.CommitPRRelation import CommitPRRelation
from source.data.bean.File import File
from source.data.bean.Commit import Commit
from source.data.bean.IssueComment import IssueComment
from source.data.bean.Review import Review
from source.data.bean.CommentPraser import CommentPraser
from source.data.bean.ReviewComment import ReviewComment
from source.data.service.ProxyHelper import ProxyHelper
from source.utils.TableItemHelper import TableItemHelper
from source.utils.StringKeyUtils import StringKeyUtils
from _datetime import datetime
from math import ceil
from source.data.bean.Repository import Repository
from source.data.bean.User import User
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Branch import Branch


class ApiHelper:

    def __init__(self, owner, repo):  # 设置对应的仓库和所属
        self.owner = owner
        self.repo = repo
        self.isUseAuthorization = False
        self.isUseProxyPool = False

    def setOwner(self, owner):
        self.owner = owner

    def setRepo(self, repo):
        self.repo = repo

    def setAuthorization(self, isUseAuthorization):
        self.isUseAuthorization = isUseAuthorization

    def setUseProxyPool(self, isUseProxyPool):
        self.isUseProxyPool = isUseProxyPool

    def getProxy(self):
        if self.isUseProxyPool:
            proxy = ProxyHelper.getSingleProxy()
            if configPraser.configPraser.getPrintMode():
                print(proxy)
            if proxy is not None:
                return {StringKeyUtils.STR_PROXY_HTTP: StringKeyUtils.STR_PROXY_HTTP_FORMAT.format(proxy)}
        return None

    def getAuthorizationHeaders(self, header):
        if header is not None and isinstance(header, dict):
            if self.isUseAuthorization:
                if configPraser.configPraser.getAuthorizationToken():
                    header[StringKeyUtils.STR_HEADER_AUTHORIZAITON] = (StringKeyUtils.STR_HEADER_TOKEN
                                                             + configPraser.configPraser.getAuthorizationToken())

        return header

    def getUserAgentHeaders(self, header):
        if header is not None and isinstance(header, dict):
            # header[self.STR_HEADER_USER_AGENT] = self.STR_HEADER_USER_AGENT_SET
            header[StringKeyUtils.STR_HEADER_USER_AGENT] = random.choice(StringKeyUtils.USER_AGENTS)
        return header

    def getMediaTypeHeaders(self, header):
        if header is not None and isinstance(header, dict):
            header[StringKeyUtils.STR_HEADER_ACCEPT] = StringKeyUtils.STR_HEADER_MEDIA_TYPE

        return header

    def getPullRequestsForProject(self, state=STR_PARM_OPEN):
        """获取一个项目的pull request的列表，但是 只能获取前30个  没参数的时候默认是open
        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, params={StringKeyUtils.STR_PARM_STARE: state}, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return list()

        res = list()
        for request in r.json():
            res.append(request.get(StringKeyUtils.STR_KEY_NUMBER))
            if configPraser.configPraser.getPrintMode():
                print(request.get(StringKeyUtils.STR_KEY_NUMBER))

        if configPraser.configPraser.getPrintMode():
            print(res.__len__())
        return res

    def getLanguageForProject(self):
        """获取一个项目的pull request的列表，但是 只能获取前30个  没参数的时候默认是open
        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PROJECT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return StringKeyUtils.STR_KEY_LANG_OTHER

        return r.json().get(StringKeyUtils.STR_KEY_LANG, StringKeyUtils.STR_KEY_LANG_OTHER)

    def getTotalPullRequestNumberForProject(self):
        """通过获取最新的pull request的编号来获取总数量  获取参数为all

        """

        if self.owner is None or self.repo is None:
            return -1

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, params={StringKeyUtils.STR_PARM_STARE: StringKeyUtils.STR_PARM_ALL}
                         , verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return -1

        list = r.json()
        if list.__len__() > 0:
            request = list[0]
            return request.get(StringKeyUtils.STR_KEY_NUMBER, -1)
        else:
            return -1

    def getMaxSolvedPullRequestNumberForProject(self):
        """通过获取最新的pull request的编号来获取总数量  获取参数为all

        """

        if self.owner is None or self.repo is None:
            return -1

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, params={StringKeyUtils.STR_PARM_STARE: StringKeyUtils.STR_PARM_CLOSED}
                         , verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return -1

        list = r.json()
        if list.__len__() > 0:
            request = list[0]
            return request.get(StringKeyUtils.STR_KEY_NUMBER, -1)
        else:
            return -1

    def getCommentsForPullRequest(self, pull_number):
        """获取一个pull request的 comments  可以获取行号

        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMENTS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))

        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return list()

        res = list()
        for review in r.json():
            # print(review)
            # print(type(review))
            praser = CommentPraser()
            res.append(praser.praser(review))

        return res

    def getCommentsForReview(self, pull_number, review_id):
        """获取一个review的相关comments  这个接口无法获取行号

        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMENTS_FOR_REVIEW
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        api = api.replace(StringKeyUtils.STR_REVIEW_ID, str(review_id))

        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return list()

        res = list()
        for review in r.json():
            # print(review)
            # print(type(review))
            praser = CommentPraser()
            res.append(praser.praser(review))
        return res

    def getReviewForPullRequest(self, pull_number):
        """获取一个pull request的review的id列表

        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_REVIEWS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))

        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return list()

        res = list()
        for review in r.json():
            # print(review)
            res.append(review.get(StringKeyUtils.STR_KEY_ID))

        return res

    def printCommon(self, r):
        if configPraser.configPraser.getPrintMode():
            if isinstance(r, requests.models.Response):
                print(type(r))
                print(r.json())
                print(r.text.encode(encoding='utf_8', errors='strict'))
                print(r.headers)
                print("status:", r.status_code.__str__())
                print("remaining:", r.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_REMIAN))
                print("rateLimit:", r.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_RESET))

    def judgeLimit(self, r):
        if isinstance(r, requests.models.Response):
            remaining = int(r.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_REMIAN))
            rateLimit = int(r.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_RESET))
            if remaining < StringKeyUtils.RATE_LIMIT:
                print("start sleep:", ceil(rateLimit - datetime.now().timestamp() + 1))
                time.sleep(ceil(rateLimit - datetime.now().timestamp() + 1))
                print("sleep end")

    def getInformationForProject(self):
        """获取一个项目的信息  返回一个项目对象
        """
        if self.owner is None or self.repo is None:
            return list()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PROJECT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        res = Repository.parser.parser(r.json())

        if configPraser.configPraser.getPrintMode():
            print(res)
        return res

    def getInformationForUser(self, login):
        """获取一个用户的详细信息"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_USER
        api = api.replace(StringKeyUtils.STR_USER, login)
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        res = User.parser.parser(r.json())

        if configPraser.configPraser.getPrintMode():
            print(res)
        return res

    def getInformationForPullRequest(self, pull_number):
        """获取一个pull request的详细信息"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        res = PullRequest.parser.parser(r.json())
        if res is not None and res.base is not None:
            res.repo_full_name = res.base.repo_full_name  # 对pull_request的repo_full_name 做一个补全

        if configPraser.configPraser.getPrintMode():
            print(res)
        return res

    def getInformationForReview(self, pull_number, review_id):
        """获取一个review 的详细信息"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_REVIEW
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        api = api.replace(StringKeyUtils.STR_REVIEW_ID, str(review_id))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        res = Review.parser.parser(r.json())

        res.repo_full_name = self.owner + '/' + self.repo  # 对repo_full_name 做一个补全
        res.pull_number = pull_number

        print(res)
        return res

    def getInformationForReviewWithPullRequest(self, pull_number):
        """获取一个pull request对应的 review的详细信息 可以节省请求数量"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_REVIEWS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        items = []
        for item in r.json():
            res = Review.parser.parser(item)

            res.repo_full_name = self.owner + '/' + self.repo  # 对repo_full_name 做一个补全
            res.pull_number = pull_number

            if configPraser.configPraser.getPrintMode():
                print(res.getValueDict())
            items.append(res)

        return items

    def getInformationForReviewCommentWithPullRequest(self, pull_number):
        """获取一个pull request对应的 review comment的详细信息 可以节省请求数量"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMENTS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        items = []
        for item in r.json():
            res = ReviewComment.parser.parser(item)
            if configPraser.configPraser.getPrintMode():
                print(res.getValueDict())
            items.append(res)

        return items

    def getInformationForIssueCommentWithIssue(self, issue_number):
        """获取一个issue 对应的 issue comment的详细信息 可以节省请求数量"""
        """但是issue 和 pull request公用一个编号 实际是请求的pull request的评论"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_ISSUE_COMMENT_FOR_ISSUE
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_ISSUE_NUMBER, str(issue_number))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        items = []
        for item in r.json():
            res = IssueComment.parser.parser(item)
            if configPraser.configPraser.getPrintMode():
                print(res.getValueDict())

            """信息补全"""
            res.repo_full_name = self.owner + '/' + self.repo  # 对repo_full_name 做一个补全
            res.pull_number = issue_number

            items.append(res)

        return items

    def getInformationCommit(self, commit_sha):
        """获取一个commit 对应的详细信息"""
        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMIT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_COMMIT_SHA, str(commit_sha))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        res = Commit.parser.parser(r.json())
        return res

    def getInformationForCommitWithPullRequest(self, pull_number):
        """获取一个pull request对应的 commit的详细信息 可以节省请求数量
        但是 status 没有统计,file 也没有统计"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMITS_FOR_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        items = []
        relations = []
        for item in r.json():
            res = Commit.parser.parser(item)
            if configPraser.configPraser.getPrintMode():
                print(res.getValueDict())
            items.append(res)

            relation = CommitPRRelation()
            relation.sha = res.sha
            relation.pull_number = pull_number
            relation.repo_full_name = self.owner + '/' + self.repo
            relations.append(relation)

        return items, relations

    def getInformationForCommitCommentsWithCommit(self, commit_sha):
        """获取一个commit对应的 commit comment的详细信息 可以节省请求"""

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_COMMIT_COMMENTS_FOR_COMMIT
        api = api.replace(StringKeyUtils.STR_OWNER, self.owner)
        api = api.replace(StringKeyUtils.STR_REPO, self.repo)
        api = api.replace(StringKeyUtils.STR_COMMIT_SHA, commit_sha)
        # print(api)
        #         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

        headers = {}
        headers = self.getUserAgentHeaders(headers)
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        proxy = self.getProxy()
        urllib3.disable_warnings(InsecureRequestWarning)
        r = requests.get(api, headers=headers, verify=False, proxies=proxy)
        self.printCommon(r)
        self.judgeLimit(r)
        if r.status_code != 200:
            return None

        items = []
        for item in r.json():
            res = CommitComment.parser.parser(item)
            if configPraser.configPraser.getPrintMode():
                print(res.getValueDict())
            items.append(res)

        return items


if __name__ == "__main__":
    helper = ApiHelper('rails', 'rails')
    helper.setAuthorization(True)
    helper.setUseProxyPool(True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # print(helper.getReviewForPullRequest(38211))
    # helper.getPullRequestsForPorject(state = ApiHelper.STR_PARM_ALL)
    #     print("total:" + helper.getTotalPullRequestNumberForProject().__str__())
    #     print(helper.getCommentsForReview(38211,341374357))
    #     print(helper.getCommentsForPullRequest(38211))
    #     print(helper.getCommentsForPullRequest(38211))
    # print(helper.getMaxSolvedPullRequestNumberForProject())
    # print(helper.getLanguageForProject())
    # print(helper.getInformationForProject().getItemKeyListWithType())
    # print(helper.getInformationForUser('jonathanhefner').getItemKeyListWithType())
    # print(helper.getTotalPullRequestNumberForProject())
    # print(Branch.getItemKeyListWithType())
    # print(helper.getInformationForPullRequest(38383).getValueDict())
    # print(Review.getItemKeyListWithType())
    # print(helper.getInformationForReview(38211, 341373994).getValueDict())
    # print(helper.getInformationForReviewWithPullRequest(38211))
    # print(helper.getInformationForReviewCommentWithPullRequest(38539))
    # print(helper.getInformationForIssueCommentWithIssue(38529))
    # print(CommitRelation.getItemKeyList())
    # print(CommitRelation().getValueDict())
    # print(helper.getInformationForCommit('b4256cea5d812660f28ca148835afcf273376c8e').parents[0].getValueDict())
    # print(helper.getInformationForCommitWithPullRequest(38449))
    # print(helper.getInformationForCommitCommentsWithCommit('2e74177d0b61f872b773285471ff9025f0eaa96c'))
