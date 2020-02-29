# coding=gbk
import asyncio
import random
import time
import traceback

import aiohttp

from source.config.configPraser import configPraser
from source.data.bean.PullRequest import PullRequest
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
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
        if header is not None and isinstance(header, dict):
            if configPraser.getAuthorizationToken():
                header[StringKeyUtils.STR_HEADER_AUTHORIZAITON] = (StringKeyUtils.STR_HEADER_TOKEN
                                                                   + configPraser.getAuthorizationToken())
        return header

    @staticmethod
    def getUserAgentHeaders(header):
        if header is not None and isinstance(header, dict):
            # header[self.STR_HEADER_USER_AGENT] = self.STR_HEADER_USER_AGENT_SET
            header[StringKeyUtils.STR_HEADER_USER_AGENT] = random.choice(StringKeyUtils.USER_AGENTS)
        return header

    @staticmethod
    async def getProxy():
        if configPraser.getProxy():
            proxy = await ProxyHelper.getAsyncSingleProxy()
            if configPraser.getPrintMode():
                print(proxy)
            if proxy is not None:
                return StringKeyUtils.STR_PROXY_HTTP_FORMAT.format(proxy)
        return None

    @staticmethod
    async def fetchPullRequest(session, pull_number):
        headers = {}
        headers = AsyncApiHelper.getUserAgentHeaders(headers)
        headers = AsyncApiHelper.getAuthorizationHeaders(headers)
        proxy = await AsyncApiHelper.getProxy()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))

        try:
            async with session.get(api, ssl=False, proxy=proxy
                    , headers=headers, timeout=configPraser.getTimeout()) as response:
                print(response.headers.get(StringKeyUtils.STR_HEADER_RATE_LIMIT_REMIAN))
                print("status:", response.status)
                # if response.status == 403:
                #     ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_KILL_POINT)
                # elif proxy is not None:
                #     ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_POSITIVE_POINT)
                return await response.json()
        except Exception as e:
            print(e)
            # traceback.print_exc()
            print('重试：', pull_number)
            # if proxy is not None:
            #     ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_NEGATIVE_POINT)
            return await AsyncApiHelper.fetchPullRequest(session, pull_number)

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
        if json is not None and isinstance(json, dict):
            if json.get(StringKeyUtils.STR_KEY_MESSAGE) == StringKeyUtils.STR_NOT_FIND:
                return True
        return False

    @staticmethod
    async def downloadInformation(pull_number, semaphore, mysql):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    json = await AsyncApiHelper.fetchPullRequest(session, pull_number)
                    pull_request = await AsyncApiHelper.parserPullRequest(json)
                    print(pull_request)
                    await AsyncSqlHelper.storeBeanData(pull_request, mysql)
                except Exception as e:
                    print(e)

