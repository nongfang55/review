# coding=gbk
import asyncio
import random
import time

import aiohttp
from retrying import retry

from source.config.configPraser import configPraser
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
    def getProxy():
        if configPraser.getProxy():
            proxy = ProxyHelper.getSingleProxy()
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
        proxy = AsyncApiHelper.getProxy()

        api = StringKeyUtils.API_GITHUB + StringKeyUtils.API_PULL_REQUEST
        api = api.replace(StringKeyUtils.STR_OWNER, AsyncApiHelper.owner)
        api = api.replace(StringKeyUtils.STR_REPO, AsyncApiHelper.repo)
        api = api.replace(StringKeyUtils.STR_PULL_NUMBER, str(pull_number))

        try:
            async with session.get(api, ssl=False, proxy=proxy
                    ,headers=headers, timeout=configPraser.getTimeout()) as response:
                print(response.headers.items())
                print("status:", response.status)
                if proxy is not None:
                    ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_POSITIVE_POINT)
                return await response.text()
        except Exception as e:
            print(e)
            print('重试：', pull_number)
            if proxy is not None:
                ProxyHelper.judgeProxy(proxy, ProxyHelper.INT_NEGATIVE_POINT)
            return await AsyncApiHelper.fetchPullRequest(session, pull_number)

    @staticmethod
    async def parser(json):
        try:
            print(json)
        except Exception as e:
            print(e)

    @staticmethod
    async def downloadInformation(pull_number, semaphore):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    json = await AsyncApiHelper.fetchPullRequest(session, pull_number)
                    await AsyncApiHelper.parser(json)
                except Exception as e:
                    print(e)
