# coding=gbk
import asyncio
import random
import time

import aiohttp
from retrying import retry

from source.config.configPraser import configPraser
from source.data.service.ProxyHelper import ProxyHelper
from source.utils.StringKeyUtils import StringKeyUtils


def getUserAgentHeaders(header):
    if header is not None and isinstance(header, dict):
        # header[self.STR_HEADER_USER_AGENT] = self.STR_HEADER_USER_AGENT_SET
        header[StringKeyUtils.STR_HEADER_USER_AGENT] = random.choice(StringKeyUtils.USER_AGENTS)
    return header


class AsyncApiHelper:
    """使用aiohttp异步通讯"""

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
    async def fetch(session, url):
        headers = {}
        headers = AsyncApiHelper.getUserAgentHeaders(headers)
        headers = AsyncApiHelper.getAuthorizationHeaders(headers)
        try:
            async with session.get(url, ssl=False, proxy=AsyncApiHelper.getProxy()
                                   , headers=headers, timeout=5) as response:
                print(response.headers.items())
                print("status:", response.status)
                return await response.text()
        except Exception as e:
            print(e)
            print('重试：', url)
            await AsyncApiHelper.fetch(session, url)

    @staticmethod
    async def parser(json):
        try:
            pass
            print(json)
        except Exception as e:
            print(e)

    @staticmethod
    async def download(url, semaphore):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    print(url)
                    json = await AsyncApiHelper.fetch(session, url)
                    await AsyncApiHelper.parser(json)
                except Exception as e:
                    print(e)

    @staticmethod
    def demo():
        semaphore = asyncio.Semaphore(1000)
        base_url = 'https://api.github.com/repos/rails/rails/pulls/{0}'
        start = 38449
        urls = []
        for number in range(38449, 38349, -1):
            urls.append(base_url.format(number))
        print(urls)

        t1 = time.time()
        loop = asyncio.get_event_loop()
        tasks = [asyncio.ensure_future(AsyncApiHelper.download(url, semaphore)) for url in urls]
        tasks = asyncio.gather(*tasks)
        loop.run_until_complete(tasks)

        t2 = time.time()
        print('cost time:', t2 - t1)


if __name__ == '__main__':
    AsyncApiHelper.demo()
