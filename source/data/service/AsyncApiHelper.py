# coding=gbk
import asyncio
import time

import aiohttp


class AsyncApiDemo:
    """使用aiohttp异步通讯"""

    @staticmethod
    async def fetch(session, url):
        async with session.get(url) as response:
            return await response.text()

    @staticmethod
    async def parser(json):
        try:
            print(json)
        except Exception as e:
            print(e)

    @staticmethod
    async def download(url, semaphore):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    json = await AsyncApiDemo.fetch(session, url)
                    await AsyncApiDemo.parser(json)
                except Exception as e:
                    print(e)

    @staticmethod
    def demo():
        semaphore = asyncio.Semaphore(400)
        base_url = 'https://api.github.com/repos/rails/rails/pulls/{0}'
        start = 38449
        urls = []
        for number in range(38449, 37349, -1):
            urls.append(base_url.format(number))
        print(urls)

        t1 = time.time()
        loop = asyncio.get_event_loop()
        tasks = [asyncio.ensure_future(AsyncApiDemo.download(url, semaphore)) for url in urls]
        tasks = asyncio.gather(*tasks)
        loop.run_until_complete(tasks)

        t2 = time.time()
        print('cost time:', t2 - t1)


if __name__ == '__main__':
    AsyncApiDemo.demo()
