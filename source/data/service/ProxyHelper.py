# coding=gbk
import aiohttp
import requests


class ProxyHelper:
    """用于使用 ip代理池 proxy_pool的api接口类"""

    STR_PROXY_GET_API = "http://127.0.0.1:5010/get/"
    STR_PROXY_GET_ALL_API = 'http://127.0.0.1:5010/get_all/'
    STR_PROXY_DELETE_API = 'http://127.0.0.1:5010/delete/?proxy={}'

    STR_KEY_PROXY = 'proxy'

    ip_pool = {}  # ip缓冲池

    INT_INITIAL_POINT = 5
    INT_POSITIVE_POINT = 1  # 正反馈分数
    INT_NEGATIVE_POINT = -1  # 负反馈分数
    INT_DELETE_POINT = 0  # 删除分数
    INT_KILL_POINT = -1000  # 直接干掉

    @staticmethod
    async def getAsyncSingleProxy():
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(ProxyHelper.STR_PROXY_GET_API) as response:
                    json = await response.json(content_type=None)
            except Exception as e:
                print(e)
            if json is not None:
                proxy = json.get(ProxyHelper.STR_KEY_PROXY, None)
                if proxy is not None and ProxyHelper.ip_pool.get(proxy, None) is None:
                    ProxyHelper.ip_pool[proxy] = ProxyHelper.INT_INITIAL_POINT
                return proxy

    @staticmethod
    def getSingleProxy():
        json = requests.get(ProxyHelper.STR_PROXY_GET_API).json()
        if json is not None:
            proxy = json.get(ProxyHelper.STR_KEY_PROXY, None)
            if proxy is not None and ProxyHelper.ip_pool.get(proxy, None) is None:
                ProxyHelper.ip_pool[proxy] = ProxyHelper.INT_INITIAL_POINT
            return proxy

    @staticmethod
    def getAllProxy():
        return requests.get(ProxyHelper.STR_PROXY_GET_ALL_API).json()

    @staticmethod
    async def delete_proxy(proxy):
        print('delete proxy:', proxy)
        # requests.get(ProxyHelper.STR_PROXY_DELETE_API.format(proxy))
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(ProxyHelper.STR_PROXY_DELETE_API.format(proxy)) as response:
                    json = await response.json(content_type=None)
            except Exception as e:
                print(e)

    @staticmethod
    async def judgeProxy(proxy, point):
        print("judge proxy:", proxy, "  :", point)
        try:
            now = ProxyHelper.ip_pool[proxy]
            if now is not None:
                now += point
                if now < ProxyHelper.INT_DELETE_POINT:
                    ProxyHelper.ip_pool.pop(proxy)
                    await ProxyHelper.delete_proxy(proxy)
                else:
                    ProxyHelper.ip_pool[proxy] = now
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print(ProxyHelper.getAllProxy().__len__())
