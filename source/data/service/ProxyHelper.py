# coding=gbk
import requests


class ProxyHelper:
    """用于使用 ip代理池 proxy_pool的api接口类"""

    STR_PROXY_GET_API = "http://127.0.0.1:5010/get/"
    STR_PROXY_GET_ALL_API = 'http://127.0.0.1:5010/get_all/'
    STR_PROXY_DELETE_API = 'http://127.0.0.1:5010/delete/?proxy={}'

    STR_KEY_PROXY = 'proxy'

    @staticmethod
    def getSingleProxy():
        json = requests.get(ProxyHelper.STR_PROXY_GET_API).json()
        if json is not None:
            return json.get(ProxyHelper.STR_KEY_PROXY, None)

    @staticmethod
    def getAllProxy():
        return requests.get(ProxyHelper.STR_PROXY_GET_ALL_API).json()

    @staticmethod
    def delete_proxy(proxy):
        requests.get(ProxyHelper.STR_PROXY_DELETE_API.format(proxy))



if __name__ == '__main__':
    print(ProxyHelper.getAllProxy())
