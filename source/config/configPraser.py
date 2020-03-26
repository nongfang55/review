# coding=gbk
import os
import random

from source.config.projectConfig import projectConfig
import configparser


class configPraser:  # 用于解析config。ini文件

    STR_TOKEN = 'token'
    STR_AUTHORIZATION = 'authorization'
    STR_DATABASE = 'database'
    STR_DEBUG = 'debug'
    STR_PROJECT = 'project'
    STR_NETWORK = 'network'
    STR_RECOMMEND = 'recommend'

    STR_USERNAME = 'username'
    STR_PASSWORD = 'password'
    STR_HOST = 'host'
    STR_PRINT = 'print'
    STR_TRUE = 'True'
    STR_RETRY = 'retry'
    STR_OWNER = 'owner'
    STR_REPO = 'repo'
    STR_LIMIT = 'limit'
    STR_PROXY = 'proxy'
    STR_START = 'start'
    STR_TIMEOUT = 'timeout'
    STR_SEMAPHORE = 'semaphore'
    STR_TOPK = 'topk'
    STR_TEST_NUMBER = 'testNumber'
    STR_REVIEWER_NUMBER = 'reviewerNumber'
    STR_FPS_REMOVE_AUTHOR = 'FPSRemoveAuthor'
    STR_FPS_CTYPES = 'FPSCtypes'
    STR_COMMIT_FETCH_LOOP = 'commitFetchLoop'

    cacheDict = {}  # 用于缓存的字典，防止多次访问拖慢速度

    @staticmethod
    def getAuthorizationToken():
        temp = configPraser.cacheDict.get((configPraser.STR_AUTHORIZATION, configPraser.STR_TOKEN), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            tokenList = cp.get(configPraser.STR_AUTHORIZATION, configPraser.STR_TOKEN).split(',')
            configPraser.cacheDict[(configPraser.STR_AUTHORIZATION, configPraser.STR_TOKEN)] = tokenList
            return tokenList[random.randint(0, tokenList.__len__() - 1)]
        else:
            return temp[random.randint(0, temp.__len__() - 1)]

    @staticmethod
    def getDataBaseUserName():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_DATABASE, configPraser.STR_USERNAME)

    @staticmethod
    def getDataBasePassword():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_DATABASE, configPraser.STR_PASSWORD)

    @staticmethod
    def getDataBaseHost():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_DATABASE, configPraser.STR_HOST)

    @staticmethod
    def getPrintMode():
        temp = configPraser.cacheDict.get((configPraser.STR_DEBUG, configPraser.STR_PRINT), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            res = cp.get(configPraser.STR_DEBUG, configPraser.STR_PRINT) == configPraser.STR_TRUE
            configPraser.cacheDict[(configPraser.STR_DEBUG, configPraser.STR_PRINT)] = res
            return res
        else:
            return temp

    @staticmethod
    def getProxy():
        temp = configPraser.cacheDict.get((configPraser.STR_NETWORK, configPraser.STR_PROXY), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            res = cp.get(configPraser.STR_NETWORK, configPraser.STR_PROXY) == configPraser.STR_TRUE
            configPraser.cacheDict[(configPraser.STR_NETWORK, configPraser.STR_PROXY)] = res
            return res
        else:
            return temp

    @staticmethod
    def getRetryTime():
        temp = configPraser.cacheDict.get((configPraser.STR_NETWORK, configPraser.STR_RETRY), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            res = int(cp.get(configPraser.STR_NETWORK, configPraser.STR_RETRY))
            configPraser.cacheDict[(configPraser.STR_NETWORK, configPraser.STR_RETRY)] = res
            return res
        else:
            return temp

    @staticmethod
    def getOwner():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_PROJECT, configPraser.STR_OWNER)

    @staticmethod
    def getRepo():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_PROJECT, configPraser.STR_REPO)

    @staticmethod
    def getLimit():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_PROJECT, configPraser.STR_LIMIT))

    @staticmethod
    def getStart():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_PROJECT, configPraser.STR_START))

    @staticmethod
    def getTimeout():
        temp = configPraser.cacheDict.get((configPraser.STR_NETWORK, configPraser.STR_TIMEOUT), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            res = int(cp.get(configPraser.STR_NETWORK, configPraser.STR_TIMEOUT))
            configPraser.cacheDict[(configPraser.STR_NETWORK, configPraser.STR_TIMEOUT)] = res
            return res
        else:
            return temp

    @staticmethod
    def getSemaphore():
        temp = configPraser.cacheDict.get((configPraser.STR_NETWORK, configPraser.STR_SEMAPHORE), None)
        if temp is None:
            cp = configparser.ConfigParser()
            cp.read(projectConfig.getConfigPath())
            res = int(cp.get(configPraser.STR_NETWORK, configPraser.STR_SEMAPHORE))
            configPraser.cacheDict[(configPraser.STR_NETWORK, configPraser.STR_SEMAPHORE)] = res
            return res
        else:
            return temp

    @staticmethod
    def getDataBase():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_DATABASE, configPraser.STR_DATABASE)

    @staticmethod
    def getTopK():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_RECOMMEND, configPraser.STR_TOPK))

    @staticmethod
    def getTestNumber():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_RECOMMEND, configPraser.STR_TEST_NUMBER))

    @staticmethod
    def getReviewerNumber():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_RECOMMEND, configPraser.STR_REVIEWER_NUMBER))

    @staticmethod
    def getFPSRemoveAuthor():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_RECOMMEND, configPraser.STR_FPS_REMOVE_AUTHOR) == configPraser.STR_TRUE

    @staticmethod
    def getFPSCtypes():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_RECOMMEND, configPraser.STR_FPS_CTYPES) == configPraser.STR_TRUE

    @staticmethod
    def getCommitFetchLoop():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_PROJECT, configPraser.STR_COMMIT_FETCH_LOOP))


if __name__ == '__main__':
    print(configPraser.getCommitFetchLoop())