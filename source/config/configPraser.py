# coding=gbk
import os
from source.config.projectConfig import projectConfig
import configparser


class configPraser:  # 用于解析config。ini文件

    STR_TOKEN = 'token'
    STR_AUTHORIZATION = 'authorization'
    STR_DATABASE = 'database'
    STR_DEBUG = 'debug'
    STR_PROJECT = 'project'

    STR_USERNAME = 'username'
    STR_PASSWORD = 'password'
    STR_HOST = 'host'
    STR_PRINT = 'print'
    STR_TRUE = 'True'
    STR_RETRY = 'retry'
    STR_OWNER = 'owner'
    STR_REPO = 'repo'
    STR_LIMIT = 'limit'

    @staticmethod
    def getAuthorizationToken():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_AUTHORIZATION, configPraser.STR_TOKEN)

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
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_DEBUG, configPraser.STR_PRINT) == configPraser.STR_TRUE

    @staticmethod
    def getRetryTime():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return int(cp.get(configPraser.STR_DEBUG, configPraser.STR_RETRY))

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



