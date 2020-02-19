#coding=gbk
import os
from source.config.projectConfig import projectConfig
import configparser


class configPraser: #用于解析config。ini文件
    
    
    STR_TOKEN = 'token'
    STR_AUTHORIZATION = 'authorization'
    STR_DATABASE = 'database'
    
    STR_USERNAME = 'username'
    STR_PASSWORD = 'password'
    STR_HOST = 'host'
    
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

        
        