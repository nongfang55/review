#coding=gbk
import os
from source.config.projectConfig import projectConfig
import configparser


class configPraser: #用于解析config。ini文件
    
    
    STR_TOKEN = 'token'
    STR_AUTHORIZATION = 'authorization'
    
    @staticmethod
    def getAuthorizationToken():
        cp = configparser.ConfigParser()
        cp.read(projectConfig.getConfigPath())
        return cp.get(configPraser.STR_AUTHORIZATION, configPraser.STR_TOKEN)
        