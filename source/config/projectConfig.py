#coding=gbk
import os


class projectConfig:
    
    projectName = 'review'
    PATH_CONFIG = '\source\config\config.ini'
    PATH_TEST_INPUT_EXCEL = '\data\Test200.xlsx'
    PATH_TEST_OUTPUT_EXCEL = '\data\output.xlsx'
    PATH_TEST_OUTPUT_PATH = '\data'
    PATH_STOP_WORD_HGD = '\data\HGDStopWord.txt'
    PATH_SPLIT_WORD_EXCEL = '\data\output_splitword.xlsx'
    PATH_USER_DICT_PATH=  r'\data\user_dict.utf8'
    
    TEST_OUT_PUT_SHEET_NAME = 'sheet1'
    

    @staticmethod
    def getRootPath():
        curPath = os.path.abspath(os.path.dirname(__file__))
        projectName = projectConfig.projectName
        rootPath = curPath.split(projectName)[0] + projectName  # 获取myProject，也就是项目的根路径
        return rootPath
    
    @staticmethod
    def getConfigPath():
        return projectConfig.getRootPath() + projectConfig.PATH_CONFIG
    
    @staticmethod
    def getDataPath():
        return projectConfig.getRootPath() + projectConfig.PATH_TEST_OUTPUT_PATH
    
    @staticmethod
    def getTestInputExcelPath():
        return projectConfig.getRootPath() + projectConfig.PATH_TEST_INPUT_EXCEL
    
    @staticmethod
    def getTestoutputExcelPath():
        return projectConfig.getRootPath() + projectConfig.PATH_TEST_OUTPUT_EXCEL
    
    @staticmethod
    def getStopWordHGDPath():
        return projectConfig.getRootPath() + projectConfig.PATH_STOP_WORD_HGD
    
    @staticmethod
    def getSplitWordExcelPath():
        return projectConfig.getRootPath() + projectConfig.PATH_SPLIT_WORD_EXCEL
    
    @staticmethod
    def getUserDictPath():
        return projectConfig.getRootPath() + projectConfig.PATH_USER_DICT_PATH
    
    
if __name__=="__main__":
    print(projectConfig.getRootPath())
    print(projectConfig.getConfigPath())
    print(projectConfig.getTestInputExcelPath())
    print(projectConfig.getDataPath())
    print(projectConfig.getTestoutputExcelPath())