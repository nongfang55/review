# coding=gbk
import os


class projectConfig:
    projectName = 'review'
    PATH_CONFIG = os.sep + 'source' + os.sep + 'config' + os.sep + 'config.ini'
    PATH_TEST_INPUT_EXCEL = os.sep + 'data' + os.sep + 'Test200.xlsx'
    PATH_TEST_OUTPUT_EXCEL = os.sep + 'data' + os.sep + 'output.xlsx'
    PATH_TEST_OUTPUT_PATH = os.sep + 'data'
    PATH_STOP_WORD_HGD = os.sep + 'data' + os.sep + 'HGDStopWord.txt'
    PATH_SPLIT_WORD_EXCEL = os.sep + 'data' + os.sep + 'output_splitword.xlsx'
    PATH_USER_DICT_PATH = os.sep + 'data' + os.sep + 'user_dict.utf8'
    PATH_TEST_CRF_INPUT = os.sep + 'data' + os.sep + 'people-daily.txt'
    PATH_TEST_CRF_TEST_RESULT = os.sep + 'data' + os.sep + 'test.rst'
    TEST_OUT_PUT_SHEET_NAME = 'sheet1'

    @staticmethod
    def getRootPath():
        curPath = os.path.abspath(os.path.dirname(__file__))
        projectName = projectConfig.projectName
        rootPath = os.path.join(curPath.split(projectName)[0], projectName)  # 获取myProject，也就是项目的根路径
        return rootPath

    @staticmethod
    def getConfigPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_CONFIG)

    @staticmethod
    def getDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_OUTPUT_PATH)

    @staticmethod
    def getTestInputExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_INPUT_EXCEL)

    @staticmethod
    def getTestoutputExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_OUTPUT_EXCEL)

    @staticmethod
    def getStopWordHGDPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_STOP_WORD_HGD)

    @staticmethod
    def getSplitWordExcelPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_SPLIT_WORD_EXCEL)

    @staticmethod
    def getUserDictPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_USER_DICT_PATH)

    @staticmethod
    def getCRFInputData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_CRF_INPUT)

    @staticmethod
    def getCRFTestDataResult():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_CRF_TEST_RESULT)


if __name__ == "__main__":
    print(projectConfig.getRootPath())
    print(projectConfig.getConfigPath())
    print(projectConfig.getTestInputExcelPath())
    print(projectConfig.getDataPath())
    print(projectConfig.getTestoutputExcelPath())
