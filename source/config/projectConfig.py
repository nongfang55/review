# coding=gbk
import os


class projectConfig:
    projectName = 'review'
    PATH_CONFIG = 'source' + os.sep + 'config' + os.sep + 'config.txt'
    PATH_TEST_INPUT_EXCEL = 'data' + os.sep + 'Test200.xlsx'
    PATH_TEST_OUTPUT_EXCEL = 'data' + os.sep + 'output.xlsx'
    PATH_TEST_OUTPUT_PATH = 'data'
    PATH_STOP_WORD_HGD = 'data' + os.sep + 'HGDStopWord.txt'
    PATH_SPLIT_WORD_EXCEL = 'data' + os.sep + 'output_splitword.xlsx'
    PATH_USER_DICT_PATH = 'data' + os.sep + 'user_dict.utf8'
    PATH_TEST_CRF_INPUT = 'data' + os.sep + 'people-daily.txt'
    PATH_TEST_CRF_TEST_RESULT = 'data' + os.sep + 'test.rst'
    PATH_TEST_REVIEW_COMMENT = 'data' + os.sep + 'reviewComments.tsv'
    PATH_TEST_WINE_RED = 'data' + os.sep + 'winequality-red.xlsx'
    PATH_TEST_REVHELPER_DATA = 'data' + os.sep + 'revhelperDemoData.csv'
    PATH_TEST_FPS_DATA = 'data' + os.sep + 'FPSDemoData.tsv'
    PATH_STOP_WORD_ENGLISH = 'data' + os.sep + 'stop-words_english_1_en.txt'
    PATH_RUBY_KEY_WORD = 'data' + os.sep + 'rubyKeyWord.txt'
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

    @staticmethod
    def getReviewCommentTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_REVIEW_COMMENT)

    @staticmethod
    def getRandomForestTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_REVHELPER_DATA)

    @staticmethod
    def getFPSTestData():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TEST_FPS_DATA)

    @staticmethod
    def getStopWordEnglishPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_STOP_WORD_ENGLISH)

    @staticmethod
    def getRubyKeyWordPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_RUBY_KEY_WORD)


if __name__ == "__main__":
    print(projectConfig.getRootPath())
    print(projectConfig.getConfigPath())
    print(projectConfig.getTestInputExcelPath())
    print(projectConfig.getDataPath())
    print(projectConfig.getTestoutputExcelPath())
