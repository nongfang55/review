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
    PATH_CHANGE_TRIGGER = 'data' + os.sep + 'pullrequest_rails.tsv'
    PATH_COMMIT_RELATION = 'data' + os.sep + 'train' + os.sep + 'prCommitRelation'
    PATH_ISSUE_COMMENT_PATH = 'data' + os.sep + 'train' + os.sep + 'issueCommentData'
    PATH_DATA_TRAIN = 'data' + os.sep + 'train'
    PATH_COMMIT_FILE = 'data' + os.sep + 'train' + os.sep + 'commitFileData'
    PATH_SEAA = 'data' + os.sep + 'SEAA'
    PATH_PULL_REQUEST = 'data' + os.sep + 'train' + os.sep + 'pullRequestData'
    PATH_PR_CHANGE_FILE = 'data' + os.sep + 'train' + os.sep + 'prChangeFile'
    PATH_REVIEW = 'data' + os.sep + 'train' + os.sep + 'reviewData'
    PATH_TIMELINE = 'data' + os.sep + 'train' + os.sep + 'prTimeLineData'

    PATH_FPS_DATA = 'data' + os.sep + 'train' + os.sep + 'FPS'
    PATH_ML_DATA = 'data' + os.sep + 'train' + os.sep + 'ML'
    PATH_IR_DATA = 'data' + os.sep + 'train' + os.sep + 'IR'
    PATH_CA_DATA = 'data' + os.sep + 'train' + os.sep + 'CA'

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

    @staticmethod
    def getChangeTriggerPRPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_CHANGE_TRIGGER)

    @staticmethod
    def getPrCommitRelationPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_COMMIT_RELATION)

    @staticmethod
    def getIssueCommentPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_ISSUE_COMMENT_PATH)

    @staticmethod
    def getDataTrainPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_DATA_TRAIN)

    @staticmethod
    def getCommitFilePath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_COMMIT_FILE)

    @staticmethod
    def getPullRequestPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_PULL_REQUEST)

    @staticmethod
    def getFPSDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_FPS_DATA)

    @staticmethod
    def getMLDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_ML_DATA)

    @staticmethod
    def getIRDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_IR_DATA)

    @staticmethod
    def getCADataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_CA_DATA)

    @staticmethod
    def getSEAADataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_SEAA)

    @staticmethod
    def getPRChangeFilePath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_PR_CHANGE_FILE)

    @staticmethod
    def getReviewDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_REVIEW)

    @staticmethod
    def getPRTimeLineDataPath():
        return os.path.join(projectConfig.getRootPath(), projectConfig.PATH_TIMELINE)

    @staticmethod
    def getLogPath():
        return projectConfig.getRootPath() + os.sep + 'log'


if __name__ == "__main__":
    print(projectConfig.getRootPath())
    print(projectConfig.getConfigPath())
    print(projectConfig.getTestInputExcelPath())
    print(projectConfig.getDataPath())
    print(projectConfig.getTestoutputExcelPath())
