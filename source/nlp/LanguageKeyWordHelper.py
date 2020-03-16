# coding=gbk
from source.config.projectConfig import projectConfig


class LanguageKeyWordLanguage:

    @staticmethod
    def getRubyKeyWordList():
        file = open(projectConfig.getRubyKeyWordPath(), mode='r+', encoding='utf-8')
        content = file.read()
        return content.split('\n')
