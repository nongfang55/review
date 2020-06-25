# coding=gbk

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class PRChangeFile(BeanBase):
    """github  pull request涉及的file改动  5项"""

    def __init__(self):
        self.repo_full_name = None
        self.pull_number = None
        self.filename = None
        self.additions = None
        self.deletions = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER,
                 StringKeyUtils.STR_KEY_FILENAME, StringKeyUtils.STR_KEY_ADDITIONS,
                 StringKeyUtils.STR_KEY_DELETIONS]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PULL_NUMBER, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_FILENAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_ADDITIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_DELETIONS, BeanBase.DATA_TYPE_INT)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,
                 StringKeyUtils.STR_KEY_PULL_NUMBER: self.pull_number,
                 StringKeyUtils.STR_KEY_FILENAME: self.filename,
                 StringKeyUtils.STR_KEY_ADDITIONS: self.additions,
                 StringKeyUtils.STR_KEY_DELETIONS: self.deletions}

        return items

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = PRChangeFile()
                res.repo_full_name = src.get(StringKeyUtils.STR_KEY_REPO_FULL_NAME, None)
                res.pull_number = src.get(StringKeyUtils.STR_KEY_PULL_NUMBER, None)
                res.filename = src.get(StringKeyUtils.STR_KEY_PATH, None)
                res.additions = src.get(StringKeyUtils.STR_KEY_ADDITIONS, None)
                res.deletions = src.get(StringKeyUtils.STR_KEY_DELETIONS, None)

            return res
