# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class CommitPRRelation(BeanBase):
    """github 为了记录commit 和  pull request关系的补偿类"""

    def __init__(self):
        self.repo_full_name = None
        self.pull_number = None
        self.sha = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER,
                StringKeyUtils.STR_KEY_SHA]

    @staticmethod
    def getItemKeyList():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER,
                StringKeyUtils.STR_KEY_SHA]

    @staticmethod
    def getItemKeyListWithType():
        return [(StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                (StringKeyUtils.STR_KEY_PULL_NUMBER, BeanBase.DATA_TYPE_INT),
                (StringKeyUtils.STR_KEY_SHA, BeanBase.DATA_TYPE_STRING)]

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,
                 StringKeyUtils.STR_KEY_PULL_NUMBER: self.pull_number,
                 StringKeyUtils.STR_KEY_SHA: self.sha}
        return items

    class parser(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = CommitPRRelation()
                res.repo_full_name = src.get(StringKeyUtils.STR_KEY_REPO_FULL_NAME, None)
                res.pull_number = src.get(StringKeyUtils.STR_KEY_PULL_NUMBER, None)
                res.sha = src.get(StringKeyUtils.STR_KEY_SHA, None)
            return res
