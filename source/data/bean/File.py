# coding=gbk

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class File(BeanBase):
    """github  commit中的file的数据结构  8项"""

    def __init__(self):
        self.commit_sha = None
        self.sha = None
        self.filename = None
        self.status = None
        self.additions = None
        self.deletions = None
        self.changes = None
        self.patch = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_COMMIT_SHA, StringKeyUtils.STR_KEY_SHA]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_COMMIT_SHA, StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_FILENAME,
                 StringKeyUtils.STR_KEY_STATUS, StringKeyUtils.STR_KEY_ADDITIONS, StringKeyUtils.STR_KEY_DELETIONS,
                 StringKeyUtils.STR_KEY_CHANGES, StringKeyUtils.STR_KEY_PATCH]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_COMMIT_SHA, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_SHA, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_FILENAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_STATUS, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_ADDITIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_DELETIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_CHANGES, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_PATCH, BeanBase.DATA_TYPE_STRING)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_COMMIT_SHA: self.commit_sha, StringKeyUtils.STR_KEY_SHA: self.sha,
                 StringKeyUtils.STR_KEY_FILENAME: self.filename, StringKeyUtils.STR_KEY_STATUS: self.status,
                 StringKeyUtils.STR_KEY_ADDITIONS: self.additions, StringKeyUtils.STR_KEY_DELETIONS: self.deletions,
                 StringKeyUtils.STR_KEY_CHANGES: self.changes, StringKeyUtils.STR_KEY_PATCH: self.patch}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = File()
                res.commit_sha = src.get(StringKeyUtils.STR_KEY_COMMIT_SHA, None)
                res.sha = src.get(StringKeyUtils.STR_KEY_SHA, None)
                res.filename = src.get(StringKeyUtils.STR_KEY_FILENAME, None)
                res.status = src.get(StringKeyUtils.STR_KEY_STATUS, None)
                res.additions = src.get(StringKeyUtils.STR_KEY_ADDITIONS, None)
                res.deletions = src.get(StringKeyUtils.STR_KEY_DELETIONS, None)
                res.changes = src.get(StringKeyUtils.STR_KEY_CHANGES, None)
                res.patch = src.get(StringKeyUtils.STR_KEY_PATCH, None)

            return res
