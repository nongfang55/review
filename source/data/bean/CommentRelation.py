# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class CommitRelation(BeanBase):
    """github 为了记录commit关系的补偿类"""

    def __init__(self):
        self.child = None
        self.parent = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_CHILD, StringKeyUtils.STR_KEY_PARENT]

    @staticmethod
    def getItemKeyList():
        return [StringKeyUtils.STR_KEY_CHILD, StringKeyUtils.STR_KEY_PARENT]

    @staticmethod
    def getItemKeyListWithType():
        return [(StringKeyUtils.STR_KEY_CHILD, BeanBase.DATA_TYPE_STRING),
                (StringKeyUtils.STR_KEY_PARENT, BeanBase.DATA_TYPE_STRING)]

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_CHILD: self.child, StringKeyUtils.STR_KEY_PARENT: self.parent}
        return items

    class parser(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = CommitRelation()
                res.child = src.get(StringKeyUtils.STR_KEY_CHILD, None)
                res.parent = src.get(StringKeyUtils.STR_KEY_PARENT, None)
            return res
