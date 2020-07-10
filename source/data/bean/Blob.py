from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class Blob(BeanBase):
    """Git 对应的Blob文件
    """

    def __init__(self):
        self.repository = None
        self.oid = None
        self.node_id = None
        self.byte_size = None
        self.is_binary = None
        self.is_truncated = None
        self.text = None
        """text github 接口最多支持100M的文本 text类型无法支持 需要使用LongText"""

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPOSITORY, StringKeyUtils.STR_KEY_OID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPOSITORY, StringKeyUtils.STR_KEY_OID,
                 StringKeyUtils.STR_KEY_NODE_ID,
                 StringKeyUtils.STR_KEY_BYTE_SIZE, StringKeyUtils.STR_KEY_IS_BINARY,
                 StringKeyUtils.STR_KEY_IS_TRUNCATED, StringKeyUtils.STR_KEY_TEXT]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPOSITORY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_OID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BYTE_SIZE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_IS_BINARY, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_IS_TRUNCATED, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_TEXT, BeanBase.DATA_TYPE_LONG_TEXT)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPOSITORY: self.repository,
                 StringKeyUtils.STR_KEY_OID: self.oid,
                 StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_BYTE_SIZE: self.byte_size,
                 StringKeyUtils.STR_KEY_IS_BINARY: self.is_binary,
                 StringKeyUtils.STR_KEY_IS_TRUNCATED: self.is_truncated,
                 StringKeyUtils.STR_KEY_TEXT: self.text}

        return items

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            if isinstance(src, dict):
                res = Blob()
                res.oid = src.get(StringKeyUtils.STR_KEY_OID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.is_binary = src.get(StringKeyUtils.STR_KEY_IS_BINARY_V4, None)
                res.is_truncated = src.get(StringKeyUtils.STR_KEY_IS_TRUNCATED_V4, None)
                res.byte_size = src.get(StringKeyUtils.STR_KEY_BYTE_SIZE_V4, None)
                res.text = src.get(StringKeyUtils.STR_KEY_TEXT, None)

                repoData = src.get(StringKeyUtils.STR_KEY_REPOSITORY, None)
                if isinstance(repoData, dict):
                    res.repository = repoData.get(StringKeyUtils.STR_KEY_NAME_WITH_OWNER, None)

                return res