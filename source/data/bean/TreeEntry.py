from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class TreeEntry(BeanBase):
    """记录 Git 对应的GitObject 之间的关系
       包括 Tree与Tree的关系， Tree与 Blob的关系
    """

    def __init__(self):
        self.repository = None
        self.parent_oid = None
        self.child_oid = None

        self.parent_type = None
        self.parent_path = None
        self.parent_node_id = None
        """parent_path 不一定保存"""

        self.child_type = None
        self.child_path = None
        self.child_node_id = None

        """text github 接口最多支持100M的文本 text类型无法支持 需要使用LongText"""

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPOSITORY, StringKeyUtils.STR_KEY_PARENT_OID,
                StringKeyUtils.STR_KEY_CHILD_OID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPOSITORY, StringKeyUtils.STR_KEY_PARENT_OID,
                 StringKeyUtils.STR_KEY_CHILD_OID, StringKeyUtils.STR_KEY_PARENT_TYPE,
                 StringKeyUtils.STR_KEY_PARENT_PATH, StringKeyUtils.STR_KEY_PARENT_NODE_ID,
                 StringKeyUtils.STR_KEY_CHILD_TYPE, StringKeyUtils.STR_KEY_CHILD_PATH,
                 StringKeyUtils.STR_KEY_CHILD_NODE_ID]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPOSITORY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PARENT_OID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHILD_OID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PARENT_TYPE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PARENT_PATH, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PARENT_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHILD_PATH, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHILD_TYPE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHILD_NODE_ID, BeanBase.DATA_TYPE_STRING)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPOSITORY: self.repository,
                 StringKeyUtils.STR_KEY_PARENT_OID: self.parent_oid,
                 StringKeyUtils.STR_KEY_CHILD_OID: self.child_oid,
                 StringKeyUtils.STR_KEY_PARENT_TYPE: self.parent_type,
                 StringKeyUtils.STR_KEY_PARENT_PATH: self.parent_path,
                 StringKeyUtils.STR_KEY_PARENT_NODE_ID: self.parent_node_id,
                 StringKeyUtils.STR_KEY_CHILD_TYPE: self.child_type,
                 StringKeyUtils.STR_KEY_CHILD_PATH: self.child_path,
                 StringKeyUtils.STR_KEY_CHILD_NODE_ID: self.child_node_id}

        return items

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res_list = []
            if isinstance(src, dict):
                repoData = src.get(StringKeyUtils.STR_KEY_REPOSITORY, None)
                if isinstance(repoData, dict):
                    repository = repoData.get(StringKeyUtils.STR_KEY_NAME_WITH_OWNER, None)

                parent_type = src.get(StringKeyUtils.STR_KEY_TYPE_NAME_JSON, None)
                parent_oid = src.get(StringKeyUtils.STR_KEY_OID, None)
                parent_node_id = src.get(StringKeyUtils.STR_KEY_ID, None)

                entryDataList = src.get(StringKeyUtils.STR_KEY_ENTRIES, None)
                if isinstance(entryDataList, list):
                    for entryData in entryDataList:
                        if isinstance(entryData, dict):
                            bean = TreeEntry()
                            bean.repository = repository
                            bean.parent_type = parent_type
                            bean.parent_oid = parent_oid
                            bean.parent_node_id = parent_node_id
                            bean.child_path = entryData.get(StringKeyUtils.STR_KEY_NAME, None)
                            bean.child_type = entryData.get(StringKeyUtils.STR_KEY_TYPE, None)

                            childData = entryData.get(StringKeyUtils.STR_KEY_OBJECT, None)
                            if isinstance(childData, dict):
                                bean.child_node_id = childData.get(StringKeyUtils.STR_KEY_ID, None)
                                bean.child_oid = childData.get(StringKeyUtils.STR_KEY_OID, None)

                                res_list.append(bean)
            return res_list








