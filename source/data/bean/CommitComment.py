# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User


class CommitComment(BeanBase):
    """github 的commit comment数据类  12个"""

    def __init__(self):
        self.commit_id = None
        self.id = None
        self.node_id = None
        self.user = None
        self.position = None
        self.line = None
        self.path = None
        self.created_at = None
        self.updated_at = None
        self.author_association = None
        self.body = None

        self.user_login = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_COMMIT_ID, StringKeyUtils.STR_KEY_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_COMMIT_ID, StringKeyUtils.STR_KEY_ID, StringKeyUtils.STR_KEY_NODE_ID,
                 StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_POSITION, StringKeyUtils.STR_KEY_LINE,
                 StringKeyUtils.STR_KEY_PATH, StringKeyUtils.STR_KEY_CREATE_AT, StringKeyUtils.STR_KEY_UPDATE_AT,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, StringKeyUtils.STR_KEY_BODY]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_COMMIT_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_POSITION, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_LINE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_PATH, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BODY, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_COMMIT_ID: self.commit_id, StringKeyUtils.STR_KEY_ID: self.id,
                 StringKeyUtils.STR_KEY_NODE_ID: self.node_id, StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login,
                 StringKeyUtils.STR_KEY_POSITION: self.position, StringKeyUtils.STR_KEY_LINE: self.line,
                 StringKeyUtils.STR_KEY_PATH: self.path, StringKeyUtils.STR_KEY_CREATE_AT: self.created_at,
                 StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION: self.author_association,
                 StringKeyUtils.STR_KEY_BODY: self.body}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = CommitComment()
                res.commit_id = src.get(StringKeyUtils.STR_KEY_COMMIT_ID, None)
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                res.position = src.get(StringKeyUtils.STR_KEY_POSITION, None)
                res.line = src.get(StringKeyUtils.STR_KEY_LINE, None)
                res.path = src.get(StringKeyUtils.STR_KEY_PATH, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)
                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, None)
                res.body = src.get(StringKeyUtils.STR_KEY_BODY, None)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    res.user = User.parser.parser(userData)
                    res.user_login = res.user.login
            return res
