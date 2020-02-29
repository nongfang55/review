# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User


class IssueComment(BeanBase):
    """github 中issue comment数据类  9项"""

    def __init__(self):
        self.repo_full_name = None
        self.pull_number = None
        self.id = None
        self.node_id = None
        self.user = None
        self.created_at = None
        self.updated_at = None
        self.author_association = None
        self.body = None

        self.user_login = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME,
                StringKeyUtils.STR_KEY_PULL_NUMBER,
                StringKeyUtils.STR_KEY_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER, StringKeyUtils.STR_KEY_ID,
                 StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_CREATE_AT,
                 StringKeyUtils.STR_KEY_UPDATE_AT, StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION,
                 StringKeyUtils.STR_KEY_BODY]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PULL_NUMBER, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BODY, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,
                 StringKeyUtils.STR_KEY_PULL_NUMBER: self.pull_number, StringKeyUtils.STR_KEY_ID: self.id,
                 StringKeyUtils.STR_KEY_NODE_ID: self.node_id, StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login,
                 StringKeyUtils.STR_KEY_CREATE_AT: self.created_at, StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION: self.author_association,
                 StringKeyUtils.STR_KEY_BODY: self.body}

        return items

    class parser(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = IssueComment()
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, None)
                res.body = src.get(StringKeyUtils.STR_KEY_BODY, None)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    user = User.parser.parser(userData)
                    res.user = user
                    res.user_login = user.login
            return res


