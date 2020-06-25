# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User


class Review(BeanBase):
    """github中review的数据类 共计11个"""

    def __init__(self):
        self.repo_full_name = None
        self.pull_number = None
        self.id = None
        self.user = None
        self.body = None
        self.state = None
        self.author_association = None
        self.submitted_at = None
        self.commit_id = None
        self.node_id = None

        self.user_login = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME,
                StringKeyUtils.STR_KEY_PULL_NUMBER, StringKeyUtils.STR_KEY_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_PULL_NUMBER, StringKeyUtils.STR_KEY_ID,
                 StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_BODY, StringKeyUtils.STR_KEY_STATE,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, StringKeyUtils.STR_KEY_SUBMITTED_AT,
                 StringKeyUtils.STR_KEY_COMMIT_ID, StringKeyUtils.STR_KEY_NODE_ID]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PULL_NUMBER, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BODY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_STATE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_SUBMITTED_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_COMMIT_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,
                 StringKeyUtils.STR_KEY_PULL_NUMBER: self.pull_number, StringKeyUtils.STR_KEY_ID: self.id,
                 StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login, StringKeyUtils.STR_KEY_BODY: self.body,
                 StringKeyUtils.STR_KEY_STATE: self.state,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION: self.author_association,
                 StringKeyUtils.STR_KEY_SUBMITTED_AT: self.submitted_at,
                 StringKeyUtils.STR_KEY_COMMIT_ID: self.commit_id, StringKeyUtils.STR_KEY_NODE_ID: self.node_id}
        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Review()
                res.repo_full_name = src.get(StringKeyUtils.STR_KEY_REPO_FULL_NAME, None)
                res.pull_number = src.get(StringKeyUtils.STR_KEY_PULL_NUMBER, None)
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)

                res.body = src.get(StringKeyUtils.STR_KEY_BODY, None)
                res.state = src.get(StringKeyUtils.STR_KEY_STATE, None)
                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, None)
                res.submitted_at = src.get(StringKeyUtils.STR_KEY_SUBMITTED_AT, None)

                if res.submitted_at is not None:
                    res.submitted_at = datetime.strptime(res.submitted_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.commit_id = src.get(StringKeyUtils.STR_KEY_COMMIT_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    user = User.parser.parser(userData)
                    res.user = user
                    res.user_login = user.login
            return res

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Review()
                """repo_full_name, pull_number 无法获取"""
                res.repo_full_name = src.get(StringKeyUtils.STR_KEY_REPO_FULL_NAME, None)
                res.pull_number = src.get(StringKeyUtils.STR_KEY_PULL_NUMBER, None)

                res.id = src.get(StringKeyUtils.STR_KEY_DATABASE_ID, None)

                res.body = src.get(StringKeyUtils.STR_KEY_BODY, None)
                res.state = src.get(StringKeyUtils.STR_KEY_STATE, None)
                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION_V4, None)
                res.submitted_at = src.get(StringKeyUtils.STR_KEY_SUBMITTED_AT_V4, None)

                if res.submitted_at is not None:
                    res.submitted_at = datetime.strptime(res.submitted_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                """获取 review的commit sha"""
                commit = src.get(StringKeyUtils.STR_KEY_COMMIT, None)
                if commit is not None and isinstance(commit, dict):
                    res.commit_id = commit.get(StringKeyUtils.STR_KEY_OID, None)

                res.node_id = src.get(StringKeyUtils.STR_KEY_ID, None)

                """获取 user_login"""
                userData = src.get(StringKeyUtils.STR_KEY_AUTHOR, None)
                if userData is not None and isinstance(userData, dict):
                    res.user = None
                    res.user_login = userData.get(StringKeyUtils.STR_KEY_LOGIN, None)

            return res
