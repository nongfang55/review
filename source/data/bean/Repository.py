# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User


class Repository(BeanBase):
    """项目数据类"""

    '''数据项'''

    def __init__(self):
        self.id = None
        self.node_id = None
        self.name = None
        self.full_name = None
        self.owner = None  # owner为用户类
        self.owner_login = None
        self.description = None
        self.created_at = None
        self.updated_at = None
        self.stargazers_count = None
        self.watchers_count = None
        self.language = None
        self.forks_count = None
        self.subscribers_count = None
        self.parent_full_name = None

    @staticmethod
    def getItemKeyList():

        items = [StringKeyUtils.STR_KEY_ID, StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_NAME,
                 StringKeyUtils.STR_KEY_FULL_NAME, StringKeyUtils.STR_KEY_OWNER_LOGIN, StringKeyUtils.STR_KEY_DESCRIPTION,
                 StringKeyUtils.STR_KEY_CREATE_AT, StringKeyUtils.STR_KEY_UPDATE_AT,
                 StringKeyUtils.STR_KEY_STARGAZERS_COUNT, StringKeyUtils.STR_KEY_WATCHERS_COUNT,
                 StringKeyUtils.STR_KEY_LANG, StringKeyUtils.STR_KEY_FORKS_COUNT,
                 StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT, StringKeyUtils.STR_KEY_PARENT_FULL_NAME]
        # 此处没有owner

        return items

    @staticmethod
    def getItemKeyListWithType():

        items = [(StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_OWNER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_DESCRIPTION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_STARGAZERS_COUNT, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_WATCHERS_COUNT, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_LANG, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_FORKS_COUNT, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_PARENT_FULL_NAME, BeanBase.DATA_TYPE_STRING)]
        # 此处没有owner

        return items

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_FULL_NAME]

    def getValueDict(self):

        items = {StringKeyUtils.STR_KEY_ID: self.id, StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_NAME: self.name, StringKeyUtils.STR_KEY_FULL_NAME: self.full_name,
                 StringKeyUtils.STR_KEY_OWNER_LOGIN: self.owner_login, StringKeyUtils.STR_KEY_DESCRIPTION: self.description,
                 StringKeyUtils.STR_KEY_CREATE_AT: self.created_at, StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at,
                 StringKeyUtils.STR_KEY_STARGAZERS_COUNT: self.stargazers_count,
                 StringKeyUtils.STR_KEY_WATCHERS_COUNT: self.watchers_count, StringKeyUtils.STR_KEY_LANG: self.language,
                 StringKeyUtils.STR_KEY_FORKS_COUNT: self.forks_count,
                 StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT: self.subscribers_count,
                 StringKeyUtils.STR_KEY_PARENT_FULL_NAME: self.parent_full_name}

        # 此处没有owner

        return items

    class parser(BeanBase.parser):
        """用于json的解析器"""

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Repository()
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                res.name = src.get(StringKeyUtils.STR_KEY_NAME, None)
                res.full_name = src.get(StringKeyUtils.STR_KEY_FULL_NAME, None)

                res.description = src.get(StringKeyUtils.STR_KEY_DESCRIPTION, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.stargazers_count = src.get(StringKeyUtils.STR_KEY_STARGAZERS_COUNT, None)
                res.watchers_count = src.get(StringKeyUtils.STR_KEY_WATCHERS_COUNT, None)
                res.language = src.get(StringKeyUtils.STR_KEY_LANG, None)
                res.forks_count = src.get(StringKeyUtils.STR_KEY_FORKS_COUNT, None)
                res.subscribers_count = src.get(StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT, None)

                userData = src.get(StringKeyUtils.STR_KEY_OWNER, None)
                if userData is not None and isinstance(userData, dict):
                    user = User.parser.parser(userData)
                    res.owner = user
                    res.owner_login = user.login

                parentData = src.get(StringKeyUtils.STR_KEY_PARENT, None) # 仓库的母仓库信息
                if parentData is not None and isinstance(parentData, dict):
                    parent = Repository.parser.parser(parentData) # 防止过度嵌套 这里没有存储母仓库信息
                    res.parent_full_name = parent.full_name

            return res
