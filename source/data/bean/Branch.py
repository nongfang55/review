# coding=gbk
from source.data.bean.Repository import Repository
from source.data.bean.Beanbase import BeanBase
from source.data.bean.User import User
from source.utils.StringKeyUtils import StringKeyUtils


class Branch(BeanBase):
    """github中的分支数据类"""

    def __init__(self):
        self.label = None
        self.ref = None
        self.user = None
        self.repo = None
        self.sha = None
        self.user_login = None
        self.repo_full_name = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_LABEL]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_LABEL, StringKeyUtils.STR_KEY_REF, StringKeyUtils.STR_KEY_USER_LOGIN,
                 StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_SHA]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_LABEL, BeanBase.DATA_TYPE_STRING),
                  (StringKeyUtils.STR_KEY_REF, BeanBase.DATA_TYPE_STRING),
                  (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                  (StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                  (StringKeyUtils.STR_KEY_SHA, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_LABEL: self.label, StringKeyUtils.STR_KEY_REF: self.ref,
                 StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login,
                 StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,
                 StringKeyUtils.STR_KEY_SHA: self.sha}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Branch()
                res.label = src.get(StringKeyUtils.STR_KEY_LABEL, None)
                res.ref = src.get(StringKeyUtils.STR_KEY_REF, None)
                res.sha = src.get(StringKeyUtils.STR_KEY_SHA, None)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    res.user = User.parser.parser(userData)
                    res.user_login = res.user.login

                repoData = src.get(StringKeyUtils.STR_KEY_REPO, None)
                if repoData is not None and isinstance(repoData, dict):
                    res.repo = Repository.parser.parser(repoData)
                    res.repo_full_name = res.repo.full_name

            return res

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                """部分信息不在解析json中  需要在上级补全"""
                res = Branch()
                res.ref = src.get(StringKeyUtils.STR_KEY_NAME, None)
                """sha 需要在上级补全"""
                res.sha = None
                res.repo = None

                repoData = src.get(StringKeyUtils.STR_KEY_REPOSITORY, None)
                if repoData is not None and isinstance(repoData, dict):
                    res.repo_full_name = repoData.get(StringKeyUtils.STR_KEY_NAME_WITH_OWNER, None)
                    """user 字段不解析"""
                    res.user = None
                    if res.repo_full_name is not None:
                        res.user_login = res.repo_full_name.split('/')[0]

                """label 需要后期由user_login和ref拼接出来"""
                if res.user_login is not None and res.ref is not None:
                    res.label = res.user_login + ':' + res.ref

            return res

