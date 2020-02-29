# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class User(BeanBase):
    """ github用户类数据  共计26个

    """

    def __init__(self):
        self.login = None
        self.site_admin = None
        self.type = None
        self.id = None
        self.email = None
        self.node_id = None
        self.followers_url = None
        self.following_url = None
        self.starred_url = None
        self.subscriptions_url = None
        self.organizations_url = None
        self.repos_url = None
        self.events_url = None
        self.received_events_url = None
        self.name = None
        self.company = None
        self.blog = None
        self.location = None
        self.hireable = None
        self.bio = None
        self.public_repos = None
        self.public_gists = None
        self.followers = None
        self.following = None
        self.created_at = None
        self.updated_at = None

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_LOGIN, StringKeyUtils.STR_KEY_SITE_ADMIN, StringKeyUtils.STR_KEY_TYPE,
                 StringKeyUtils.STR_KEY_ID, StringKeyUtils.STR_KEY_EMAIL, StringKeyUtils.STR_KEY_NODE_ID,
                 StringKeyUtils.STR_KEY_FOLLOWERS_URL, StringKeyUtils.STR_KEY_FOLLOWING_URL,
                 StringKeyUtils.STR_KEY_STARRED_URL, StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL,
                 StringKeyUtils.STR_KEY_ORGANIZATIONS_URL, StringKeyUtils.STR_KEY_REPOS_URL,
                 StringKeyUtils.STR_KEY_EVENTS_URL, StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL,
                 StringKeyUtils.STR_KEY_NAME, StringKeyUtils.STR_KEY_COMPANY, StringKeyUtils.STR_KEY_BLOG,
                 StringKeyUtils.STR_KEY_LOCATION, StringKeyUtils.STR_KEY_HIREABLE, StringKeyUtils.STR_KEY_BIO,
                 StringKeyUtils.STR_KEY_PUBLIC_REPOS, StringKeyUtils.STR_KEY_PUBLIC_GISTS,
                 StringKeyUtils.STR_KEY_FOLLOWERS, StringKeyUtils.STR_KEY_FOLLOWING, StringKeyUtils.STR_KEY_CREATE_AT,
                 StringKeyUtils.STR_KEY_UPDATE_AT]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_SITE_ADMIN, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_TYPE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_EMAIL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_FOLLOWERS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_FOLLOWING_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_STARRED_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_ORGANIZATIONS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_REPOS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_EVENTS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_COMPANY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BLOG, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_LOCATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_HIREABLE, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_BIO, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PUBLIC_REPOS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_PUBLIC_GISTS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_FOLLOWERS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_FOLLOWING, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_LOGIN: self.login, StringKeyUtils.STR_KEY_SITE_ADMIN: self.site_admin,
                 StringKeyUtils.STR_KEY_TYPE: self.type, StringKeyUtils.STR_KEY_ID: self.id,
                 StringKeyUtils.STR_KEY_EMAIL: self.email, StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_FOLLOWERS_URL: self.followers_url,
                 StringKeyUtils.STR_KEY_FOLLOWING_URL: self.following_url,
                 StringKeyUtils.STR_KEY_STARRED_URL: self.starred_url,
                 StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL: self.subscriptions_url,
                 StringKeyUtils.STR_KEY_ORGANIZATIONS_URL: self.organizations_url,
                 StringKeyUtils.STR_KEY_REPOS_URL: self.repos_url, StringKeyUtils.STR_KEY_EVENTS_URL: self.events_url,
                 StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL: self.received_events_url,
                 StringKeyUtils.STR_KEY_NAME: self.name, StringKeyUtils.STR_KEY_COMPANY: self.company,
                 StringKeyUtils.STR_KEY_BLOG: self.blog, StringKeyUtils.STR_KEY_LOCATION: self.location,
                 StringKeyUtils.STR_KEY_HIREABLE: self.hireable, StringKeyUtils.STR_KEY_BIO: self.bio,
                 StringKeyUtils.STR_KEY_PUBLIC_REPOS: self.public_repos,
                 StringKeyUtils.STR_KEY_PUBLIC_GISTS: self.public_gists,
                 StringKeyUtils.STR_KEY_FOLLOWERS: self.followers, StringKeyUtils.STR_KEY_FOLLOWING: self.following,
                 StringKeyUtils.STR_KEY_CREATE_AT: self.created_at, StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at}

        return items

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_LOGIN]

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = User()
                res.login = src.get(StringKeyUtils.STR_KEY_LOGIN, None)
                res.site_admin = src.get(StringKeyUtils.STR_KEY_SITE_ADMIN, None)
                res.type = src.get(StringKeyUtils.STR_KEY_TYPE, None)
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.email = src.get(StringKeyUtils.STR_KEY_EMAIL, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)

                res.followers_url = src.get(StringKeyUtils.STR_KEY_FOLLOWERS_URL, None)
                res.following_url = src.get(StringKeyUtils.STR_KEY_FOLLOWING_URL, None)
                res.starred_url = src.get(StringKeyUtils.STR_KEY_STARRED_URL, None)
                res.subscriptions_url = src.get(StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL, None)
                res.organizations_url = src.get(StringKeyUtils.STR_KEY_ORGANIZATIONS_URL, None)
                res.repos_url = src.get(StringKeyUtils.STR_KEY_REPOS_URL, None)
                res.events_url = src.get(StringKeyUtils.STR_KEY_EVENTS_URL, None)
                res.received_events_url = src.get(StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL, None)
                res.name = src.get(StringKeyUtils.STR_KEY_NAME, None)
                res.company = src.get(StringKeyUtils.STR_KEY_COMPANY, None)

                res.blog = src.get(StringKeyUtils.STR_KEY_BLOG, None)
                res.location = src.get(StringKeyUtils.STR_KEY_LOCATION, None)
                res.hireable = src.get(StringKeyUtils.STR_KEY_HIREABLE, None)
                res.bio = src.get(StringKeyUtils.STR_KEY_BIO, None)
                res.public_repos = src.get(StringKeyUtils.STR_KEY_PUBLIC_REPOS, None)
                res.public_gists = src.get(StringKeyUtils.STR_KEY_PUBLIC_GISTS, None)
                res.followers = src.get(StringKeyUtils.STR_KEY_FOLLOWERS, None)
                res.following = src.get(StringKeyUtils.STR_KEY_FOLLOWING, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)

            return res
