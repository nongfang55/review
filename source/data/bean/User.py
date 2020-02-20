#coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
class User(BeanBase):
    ''' github用户类数据  共计26个 
     
    '''
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
        items = []
        
        items.append(StringKeyUtils.STR_KEY_LOGIN)
        items.append(StringKeyUtils.STR_KEY_SITE_ADMIN)
        items.append(StringKeyUtils.STR_KEY_TYPE)
        items.append(StringKeyUtils.STR_KEY_ID)
        items.append(StringKeyUtils.STR_KEY_EMAIL)
        items.append(StringKeyUtils.STR_KEY_NODE_ID)
        items.append(StringKeyUtils.STR_KEY_FOLLOWERS_URL)
        items.append(StringKeyUtils.STR_KEY_FOLLOWING_URL)
        items.append(StringKeyUtils.STR_KEY_STARRED_URL)
        items.append(StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL)
        items.append(StringKeyUtils.STR_KEY_ORGANIZATIONS_URL)
        items.append(StringKeyUtils.STR_KEY_REPOS_URL)
        items.append(StringKeyUtils.STR_KEY_EVENTS_URL)
        items.append(StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL)
        items.append(StringKeyUtils.STR_KEY_NAME)
        items.append(StringKeyUtils.STR_KEY_COMPANY)
        items.append(StringKeyUtils.STR_KEY_BLOG)
        items.append(StringKeyUtils.STR_KEY_LOCATION)
        items.append(StringKeyUtils.STR_KEY_HIREABLE)
        items.append(StringKeyUtils.STR_KEY_BIO)
        items.append(StringKeyUtils.STR_KEY_PUBLIC_REPOS)
        items.append(StringKeyUtils.STR_KEY_PUBLIC_GISTS)
        items.append(StringKeyUtils.STR_KEY_FOLLOWERS)
        items.append(StringKeyUtils.STR_KEY_FOLLOWING)
        items.append(StringKeyUtils.STR_KEY_CREATE_AT)
        items.append(StringKeyUtils.STR_KEY_UPDATE_AT)
        
        return items
    
    @staticmethod
    def getItemKeyListWithType():
        items = []
        
        items.append((StringKeyUtils.STR_KEY_LOGIN,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_SITE_ADMIN,BeanBase.DATA_TYPE_BOOLEAN))
        items.append((StringKeyUtils.STR_KEY_TYPE,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_ID,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_EMAIL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_NODE_ID,BeanBase.DATA_TYPE_STRING))
        
        items.append((StringKeyUtils.STR_KEY_FOLLOWERS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_FOLLOWING_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_STARRED_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_ORGANIZATIONS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_REPOS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_EVENTS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_NAME,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_COMPANY,BeanBase.DATA_TYPE_STRING))
        
        items.append((StringKeyUtils.STR_KEY_BLOG,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_LOCATION,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_HIREABLE,BeanBase.DATA_TYPE_BOOLEAN))
        items.append((StringKeyUtils.STR_KEY_BIO,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_PUBLIC_REPOS,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_PUBLIC_GISTS,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_FOLLOWERS,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_FOLLOWING,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_CREATE_AT,BeanBase.DATA_TYPE_DATE_TIME))
        items.append((StringKeyUtils.STR_KEY_UPDATE_AT,BeanBase.DATA_TYPE_DATE_TIME))
        
        return items
    
    def getValueDict(self):
        items = {}
        
        items[StringKeyUtils.STR_KEY_LOGIN] = self.login
        items[StringKeyUtils.STR_KEY_SITE_ADMIN] = self.site_admin
        items[StringKeyUtils.STR_KEY_TYPE] = self.type
        items[StringKeyUtils.STR_KEY_ID] = self.id
        items[StringKeyUtils.STR_KEY_EMAIL] = self.email
        items[StringKeyUtils.STR_KEY_NODE_ID] = self.node_id
        
        items[StringKeyUtils.STR_KEY_FOLLOWERS_URL] = self.followers_url
        items[StringKeyUtils.STR_KEY_FOLLOWING_URL] = self.following_url
        items[StringKeyUtils.STR_KEY_STARRED_URL] = self.starred_url
        items[StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL] = self.subscriptions_url
        items[StringKeyUtils.STR_KEY_ORGANIZATIONS_URL] = self.organizations_url
        items[StringKeyUtils.STR_KEY_REPOS_URL] = self.repos_url
        items[StringKeyUtils.STR_KEY_EVENTS_URL] = self.events_url
        items[StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL] = self.received_events_url
        items[StringKeyUtils.STR_KEY_NAME] = self.name
        items[StringKeyUtils.STR_KEY_COMPANY] = self.company
        
        items[StringKeyUtils.STR_KEY_BLOG] = self.blog
        items[StringKeyUtils.STR_KEY_LOCATION] = self.location
        items[StringKeyUtils.STR_KEY_HIREABLE] = self.hireable
        items[StringKeyUtils.STR_KEY_BIO] = self.bio
        items[StringKeyUtils.STR_KEY_PUBLIC_REPOS] = self.public_repos
        items[StringKeyUtils.STR_KEY_PUBLIC_GISTS] = self.public_gists
        items[StringKeyUtils.STR_KEY_FOLLOWERS] = self.followers
        items[StringKeyUtils.STR_KEY_FOLLOWING] = self.following
        items[StringKeyUtils.STR_KEY_CREATE_AT] = self.created_at
        items[StringKeyUtils.STR_KEY_UPDATE_AT] = self.updated_at
        
        return items
    
    
    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_ID]
    
    class parser(BeanBase.parser):
        
        @staticmethod
        def parser(src):
            
            res = None
            if(isinstance(src,dict)):
                res = User()
                res.login = src.get(StringKeyUtils.STR_KEY_LOGIN, None)
                res.site_admin = src.get(StringKeyUtils.STR_KEY_SITE_ADMIN, None)
                res.type = src.get(StringKeyUtils.STR_KEY_TYPE, None)
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.email = src.get(StringKeyUtils.STR_KEY_EMAIL, None)
                res.node_id =src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                
                res.followers_url = src.get(StringKeyUtils.STR_KEY_FOLLOWERS_URL,None)
                res.following_url = src.get(StringKeyUtils.STR_KEY_FOLLOWING_URL,None)
                res.starred_url = src.get(StringKeyUtils.STR_KEY_STARRED_URL,None)
                res.subscriptions_url = src.get(StringKeyUtils.STR_KEY_SUBSCRIPTIONS_URL,None)
                res.organizations_url = src.get(StringKeyUtils.STR_KEY_ORGANIZATIONS_URL,None)
                res.repos_url = src.get(StringKeyUtils.STR_KEY_REPOS_URL,None)
                res.events_url = src.get(StringKeyUtils.STR_KEY_EVENTS_URL,None)
                res.received_events_url = src.get(StringKeyUtils.STR_KEY_RECEVIED_EVENTS_URL,None)
                res.name = src.get(StringKeyUtils.STR_KEY_NAME,None)
                res.company = src.get(StringKeyUtils.STR_KEY_COMPANY,None)
                
                res.blog = src.get(StringKeyUtils.STR_KEY_BLOG,None)
                res.location = src.get(StringKeyUtils.STR_KEY_LOCATION,None)
                res.hireable = src.get(StringKeyUtils.STR_KEY_HIREABLE,None)
                res.bio = src.get(StringKeyUtils.STR_KEY_BIO,None)
                res.public_repos = src.get(StringKeyUtils.STR_KEY_PUBLIC_REPOS,None)
                res.public_gists = src.get(StringKeyUtils.STR_KEY_PUBLIC_GISTS,None)
                res.followers = src.get(StringKeyUtils.STR_KEY_FOLLOWERS,None)
                res.following = src.get(StringKeyUtils.STR_KEY_FOLLOWING,None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT,None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT,None)
                
            return res
        
        