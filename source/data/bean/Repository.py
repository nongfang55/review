#coding=gbk

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User

class Respository(BeanBase):
    '''项目数据类'''
    
    '''数据项'''
    
    def __init__(self):
        self.id = None
        self.node_id = None
        self.name = None
        self.full_name = None
        self.owner = None #owner为用户类
        self.owner_id = None
        self.description = None
        self.created_at = None
        self.updated_at = None
        self.stargazers_count = None
        self.watchers_count = None
        self.language = None
        self.forks_count = None
        self.subscribers_count = None
    
    @staticmethod
    def getItemKeyList():
        
        items = []
        items.append(StringKeyUtils.STR_KEY_ID)
        items.append(StringKeyUtils.STR_KEY_NODE_ID)
        items.append(StringKeyUtils.STR_KEY_NAME)
        items.append(StringKeyUtils.STR_KEY_FULL_NAME)
        #此处没有owner
        items.append(StringKeyUtils.STR_KEY_OWNER_ID)
        items.append(StringKeyUtils.STR_KEY_DESCRIPTION)
        items.append(StringKeyUtils.STR_KEY_CREATE_AT)
        items.append(StringKeyUtils.STR_KEY_UPDATE_AT)
        items.append(StringKeyUtils.STR_KEY_STARGAZERS_COUNT)
        items.append(StringKeyUtils.STR_KEY_WATCHERS_COUNT)
        items.append(StringKeyUtils.STR_KEY_LANG)
        items.append(StringKeyUtils.STR_KEY_FORKS_COUNT)
        items.append(StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT)
        
        return items
    
    @staticmethod
    def getItemKeyListWithType():
        
        items = []
        items.append((StringKeyUtils.STR_KEY_ID,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_NODE_ID,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_NAME,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_FULL_NAME,BeanBase.DATA_TYPE_STRING))
        #此处没有owner
        items.append((StringKeyUtils.STR_KEY_OWNER_ID,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_DESCRIPTION,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_CREATE_AT,BeanBase.DATA_TYPE_DATE_TIME))
        items.append((StringKeyUtils.STR_KEY_UPDATE_AT,BeanBase.DATA_TYPE_DATE_TIME))
        items.append((StringKeyUtils.STR_KEY_STARGAZERS_COUNT,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_WATCHERS_COUNT,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_LANG,BeanBase.DATA_TYPE_STRING))
        items.append((StringKeyUtils.STR_KEY_FORKS_COUNT,BeanBase.DATA_TYPE_INT))
        items.append((StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT,BeanBase.DATA_TYPE_INT))
        
        return items

    
    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_ID]

    
    
    def getValueDict(self):
        
        items = {}
        
        items[StringKeyUtils.STR_KEY_ID] = self.id
        items[StringKeyUtils.STR_KEY_NODE_ID] = self.node_id
        items[StringKeyUtils.STR_KEY_NAME] = self.name
        items[StringKeyUtils.STR_KEY_FULL_NAME] = self.full_name
        #此处没有owner
        items[StringKeyUtils.STR_KEY_OWNER_ID] = self.owner_id
        items[StringKeyUtils.STR_KEY_DESCRIPTION] = self.description
        items[StringKeyUtils.STR_KEY_CREATE_AT] = self.created_at
        items[StringKeyUtils.STR_KEY_UPDATE_AT] = self.updated_at
        items[StringKeyUtils.STR_KEY_STARGAZERS_COUNT] = self.stargazers_count
        items[StringKeyUtils.STR_KEY_WATCHERS_COUNT] = self.watchers_count
        items[StringKeyUtils.STR_KEY_LANG] = self.language
        items[StringKeyUtils.STR_KEY_FORKS_COUNT] = self.forks_count
        items[StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT] = self.subscribers_count
        
        return items
        

    class parser(BeanBase.parser):
        '''用于json的解析器'''
        
        @staticmethod
        def parser(src):
            res = None
            if(isinstance(src,dict)):
                res = Respository()
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                res.name = src.get(StringKeyUtils.STR_KEY_NAME, None)
                res.full_name = src.get(StringKeyUtils.STR_KEY_FULL_NAME, None)
                
                res.description = src.get(StringKeyUtils.STR_KEY_DESCRIPTION, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)
                res.stargazers_count = src.get(StringKeyUtils.STR_KEY_STARGAZERS_COUNT, None)
                res.watchers_count = src.get(StringKeyUtils.STR_KEY_WATCHERS_COUNT, None)
                res.language = src.get(StringKeyUtils.STR_KEY_LANG, None)
                res.forks_count = src.get(StringKeyUtils.STR_KEY_FORKS_COUNT, None)
                res.subscribers_count = src.get(StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT, None)
                
                userData = src.get(StringKeyUtils.STR_KEY_OWNER, None)
                if(userData != None and isinstance(userData,dict)):
                    user = User.parser.parser(userData)
                    res.owner = user
                    res.owner_id = user.id
                
            return res
        
            
        
        
        