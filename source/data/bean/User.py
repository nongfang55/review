#coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
class User(BeanBase):
    ''' github用户类数据 
     
    '''
    def __init__(self):
        self.login = None
        self.site_admin = None
        self.type = None
        self.id = None
        self.email = None
        self.node_id = None
    
    @staticmethod
    def getItemKeyList():
        items = []
        
        items.append(StringKeyUtils.STR_KEY_LOGIN)
        items.append(StringKeyUtils.STR_KEY_SITE_ADMIN)
        items.append(StringKeyUtils.STR_KEY_TYPE)
        items.append(StringKeyUtils.STR_KEY_ID)
        items.append(StringKeyUtils.STR_KEY_EMAIL)
        items.append(StringKeyUtils.STR_KEY_NODE_ID)
        
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
        
        return items
    
    def getValueDict(self):
        items = {}
        
        items[StringKeyUtils.STR_KEY_LOGIN] = self.login
        items[StringKeyUtils.STR_KEY_SITE_ADMIN] = self.site_admin
        items[StringKeyUtils.STR_KEY_TYPE] = self.type
        items[StringKeyUtils.STR_KEY_ID] = self.id
        items[StringKeyUtils.STR_KEY_EMAIL] = self.email
        items[StringKeyUtils.STR_KEY_NODE_ID] = self.node_id
        
        return items
    
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
                
            return res
        
        