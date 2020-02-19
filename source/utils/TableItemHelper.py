#coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils

class TableItemHelper:
    '''用于展现在excel 表和数据库中不同的数据的存储数据项
    '''
    
    @staticmethod
    def getProjectTableItem():
        
        items = []
        items.append(StringKeyUtils.STR_KEY_ID)
        items.append(StringKeyUtils.STR_KEY_NODE_ID)
        items.append(StringKeyUtils.STR_KEY_NAME)
        items.append(StringKeyUtils.STR_KEY_FULL_NAME)
        items.append(StringKeyUtils.STR_KEY_OWNER)
        items.append(StringKeyUtils.STR_KEY_DESCRIPTION)
        items.append(StringKeyUtils.STR_KEY_CREATE_AT)
        items.append(StringKeyUtils.STR_KEY_UPDATE_AT)
        items.append(StringKeyUtils.STR_KEY_STARGAZERS_COUNT)
        items.append(StringKeyUtils.STR_KEY_WATCHERS_COUNT)
        items.append(StringKeyUtils.STR_KEY_LANG)
        items.append(StringKeyUtils.STR_KEY_FORKS_COUNT)
        items.append(StringKeyUtils.STR_KEY_SUBSCRIBERS_COUNT)
        
        return items



if __name__ == '__main__':
    print(TableItemHelper.getProjectTableItem())
        