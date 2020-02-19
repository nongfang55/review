#coding=gbk

class SqlUtils:
    '''用于存储各SQL语句'''
    
    STR_SQL_CREATE_TABLE = 'create table %s'
    
    
    '''预计存储的表名字'''
    STR_TABLE_NAME_REPOS = 'repos'
    
    
    '''存储的表中的类型'''
    STR_KEY_INT = 'int'
    STR_KEY_VARCHAR_MAX = 'varchar(8000)'
    STR_KEY_VARCHAR_MIDDLE = 'varchar(5000)'
    STR_KEY_DATE_TIME = 'datatime'
    STR_KEY_TEXT = 'text'
    
    
    '''预计存储的表的key'''
    TABLE_REPOS_ITEMS_LIST = [()]
    