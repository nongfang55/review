#coding=gbk
from source.database.DataBaseOpenHelper import DataBaseOpenHelper
from source.database.SqlUtils import SqlUtils

class SqlExecuteHelper:
    '''用来执行sql语句'''
    
    
    @staticmethod
    def insertValuesIntoTable(tableName, items, valueDict, primaryKeys = None):
        '''插入语句'''
        
        res = SqlExecuteHelper.queryValuesFromTable(tableName, primaryKeys, valueDict)
        if(res != None and res.__len__()> 0):
            print('数据重复插入失败')
            return
        
        conn = DataBaseOpenHelper.connect() 
        cursor = conn.cursor()
        format_table = SqlUtils.getInsertTableFormatString(tableName, items)
        format_values = SqlUtils.getInsertTableValuesString(items.__len__())
        print(format_table)
        print(format_values)
        sql = SqlUtils.STR_SQL_INSERT_TABLE_UTILS.format(format_table,format_values)
        print(sql)
        
        values = ()
        for item in items:
            values = values + (valueDict.get(item,None),) #元组相加
        try:
            cursor.execute(sql,values)
            conn.commit()
        except Exception as e:
            print(e)
            conn.rollback()
        conn.close()
        
    @staticmethod
    def queryValuesFromTable(tableName, items, valueDict):
        '''查询数据库'''
        
        ret = []
        conn = DataBaseOpenHelper.connect()
        
        cursor = conn.cursor()
        format_values = SqlUtils.getQueryTableConditionString(items)
        sql =SqlUtils.STR_SQL_QUERY_TABLE_UTILS.format(tableName, format_values)
        print(sql)
          
        values = ()
        if(items != None):
            for item in items:
                values = values + (valueDict.get(item,None),) #元组相加
        try:
            cursor.execute(sql,values)
            ret = cursor.fetchall()
            print(ret)
        except Exception as e:
            print(e)
        conn.close()
        return ret
        
        
if __name__ == '__main__':
    SqlExecuteHelper().queryValuesFromTable('repository',['id','node_id'], {'id':8514,'node_id':'MDEwOlJlcG9zaXRvcnk4NTE0'})
        
        