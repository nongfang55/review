#coding=gbk
from source.utils.DataInterceptor import DataInterceptor
from source.data.bean.Beanbase import BeanBase

class SqlServerInterceptor(DataInterceptor):
    '''dataInterceptor的实现类'''
    
    @staticmethod
    def convertFromBeanbaseToOutput(bean):
        
        if(not isinstance(bean, BeanBase)):
            return None #错误类型转换失败
        
        #sqlserver中bit存储和bool类型需要转换
        
        for item in bean.getItemKeyListWithType():
            if(item[1] == BeanBase.DATA_TYPE_BOOLEAN):
                if(getattr(bean,item[0], None) == True):
                    setattr(bean,item[0], 1)
                elif(getattr(bean, item[0], None) == False):
                    setattr(bean,item[0],0)
        
        return bean
        