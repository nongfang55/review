#coding=gbk
from source.data.bean.Beanbase import BeanBase

class DataInterceptor:
    
    ''' 用于数据类和具体实现做转换的类'''
    
    @staticmethod
    def convertFromBeanbaseToOutput(bean):
        if(not isinstance(bean, BeanBase)):
            return None #错误类型转换失败
        
    @staticmethod
    def convertFromOutputToBeanBase(bean):
        if(not isinstance(bean, BeanBase)):
            return None #错误类型转换失败
            