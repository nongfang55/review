# coding=gbk
import numpy
import pandas
from pandas import DataFrame

from source.config.configPraser import configPraser
from source.data.bean.Beanbase import BeanBase


class BeanNumpyHelper:
    """对numpy和dataBean做转换的工具类"""

    @staticmethod
    def getBeansFromDataFrame(beanClass, columns, dataFrame):
        result = []
        beanIndex = {}  # 字典用于做主键映射
        # if configPraser.getPrintMode():
        #     print(columns)
        if isinstance(dataFrame, DataFrame) and isinstance(beanClass, BeanBase):
            matrix = dataFrame.as_matrix()
            [rows, cols] = matrix.shape
            for i in range(rows):
                """采用反射机制实例化对象"""
                obj = beanClass.__class__()
                itemList = obj.getItemKeyList()
                for item in itemList:
                    value = BeanNumpyHelper.findItemInArray(item, columns, matrix[i, :])
                    if value is not None:
                        setattr(obj, item, value)

                mapKey = BeanNumpyHelper.getBeanIdentifyTuple(obj)
                if mapKey is not None:
                    if beanIndex.get(mapKey, None) is None:
                        beanIndex[mapKey] = result.__len__()
                        result.append(obj)
                # if configPraser.getPrintMode():
                #     print(obj.getValueDict())
            # if configPraser.getPrintMode():
            #     print(result.__len__())
            #     print(beanIndex)
        return result, beanIndex

    @staticmethod
    def findItemInArray(item, columns, array):
        index = -1
        try:
            index = columns.index(item)
        except Exception as e:
            pass
        if index == -1:
            return None
        value = array[index]
        if pandas.isna(value):
            return None
        else:
            return value

    @staticmethod
    def getBeanIdentifyTuple(bean):
        res = None
        if isinstance(bean, BeanBase):
            valuesDict = bean.getValueDict()
            identifyKeys = bean.getIdentifyKeys()

            res = ()
            for item in identifyKeys:
                res = res + (valuesDict[item],)
        if configPraser.getPrintMode():
            print(res)
        return res

    @staticmethod
    def beanAssociate(BeanList1, KeyList1, BeanList2, KeyList2):
        """把数据之间通过key联系起来做一个索引， 从1来做2的索引
           索引结构  tuple -> list  索引的tuple是标识符的元组，
           而输入的两个key是自己定义的列表，按列表顺序去找项对比"""

        if KeyList1.__len__() != KeyList2.__len__():
            return None

        index = {}

        for bean2 in BeanList2:
            identifyTuple2 = BeanNumpyHelper.getBeanIdentifyTuple(bean2)
            for bean1 in BeanList1:
                pos = 0
                success = True
                for key2 in KeyList2:
                    value2 = bean2.getValueDict().get(key2, None)
                    value1 = bean1.getValueDict().get(KeyList1[pos], None)
                    if value2 == value1:
                        pos += 1
                    else:
                        success = False
                        break
                if success:
                    identifyTuple1 = BeanNumpyHelper.getBeanIdentifyTuple(bean1)
                    if index.get(identifyTuple1, None) is None:
                        index[identifyTuple1] = [identifyTuple2]
                    else:
                        index[identifyTuple1].append(identifyTuple2)
        return index


