# coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils
import pandas


class pandasHelper:
    """  pandas接口分装工具类 """

    INT_READ_FILE_WITHOUT_HEAD = None
    INT_READ_FILE_WITH_HEAD = 0

    @staticmethod
    def readTSVFile(fileName, header=INT_READ_FILE_WITHOUT_HEAD, sep=StringKeyUtils.STR_SPLIT_SEP_TSV):  # 负一为无表头
        train = pandas.read_csv(fileName, sep=sep, header=header)
        return train

    @staticmethod
    def toDataFrame(data, columns=None, dtype=None):
        return pandas.DataFrame(data, columns=columns, dtype=dtype)

    @staticmethod
    def writeTSVFile(fileName, dataFrame):  # 写入tsv文件
        with open(fileName, 'w', encoding='utf-8') as write_tsv:
            print(fileName)
            write_tsv.write(dataFrame.to_csv(sep=StringKeyUtils.STR_SPLIT_SEP_TSV, index=False))
