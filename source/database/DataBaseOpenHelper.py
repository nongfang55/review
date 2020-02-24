# coding=gbk
import pymssql
from source.config.configPraser import configPraser


class DataBaseOpenHelper:
    '''用于连接数据库接口类'''

    @staticmethod
    def connect():
        conn = pymssql.connect(configPraser.getDataBaseHost(),
                               configPraser.getDataBaseUserName(),
                               configPraser.getDataBasePassword())
        if conn:
            if configPraser.getPrintMode():
                print('数据库连接成功，host:', configPraser.getDataBaseHost(), ' user:', configPraser.getDataBaseUserName())
        return conn


if __name__ == '__main__':
    conn = DataBaseOpenHelper.connect()
