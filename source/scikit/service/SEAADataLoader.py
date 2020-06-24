import datetime
import json
import os

import pandas

from source.config.projectConfig import projectConfig
from source.scikit.service.DataProcessUtils import DataProcessUtils


def convertTimeStampToTime(timestamp):
    """把时间戳转化为当前对象 2018-07-02 06:32:11
    """
    timestamp = int(timestamp) / 1000
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class SEAADataLoader:
    """  用于处理加载   论文《A Large-Scale Study on Source Code Reviewer Recommendation》
       的数据集 包括项目json文件处理的分割

       数据只保存了原始数据  训练数据需要本地生成
    """

    SEAA_DATAFRAME_COL = ['repo_full_name', 'pull_number', 'pr_created_at',
                          'review_user_login', 'commit_sha', 'file_filename']

    @staticmethod
    def contactFPSData(projectName):
        """
        通过SEAA的数据拼接出FPS所用的数据集
        """
        filename = os.path.join(projectConfig.getSEAADataPath(), f'{projectName}.json')
        print(filename)
        file = open(filename, 'rb')
        dataJson = json.load(file)
        """遍历数据集 用字典方式生成数据"""
        dictList = []
        for data in dataJson:
            new_row = {'repo_full_name': projectName + '/' + projectName, 'pull_number': data['changeNumber'],
                       'pr_created_at': convertTimeStampToTime(data['timestamp']), 'commit_sha':0}
            for reviewer in data['reviewers']:
                reviewer = reviewer['name']
                for path in data['filePaths']:
                    path = path['location']
                    new_row_c = new_row.copy()
                    new_row_c['review_user_login'] = reviewer
                    new_row_c['file_filename'] = path
                    dictList.append(new_row_c)
        df = pandas.DataFrame(dictList, columns=SEAADataLoader.SEAA_DATAFRAME_COL)
        print(df.shape)
        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getFPSDataPath(),
                                          targetFileName=f'FPS_SEAA_{projectName}_data', dateCol='pr_created_at',
                                          dataFrame=df)

    @staticmethod
    def contactCAData(projectName):
        """
        通过SEAA的数据拼接出FPS所用的数据集
        """
        filename = os.path.join(projectConfig.getSEAADataPath(), f'{projectName}.json')
        print(filename)
        file = open(filename, 'rb')
        dataJson = json.load(file)
        """遍历数据集 用字典方式生成数据"""
        dictList = []
        for data in dataJson:
            new_row = {'repo_full_name': projectName + '/' + projectName, 'pull_number': data['changeNumber'],
                       'pr_created_at': convertTimeStampToTime(data['timestamp']), 'commit_sha':0}
            for reviewer in data['reviewers']:
                reviewer = reviewer['name']
                for path in data['filePaths']:
                    path = path['location']
                    new_row_c = new_row.copy()
                    new_row_c['review_user_login'] = reviewer
                    new_row_c['file_filename'] = path
                    dictList.append(new_row_c)
        df = pandas.DataFrame(dictList, columns=SEAADataLoader.SEAA_DATAFRAME_COL)
        print(df.shape)
        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getCADataPath(),
                                          targetFileName=f'CA_SEAA_{projectName}_data', dateCol='pr_created_at',
                                          dataFrame=df)


if __name__ == '__main__':
    SEAADataLoader.contactCAData('opencv')
