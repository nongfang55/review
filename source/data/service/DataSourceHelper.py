#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from datetime import datetime
import operator

import pymysql
import numpy as np
import pandas
import asyncio
import math
from gensim import corpora, models
from functools import reduce

from sklearn.decomposition import PCA

from source.config.projectConfig import projectConfig
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.database.AsyncSqlExecuteHelper import getMysqlObj, AsyncSqlExecuteHelper
from source.nlp.FleshReadableUtils import FleshReadableUtils
from source.nlp.SplitWordHelper import SplitWordHelper
from source.nltk import nltkFunction
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils as kt, StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper

# pullRequest表 字段、别名、是否使用
PULL_REQUEST_COLUMNS = [
    {'name': kt.STR_KEY_REPO_FULL_NAME, 'alias': 'pr_repo_full_name', 'used': False},
    {'name': 'number', 'alias': 'pr_id', 'used': True},
    {'name': kt.STR_KEY_STATE, 'alias': 'pr_state', 'used': True},
    {'name': kt.STR_KEY_TITLE, 'alias': 'pr_title', 'used': True},
    {'name': 'user_login', 'alias': 'pr_author', 'used': True},
    {'name': kt.STR_KEY_BODY, 'alias': 'pr_body', 'used': True},
    {'name': 'created_at', 'alias': 'pr_created_at', 'used': True},
    {'name': 'updated_at', 'alias': 'pr_updated_at', 'used': True},
    {'name': kt.STR_KEY_CLOSED_AT, 'alias': 'pr_closed_at', 'used': True},
    {'name': kt.STR_KEY_MERGED_AT, 'alias': 'pr_merged_at', 'used': True},
    {'name': kt.STR_KEY_MERGE_COMMIT_SHA, 'alias': 'pr_merge_commit_sha', 'used': True},
    {'name': kt.STR_KEY_AUTHOR_ASSOCIATION, 'alias': 'pr_author_association', 'used': True},
    {'name': kt.STR_KEY_MERGED, 'alias': 'pr_merged', 'used': True},
    {'name': kt.STR_KEY_COMMENTS, 'alias': 'pr_comments', 'used': True},
    {'name': kt.STR_KEY_REVIEW_COMMENTS, 'alias': 'pr_review_comments', 'used': True},
    {'name': kt.STR_KEY_COMMITS, 'alias': 'pr_commits', 'used': True},
    {'name': kt.STR_KEY_ADDITIONS, 'alias': 'pr_additions', 'used': True},
    {'name': kt.STR_KEY_DELETIONS, 'alias': 'pr_deletions', 'used': True},
    {'name': kt.STR_KEY_CHANGED_FILES, 'alias': 'pr_changed_files', 'used': True},
    {'name': kt.STR_KEY_HEAD_LABEL, 'alias': 'pr_head_label', 'used': True},
    {'name': kt.STR_KEY_BASE_LABEL, 'alias': 'pr_base_label', 'used': True},
]


async def queryPR(loop, project):
    """
    获取某个project的所有pull request
    @param loop:
    @param project: string
    @return: pr DataFrame
    """
    # 数据库连接
    mysql = await getMysqlObj(loop)
    # 查询维度
    dimension = []
    # 最终结果（DataFrame）表头
    df_columns = []
    # 根据PULL_REQUEST_COLUMNS的used字段构造需要查询的维度
    for column in PULL_REQUEST_COLUMNS:
        if not column['used']:
            continue
        dimension.append(column['name'] + " as " + column['alias'])
        df_columns.append(column['alias'])
    # 拼接查询维度
    dimension = ','.join(dimension)
    # 拼接SQL查询语句
    sql = "SELECT " + dimension + " FROM pullRequest WHERE repo_full_name = %s LIMIT 20"
    # 执行SQL语句，获得结果
    pr = await AsyncSqlHelper.query(mysql, sql, project)
    return pandas.DataFrame(data=list(pr), columns=df_columns)


async def queryReviews(loop, project):
    """
    获取某个projectP所有review
    @param loop:
    @param PR:
    @return:
    """
    # 数据库连接
    mysql = await getMysqlObj(loop)
    # 查询维度
    dimension = []
    # 最终结果（DataFrame）表头
    df_columns = []
    # 根据PULL_REQUEST_COLUMNS的used字段构造需要查询的维度
    for column in PULL_REQUEST_COLUMNS:
        if not column['used']:
            continue
        dimension.append(column['name'] + " as " + column['alias'])
        df_columns.append(column['alias'])
    # 拼接查询维度
    dimension = ','.join(dimension)
    # 拼接SQL查询语句
    sql = "SELECT " + dimension + " FROM pullRequest WHERE repo_full_name = %s LIMIT 20"
    # 执行SQL语句，获得结果
    pr = await AsyncSqlHelper.query(mysql, sql, project)
    return pandas.DataFrame(data=list(pr), columns=df_columns)


async def queryCommitFiles(loop):
    """
    获取所有git_file
    @rtype: commit_files dataframe
    """
    # 数据库连接
    mysql = await getMysqlObj(loop)
    df_columns = ['commit_sha', 'filename']
    # 获取所有commit
    sql = "SELECT commit_sha, filename FROM gitFile"
    files = await AsyncSqlHelper.query(mysql, sql, None)
    return pandas.DataFrame(data=list(files), columns=df_columns)


def query(project=None, startYear=None, startMonth=None, endYear=None, endMonth=None):
    loop = asyncio.get_event_loop()
    task = [asyncio.ensure_future(queryCommitFiles(loop))]
    tasks = asyncio.gather(*task)
    loop.run_until_complete(tasks)
    results = tasks.result()
    return results


def splitFileName(name):
    """
    分词函数
    分词规则：“actionpack／lib／action_view／helpers／form_helperrb：[“actionpack”,“actionpack／lib”,“actionpack／lib／action_view”]
    @param name: 文件路径
    @return:
    """
    # 初始化结果集
    result = set()
    # 获取filename中所有词汇
    vocs = name.split("/")
    # 按照论文规则拼接词汇
    # 如“actionpack／lib／action_view／helpers／form_helperrb可分隔为：“actionpack”,“actionpack／lib”,“actionpack／lib／action_view”)
    tmp = ""
    for voc in vocs:
        tmp += "/" + voc
        result.add(tmp)
    return result


def attachFileNameToOriginData(project, date):
    """
    在训练集中加入file信息
    @rtype: None
    """
    print("-----------------start------------------")
    start_time = datetime.now()

    # 训练数据路径
    train_data_path = projectConfig.getRootPath() + os.sep + r'data' + os.sep + 'train' + os.sep
    # 表格文件路径
    origin_filepath = train_data_path + f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}.tsv'
    target_filepath = train_data_path + f'ML_{project}_data_{date[0]}_{date[1]}_to_{date[2]}_{date[3]}_include_filepath.csv'
    # 获取原始表格
    origin_df = pandasHelper.readTSVFile(origin_filepath, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)

    # 原始表格表头
    columns = ['reviewer_reviewer', 'pr_number', 'review_id', 'commit_sha', 'author', 'pr_created_at',
               'pr_commits', 'pr_additions', 'pr_deletions', 'pr_head_label', 'pr_base_label',
               'review_submitted_at', 'commit_status_total', 'commit_status_additions',
               'commit_status_deletions', 'commit_files', 'author_review_count',
               'author_push_count', 'author_submit_gap']
    origin_df.columns = columns
    print("fetch origin data success!")

    print("start fetching commit_file data from mysql......")
    # 从数据库获取commitFiles DataFrame，包含了每次commit的文件信息
    results = query(project)
    commit_files = results[0]
    cur_time = datetime.now()
    print("fetch commit_file data success! cur_cost_time: ", cur_time - start_time)

    # 根据commit_sha，合并原始数据和commitFile
    new_df = pandas.merge(origin_df, commit_files, on="commit_sha", how="left")
    new_df.to_csv(target_filepath, encoding='utf-8', index=False, header=True)
    print("attach commit_file data to origin data success! result output to :" + target_filepath)
    print("-----------------finish------------------")


def processFileNameVector(filename):
    """
    手工计算tf-idf
    @param filename: 要读取的文件名（文件是带"include_filepath"的数据）
    @return: df: 添加路径权重后的dataframe，可直接用于机器学习算法
    """
    # 获取包含filename的df
    df = pandasHelper.readTSVFile(fileName=filename, header=pandasHelper.INT_READ_FILE_WITH_HEAD,
                                  sep=StringKeyUtils.STR_SPLIT_SEP_CSV)
    # 统计包含s的pr
    sub2pr = {}
    # 统计每个pr中s出现的次数 数据结构：key: prNumber, value: {s:2}
    pr2sub = {}
    for index, row in df.iterrows():
        subs = splitFileName(row['filename'])
        for sub in subs:
            if sub not in sub2pr:
                sub2pr[sub] = set()
            # 添加出现sub的pr
            sub2pr[sub].add(row['pr_number'])
            if row['pr_number'] not in pr2sub:
                pr2sub[row['pr_number']] = {}
            if sub not in pr2sub[row['pr_number']]:
                pr2sub[row['pr_number']][sub] = 0
            # sub在该pr中出现的次数+1
            pr2sub[row['pr_number']][sub] += 1
    # 获取所有出现过的s，添加到表头作为维度
    path_vector = list(sub2pr.keys())
    # 计算weight(pr,s) = （s在pr中出现的次数）* （log(所有pr的数量/出现s的PR数量) + 1）
    pr_path_weight_df_columns = ['pr_number'].extend(path_vector)
    pr_path_weight_df = pandas.DataFrame(columns=pr_path_weight_df_columns)
    # 所有pr的数量
    nt = len(pr2sub.keys())
    for pr in pr2sub:
        new_row = {'pr_number': pr}
        for sub in sub2pr:
            # 在df中添加新列，默认path权值都为0
            df[sub] = 0
            # s在pr中出现的次数
            tf = 0
            if sub in pr2sub[pr]:
                s_cnt = pr2sub[pr][sub]
            # 出现s的pr数量
            pr_cnt = len(sub2pr[sub])
            idf = math.log(nt / pr_cnt) + 1
            # 计算s在pr中的权值
            pr_s_weight = tf * idf
            new_row[sub] = pr_s_weight
        pr_path_weight_df = pr_path_weight_df.append([new_row], ignore_index=True)

    # 根据pr_number关联pr_path_weight_df和pr_df
    df = pandas.merge(df, pr_path_weight_df, on="pr_number", how="left")
    return df


def processFilePathVectorByGensim(filename=None, df=None):
    """
    用tf-idf模型计算filepath权重
    @description: 给定文件名或df, 计算filepath的tf-idf（文件是带"include_filepath"的数据）
    @notice: 语料是按pr_number -> [subFilepaths]定义的
    @param df: 预先读取好的dataframe
    @param filename: 指定文件名
    @return: df: 添加路径权重后的dataframe，可直接用于机器学习算法
    """
    print("---------start calculate tf-idf----------")
    start_time = datetime.now()
    if df is None:
        df = pandasHelper.readTSVFile(fileName=filename, header=pandasHelper.INT_READ_FILE_WITH_HEAD,
                                      sep=StringKeyUtils.STR_SPLIT_SEP_CSV)
        print("load file success! df size: (%d, %d)" % (df.shape[0], df.shape[1]))

    """获取filepath -> sub_filepath映射表"""
    file_path_list = set(df['filename'].copy(deep=True))
    file_path_dict = {}
    for file_path in file_path_list:
        sub_file_path = splitFileName(file_path)
        if file_path not in file_path_dict:
            file_path_dict[file_path] = set()
        file_path_dict[file_path] = file_path_dict[file_path].union(sub_file_path)
    cur_time = datetime.now()
    print("init dict(filepath -> sub_filepath) success! number of filepath: %d cur_cost_time: %s" % (
        len(file_path_dict.keys()), cur_time - start_time))

    """获取pr_number -> sub_filepath语料"""
    pr_to_file_path = df[['pr_number', 'filename']]
    # 按照pr_number分组，获得原始语料（未经过分词的filepath）"""
    groups = dict(list(pr_to_file_path.groupby('pr_number')))
    # 获取目标语料（即经过自定义分词后的语料）
    pr_file_path_corpora = []
    for pr in groups:
        paths = list(groups[pr]['filename'])
        sub_paths = list(map(lambda x: list(file_path_dict[x]), paths))
        sub_paths = reduce(lambda x, y: x + y, sub_paths)
        pr_file_path_corpora.append(sub_paths)
    cur_time = datetime.now()
    print("init pr_corpora success! cur_cost_time: ", cur_time - start_time)

    """计算tf-idf"""
    print("start tf_idf algorithm......")
    # 建立词典
    dictionary = corpora.Dictionary(pr_file_path_corpora)
    # 基于词典建立新的语料库
    corpus = [dictionary.doc2bow(text) for text in pr_file_path_corpora]
    # 用语料库训练TF-IDF模型
    tf_idf_model = models.TfidfModel(corpus)
    # 得到加权矩阵
    path_tf_tdf = list(tf_idf_model[corpus])
    cur_time = datetime.now()
    print("finish tf_idf algorithm! cur_cost_time: ", cur_time - start_time)

    """处理path_tf_tdf，构造pr_path加权矩阵"""
    print("start merge tf_idf to origin_df......")
    pr_list = list(groups.keys())
    columns = ['pr_number']
    path_ids = list(dictionary.token2id.values())
    path_ids = list(map(lambda x: str(x), path_ids))
    columns.extend(path_ids)
    pr_path_weight_df = pandas.DataFrame(columns=columns).fillna(value=0)
    for index, row in enumerate(path_tf_tdf):
        new_row = {'pr_number': pr_list[index]}
        row = list(map(lambda x: (str(x[0]), x[1]), row))
        path_weight = dict(row)
        new_row = dict(new_row, **path_weight)
        pr_path_weight_df = pr_path_weight_df.append(new_row, ignore_index=True)

    """其它数据处理操作"""
    # NAN填充为0
    pr_path_weight_df = pr_path_weight_df.fillna(value=0)
    # 去掉无用的filename列
    df.drop(axis=1, columns=['filename'], inplace=True)
    # 因为review和filename是一对多的关系，去掉filename后会存在重复数据，删除重复数据
    df.drop_duplicates(subset='review_id', keep='first', inplace=True)

    """根据pr_number关联pr_path_weight_df和pr_df"""
    df = pandas.merge(df, pr_path_weight_df, on="pr_number", how="left")
    cur_time = datetime.now()

    print("finish merger tf_idf to origin_df! total_cost_time: ", cur_time - start_time)
    print("------------------finish------------------")
    return df


# def appendTextualFeatureVector(inputDf, project, date):
#     """
#     用tf-idf模型计算pr的title, body, review的message 和commit的message权重
#     @description: 给df, 在之前的dataframe的基础上面追加   pr和review的文本形成的tf-idf特征向量
#     @notice: datafrme 必须每一条;]       就是一个单独的review，每个review值能占一条
#     @param df: 预先读取好的dataframe
#     @param project: 指定项目名
#     @param date: 开始年，开始月，结束年，结束月的四元组
#     @return: df: 添加路径权重后的dataframe，可直接用于机器学习算法
#     """
#
#     """ 注意： ALL_data 为73个项目 有列名 少了一个是来源于合并"""
#
#     print("input shape:", inputDf.shape)
#     print(date)
#
#     df = None
#     for i in range(date[0] * 12 + date[1], date[2] * 12 + date[3] + 1):
#         y = int((i - i % 12) / 12)
#         m = i % 12
#         if m == 0:
#             m = 12
#             y = y - 1
#         print(y, m)
#         filename = f'ALL_{project}_data_{y}_{m}_to_{y}_{m}.tsv'
#         path = projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train' + os.sep + 'all' + os.sep
#         temp = pandasHelper.readTSVFile(os.path.join(path, filename), pandasHelper.INT_READ_FILE_WITH_HEAD
#                                         , sep=StringKeyUtils.STR_SPLIT_SEP_TSV)
#         if df is None:
#             df = temp
#         else:
#             df = df.append(temp)
#         print(df.shape)
#     df.reset_index(drop=True, inplace=True)
#
#     """处理NAN"""
#     df.fillna(value='', inplace=True)
#
#     """过滤状态非关闭的review"""
#     df = df.loc[df['pr_state'] == 'closed'].copy(deep=True)
#     print("after fliter closed review:", df.shape)
#
#     """先对输入数据做精简 只留下感兴趣的数据"""
#     df = df[['pr_number', 'review_id', 'review_comment_id', 'pr_title', 'pr_body',
#              'commit_commit_message', 'review_comment_body']].copy(deep=True)
#
#     print("before filter:", df.shape)
#     df.drop_duplicates(['pr_number', 'review_id', 'review_comment_id'], inplace=True)
#     print("after filter:", df.shape)
#
#     """处理一个review有多个comment的场景  把comment都合并到一起"""
#     commentGroups = dict(list(df['review_comment_body'].groupby(df['review_id'])))  # 一个review的所有评论字符串连接
#
#     """留下去除comment之后的信息 去重"""
#     df = df[['pr_number', 'review_id', 'pr_title', 'pr_body', 'commit_commit_message']].copy(deep=True)
#     print(df.shape)
#     df.drop_duplicates(inplace=True)
#     print(df.shape)
#
#     def sumString(series):
#         res = ""
#         for s in series:
#             res += " " + s
#         return res
#
#     """转换comment为文本向量"""
#     df['review_comment_body'] = df['review_id'].apply(lambda x: sumString(commentGroups[x]))
#     # print(df.loc[df['review_comment_body'] != " "].shape)
#
#     """先尝试所有信息团在一起"""
#
#     """用于收集所有文本向量分词"""
#     stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词
#
#     textList = []
#     for row in df.itertuples(index=True, name='Pandas'):
#         """获取pull request的标题"""
#         pr_title = getattr(row, 'pr_title')
#         pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]
#
#         """初步尝试提取词干效果反而下降了 。。。。"""
#
#         """对单词做提取词干"""
#         pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
#         textList.append(pr_title_word_list)
#
#         """pull request的body"""
#         pr_body = getattr(row, 'pr_body')
#         pr_body_word_list = [x for x in FleshReadableUtils.word_list(pr_body) if x not in stopwords]
#         """对单词做提取词干"""
#         pr_body_word_list = nltkFunction.stemList(pr_body_word_list)
#         textList.append(pr_body_word_list)
#
#         """review 的comment"""
#         review_comment = getattr(row, 'review_comment_body')
#         review_comment_word_list = [x for x in FleshReadableUtils.word_list(review_comment) if x not in stopwords]
#         """对单词做提取词干"""
#         review_comment_word_list = nltkFunction.stemList(review_comment_word_list)
#         textList.append(review_comment_word_list)
#
#         """review的commit的 message"""
#         commit_message = getattr(row, 'commit_commit_message')
#         commit_message_word_list = [x for x in FleshReadableUtils.word_list(commit_message) if x not in stopwords]
#         """对单词做提取词干"""
#         commit_message_word_list = nltkFunction.stemList(commit_message_word_list)
#         textList.append(commit_message_word_list)
#
#     print(textList.__len__())
#
#     """对分词列表建立字典 并提取特征数"""
#     dictionary = corpora.Dictionary(textList)
#     print('词典：', dictionary)
#
#     feature_cnt = len(dictionary.token2id)
#     print("词典特征数：", feature_cnt)
#
#     """根据词典建立语料库"""
#     corpus = [dictionary.doc2bow(text) for text in textList]
#     # print('语料库:', corpus)
#
#     """语料库训练TF-IDF模型"""
#     tfidf = models.TfidfModel(corpus)
#
#     """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
#     wordVectors = []
#     for i in range(0, df.shape[0]):
#         words = []
#         for j in range(0, 4):
#             words.extend(textList[4 * i + j])
#         # print(words)
#         wordVectors.append(dict(tfidf[dictionary.doc2bow(words)]))
#     print(wordVectors.__len__())
#     """代表文本特征的dataframe"""
#     wordFeatures = DataProcessUtils.convertFeatureDictToDataFrame(wordVectors, feature_cnt)
#     wordFeatures.reset_index(drop=True, inplace=True)
#     # print(inputDf.dtypes)
#     # print(wordFeatures.dtypes)
#     wordFeatures.astype('float64')
#     wordFeatures.columns = ["word_" + str(x) for x in range(1, wordFeatures.shape[1] + 1)]
#     print("word features:", wordFeatures.shape)
#     inputDf = pandas.concat([inputDf, wordFeatures], axis=1)
#     # print("input df:", inputDf.shape)
#     # print(inputDf.columns)
#     return inputDf


def appendFilePathFeatureVector(inputDf, projectName, date, pull_number_name):
    """
       用tf-idf模型计算pr的所有commit的设计的文件的路径
       @description: 给df, 在之前的dataframe的基础上面追加   pr路径形成的tf-idf特征向量
       @notice: datafrme 必须有pull_numberzid店，可以重复
       @param origin_df: 预先读取好的dataframe
       @param projectName: 指定项目名
       @param date: 开始年，开始月，结束年，结束月的四元组
       @return: df: 添加路径权重后的dataframe，可直接用于机器学习算法
       """
    df = inputDf[[pull_number_name]].copy(deep=True)
    df.drop_duplicates(inplace=True)
    df.columns = ['pr_number']

    """读取commit pr relation文件"""
    time1 = datetime.now()
    data_train_path = projectConfig.getDataTrainPath()
    commit_file_data_path = projectConfig.getCommitFilePath()
    pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
    """commit file 信息是拼接出来的 所以有抬头"""
    commitFileData = pandasHelper.readTSVFile(
        os.path.join(commit_file_data_path, f'ALL_{projectName}_data_commit_file.tsv'), low_memory=False,
        header=pandasHelper.INT_READ_FILE_WITH_HEAD)
    print("raw commit file :", commitFileData.shape)

    commitPRRelationData = pandasHelper.readTSVFile(
        os.path.join(pr_commit_relation_path, f'ALL_{projectName}_data_pr_commit_relation.tsv'),
        pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False
    )
    commitPRRelationData.columns = DataProcessUtils.COLUMN_NAME_PR_COMMIT_RELATION
    print("pr_commit_relation:", commitPRRelationData.shape)

    print("read file cost time:", datetime.now() - time1)

    """做三者连接"""
    df = pandas.merge(df, commitPRRelationData, left_on='pr_number', right_on='pull_number')
    print("merge relation:", df.shape)
    df = pandas.merge(df, commitFileData, left_on='sha', right_on='commit_sha')
    df.reset_index(drop=True, inplace=True)
    df = df[['pr_number', 'commit_sha', 'file_filename']].copy(deep=True)
    df.drop_duplicates(inplace=True)

    print("after merge:", df.shape)

    """获取filepath -> sub_filepath映射表"""
    file_path_list = set(df['file_filename'].copy(deep=True))
    file_path_dict = {}
    for file_path in file_path_list:
        sub_file_path = splitFileName(file_path)
        if file_path not in file_path_dict:
            file_path_dict[file_path] = set()
        file_path_dict[file_path] = file_path_dict[file_path].union(sub_file_path)

    """获取pr_number -> sub_filepath语料"""
    pr_to_file_path = df[['pr_number', 'file_filename']]
    # 按照pr_number分组，获得原始语料（未经过分词的filepath）"""
    groups = dict(list(pr_to_file_path.groupby('pr_number')))
    # 获取目标语料（即经过自定义分词后的语料）
    pr_file_path_corpora = []
    for pr in groups:
        paths = list(groups[pr]['file_filename'])
        sub_paths = list(map(lambda x: list(file_path_dict[x]), paths))
        sub_paths = reduce(lambda x, y: x + y, sub_paths)
        pr_file_path_corpora.append(sub_paths)

    """计算tf-idf"""
    print("start tf_idf algorithm......")
    # 建立词典
    dictionary = corpora.Dictionary(pr_file_path_corpora)
    # 基于词典建立新的语料库
    corpus = [dictionary.doc2bow(text) for text in pr_file_path_corpora]
    # 用语料库训练TF-IDF模型
    tf_idf_model = models.TfidfModel(corpus)
    # 得到加权矩阵
    path_tf_tdf = list(tf_idf_model[corpus])

    """处理path_tf_tdf，构造pr_path加权矩阵"""
    print("start merge tf_idf to origin_df......")
    pr_list = list(groups.keys())
    columns = ['pr_number']
    path_ids = list(dictionary.token2id.values())
    path_ids = list(map(lambda x: str(x), path_ids))
    columns.extend(path_ids)
    pr_path_weight_df = pandas.DataFrame(columns=columns).fillna(value=0)
    for index, row in enumerate(path_tf_tdf):
        new_row = {'pr_number': pr_list[index]}
        row = list(map(lambda x: (str(x[0]), x[1]), row))
        path_weight = dict(row)
        new_row = dict(new_row, **path_weight)
        pr_path_weight_df = pr_path_weight_df.append(new_row, ignore_index=True)
    pr_path_weight_df = pr_path_weight_df.fillna(value=0)
    print(pr_path_weight_df.shape)

    tempData = pr_path_weight_df.copy(deep=True)
    tempData.drop(columns=['pr_number'], inplace=True)

    """PAC 做缩减"""
    pca = PCA(n_components=10)
    tempData = pca.fit_transform(tempData)
    print("after pca :", tempData.shape)
    print(pca.explained_variance_ratio_)
    tempData = pandas.DataFrame(tempData)

    """和提供的数据做拼接"""
    tempData['pr_number_t'] = list(pr_path_weight_df['pr_number'])
    inputDf = pandas.merge(inputDf, tempData, left_on=pull_number_name, right_on='pr_number_t')
    inputDf.drop(columns=['pr_number_t'], inplace=True)

    return inputDf


def appendTextualFeatureVector(inputDf, projectName, date, pull_number_name):
    """
       用tf-idf模型计算pr的所有title,pr的文本的路径
       @description: 给df, 在之前的dataframe的基础上面追加   pr路径形成的tf-idf特征向量
       @notice: datafrme 必须有pull_number_id，可以重复
       @param origin_df: 预先读取好的dataframe
       @param projectName: 指定项目名
       @param date: 开始年，开始月，结束年，结束月的四元组
       @return: df: 添加路径权重后的dataframe，可直接用于机器学习算法
    """

    print("input shape:", inputDf.shape)
    print(date)

    df = inputDf[[pull_number_name]].copy(deep=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = ['pr_number']

    """读取commit pr relation文件"""
    data_train_path = projectConfig.getDataTrainPath()
    pr_review_data = pandasHelper.readTSVFile(data_train_path + os.sep + f'ALL_{projectName}_data_pr_review_commit_file.tsv')
    pr_review_data.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
    pr_review_data = pr_review_data[['pr_number', 'pr_title', 'pr_body']].copy(deep=True)
    pr_review_data.drop_duplicates(inplace=True)
    pr_review_data.reset_index(drop=True, inplace=True)
    """处理NAN"""
    pr_review_data.fillna(value='', inplace=True)

    """pull_number和pr review commit relation做拼接"""
    df = pandas.merge(df, pr_review_data, how='left')
    df.fillna(value='', inplace=True)

    """用于收集所有文本向量分词"""
    stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词

    textList = []
    for row in df.itertuples(index=True, name='Pandas'):
        tempList = []
        """获取pull request的标题"""
        pr_title = getattr(row, 'pr_title')
        pr_title_word_list = [x for x in FleshReadableUtils.word_list(pr_title) if x not in stopwords]

        """初步尝试提取词干效果反而下降了 。。。。"""

        """对单词做提取词干"""
        pr_title_word_list = nltkFunction.stemList(pr_title_word_list)
        tempList.extend(pr_title_word_list)

        """pull request的body"""
        pr_body = getattr(row, 'pr_body')
        pr_body_word_list = [x for x in FleshReadableUtils.word_list(pr_body) if x not in stopwords]
        """对单词做提取词干"""
        pr_body_word_list = nltkFunction.stemList(pr_body_word_list)
        tempList.extend(pr_body_word_list)
        textList.append(tempList)

    print(textList.__len__())
    """对分词列表建立字典 并提取特征数"""
    dictionary = corpora.Dictionary(textList)
    print('词典：', dictionary)

    feature_cnt = len(dictionary.token2id)
    print("词典特征数：", feature_cnt)

    """根据词典建立语料库"""
    corpus = [dictionary.doc2bow(text) for text in textList]
    # print('语料库:', corpus)
    """语料库训练TF-IDF模型"""
    tfidf = models.TfidfModel(corpus)

    """再次遍历数据，形成向量，向量是稀疏矩阵的形式"""
    wordVectors = []
    for i in range(0, df.shape[0]):
        wordVectors.append(dict(tfidf[dictionary.doc2bow(textList[i])]))

    """填充为向量"""
    wordVectors = DataProcessUtils.convertFeatureDictToDataFrame(wordVectors, featureNum=feature_cnt)

    """PAC 做缩减"""
    pca = PCA(n_components=10)
    tempData = pca.fit_transform(wordVectors)
    print("after pca :", tempData.shape)
    print(pca.explained_variance_ratio_)
    tempData = pandas.DataFrame(tempData)
    tempData['pr_number_t'] = df['pr_number']

    """和原来特征做拼接"""
    inputDf = pandas.merge(inputDf, tempData, left_on=pull_number_name, right_on='pr_number_t')
    inputDf.drop(columns=['pr_number_t'], inplace=True)
    return inputDf


if __name__ == '__main__':
    # 给定项目和日期，为指定训练集添加filepath信息
    # dates = [[2019, 4, 2019, 10], [2019, 7, 2019, 10], [2019, 9, 2019, 10]]
    # for date in dates:
    #     attachFileNameToOriginData("akka", date)
    # 训练数据路径
    train_data_path = projectConfig.getRootPath() + r'/data/train/'
    # 表格文件路径
    filepath = train_data_path + f'ML_rails_data_2018_4_to_2019_4_include_filepath.csv'
    processFilePathVectorByGensim(filepath)
