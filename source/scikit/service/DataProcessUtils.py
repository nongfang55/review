# coding=gbk
import heapq
import os
import random
import time
from datetime import datetime

import numpy
import pandas
import scikit_posthocs
import scipy
import seaborn
from pandas import DataFrame
from pyecharts.charts import HeatMap
from scipy.stats import mannwhitneyu, ranksums, ttest_1samp, ttest_ind, wilcoxon

from source.config.projectConfig import projectConfig
from source.nlp.SplitWordHelper import SplitWordHelper
from source.scikit.service.BotUserRecognizer import BotUserRecognizer
from source.scikit.service.RecommendMetricUtils import RecommendMetricUtils
from source.utils.ExcelHelper import ExcelHelper
from source.utils.StringKeyUtils import StringKeyUtils
from source.utils.pandas.pandasHelper import pandasHelper
import matplotlib.pyplot as plt


class DataProcessUtils:
    """用于处理项目数据一些通用的方法工具类"""

    """由于有些列名会有重合，采用重命名"""
    COLUMN_NAME_ALL = ['pr_repo_full_name', 'pr_number', 'pr_id', 'pr_node_id',
                       'pr_state', 'pr_title', 'pr_user_login', 'pr_body',
                       'pr_created_at',
                       'pr_updated_at', 'pr_closed_at', 'pr_merged_at', 'pr_merge_commit_sha',
                       'pr_author_association', 'pr_merged', 'pr_comments', 'pr_review_comments',
                       'pr_commits', 'pr_additions', 'pr_deletions', 'pr_changed_files',
                       'pr_head_label', 'pr_base_label',
                       'review_repo_full_name', 'review_pull_number',
                       'review_id', 'review_user_login', 'review_body', 'review_state', 'review_author_association',
                       'review_submitted_at', 'review_commit_id', 'review_node_id',

                       'commit_sha',
                       'commit_node_id', 'commit_author_login', 'commit_committer_login', 'commit_commit_author_date',
                       'commit_commit_committer_date', 'commit_commit_message', 'commit_commit_comment_count',
                       'commit_status_total', 'commit_status_additions', 'commit_status_deletions',

                       'file_commit_sha',
                       'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions', 'file_changes',
                       'file_patch',

                       'review_comment_id', 'review_comment_user_login', 'review_comment_body',
                       'review_comment_pull_request_review_id', 'review_comment_diff_hunk', 'review_comment_path',
                       'review_comment_commit_id', 'review_comment_position', 'review_comment_original_position',
                       'review_comment_original_commit_id', 'review_comment_created_at', 'review_comment_updated_at',
                       'review_comment_author_association', 'review_comment_start_line',
                       'review_comment_original_start_line',
                       'review_comment_start_side', 'review_comment_line', 'review_comment_original_line',
                       'review_comment_side', 'review_comment_in_reply_to_id', 'review_comment_node_id',
                       'review_comment_change_trigger']

    """ 上面col来源SQL语句：
            select *
        from pullRequest, review, gitCommit, gitFile, reviewComment
        where pullRequest.repo_full_name = 'scala/scala' and
          review.repo_full_name = pullRequest.repo_full_name
        and pullRequest.number = review.pull_number and
          gitCommit.sha = review.commit_id and gitFile.commit_sha = gitCommit.sha
        and reviewComment.pull_request_review_id = review.id
    """

    """
     不幸的是 这样会有漏洞，导致没有reviewcomment的数据被忽视掉，需要reviewcomment那里外连接
    """

    COLUMN_NAME_PR_REVIEW_COMMIT_FILE = ['pr_repo_full_name', 'pr_number', 'pr_id', 'pr_node_id',
                                         'pr_state', 'pr_title', 'pr_user_login', 'pr_body',
                                         'pr_created_at',
                                         'pr_updated_at', 'pr_closed_at', 'pr_merged_at', 'pr_merge_commit_sha',
                                         'pr_author_association', 'pr_merged', 'pr_comments', 'pr_review_comments',
                                         'pr_commits', 'pr_additions', 'pr_deletions', 'pr_changed_files',
                                         'pr_head_label', 'pr_base_label',
                                         'review_repo_full_name', 'review_pull_number',
                                         'review_id', 'review_user_login', 'review_body', 'review_state',
                                         'review_author_association',
                                         'review_submitted_at', 'review_commit_id', 'review_node_id',

                                         'commit_sha',
                                         'commit_node_id', 'commit_author_login', 'commit_committer_login',
                                         'commit_commit_author_date',
                                         'commit_commit_committer_date', 'commit_commit_message',
                                         'commit_commit_comment_count',
                                         'commit_status_total', 'commit_status_additions', 'commit_status_deletions',

                                         'file_commit_sha',
                                         'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions',
                                         'file_changes',
                                         'file_patch']

    COLUMN_NAME_REVIEW_COMMENT = [
        'review_comment_id', 'review_comment_user_login', 'review_comment_body',
        'review_comment_pull_request_review_id', 'review_comment_diff_hunk', 'review_comment_path',
        'review_comment_commit_id', 'review_comment_position', 'review_comment_original_position',
        'review_comment_original_commit_id', 'review_comment_created_at', 'review_comment_updated_at',
        'review_comment_author_association', 'review_comment_start_line',
        'review_comment_original_start_line',
        'review_comment_start_side', 'review_comment_line', 'review_comment_original_line',
        'review_comment_side', 'review_comment_in_reply_to_id', 'review_comment_node_id',
        'review_comment_change_trigger']

    COLUMN_NAME_COMMIT_FILE = [
        'commit_sha',
        'commit_node_id', 'commit_author_login', 'commit_committer_login',
        'commit_commit_author_date',
        'commit_commit_committer_date', 'commit_commit_message',
        'commit_commit_comment_count',
        'commit_status_total', 'commit_status_additions', 'commit_status_deletions',
        'file_commit_sha',
        'file_sha', 'file_filename', 'file_status', 'file_additions', 'file_deletions',
        'file_changes',
        'file_patch'
    ]

    COLUMN_NAME_PR_COMMIT_RELATION = [
        'repo_full_name', 'pull_number', 'sha'
    ]

    @staticmethod
    def splitDataByMonth(filename, targetPath, targetFileName, dateCol, dataFrame=None, hasHead=False,
                         columnsName=None):
        """把提供的filename 中的数据按照日期分类并切分生成不同文件
            targetPath: 切分文件目标路径
            targetFileName: 提供存储的文件名字
            dateCol: 用于分时间的列名
            dataFrame: 若是提供数据集，则不读文件
            columnsName； 在没有读取文件head的时候必须提供columnsName
        """
        df = None
        if dataFrame is not None:
            df = dataFrame
        elif not hasHead:
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False)
            if columnsName is None:
                raise Exception("columnName is None without head")
            df.columns = columnsName
        else:
            df = pandasHelper.readTSVFile(filename, pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False)
        # print(df[dateCol])

        df['label'] = df[dateCol].apply(lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        df['label_y'] = df['label'].apply(lambda x: x.tm_year)
        df['label_m'] = df['label'].apply(lambda x: x.tm_mon)
        print(max(df['label']), min(df['label']))

        maxYear = max(df['label']).tm_year
        maxMonth = max(df['label']).tm_mon
        minYear = min(df['label']).tm_year
        minMonth = min(df['label']).tm_mon
        print(maxYear, maxMonth, minYear, minMonth)

        # 新增路径判断
        if not os.path.isdir(targetPath):
            os.makedirs(targetPath)

        start = minYear * 12 + minMonth
        end = maxYear * 12 + maxMonth
        for i in range(start, end + 1):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1
            print(y, m)
            subDf = df.loc[(df['label_y'] == y) & (df['label_m'] == m)].copy(deep=True)
            subDf.drop(columns=['label', 'label_y', 'label_m'], inplace=True)
            # print(subDf)
            print(subDf.shape)
            # targetFileName = filename.split(os.sep)[-1].split(".")[0]
            sub_filename = f'{targetFileName}_{y}_{m}_to_{y}_{m}.tsv'
            pandasHelper.writeTSVFile(os.path.join(targetPath, sub_filename), subDf
                                      , pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def changeStringToNumber(data, columns):  # 对dataframe的一些特征做文本转数字  input: dataFrame，需要处理的某些列
        if isinstance(data, DataFrame):  # 注意： dataframe之前需要resetindex
            count = 0
            convertDict = {}  # 用于转换的字典  开始为1
            for column in columns:
                pos = 0
                for item in data[column]:
                    if convertDict.get(item, None) is None:
                        count += 1
                        convertDict[item] = count
                    data.at[pos, column] = convertDict[item]
                    pos += 1
            return convertDict  # 新增返回映射字典

    @staticmethod
    def judgeRecommend(recommendList, answer, recommendNum):

        """评价推荐表现"""
        topk = RecommendMetricUtils.topKAccuracy(recommendList, answer, recommendNum)
        print("topk")
        print(topk)
        mrr = RecommendMetricUtils.MRR(recommendList, answer, recommendNum)
        print("mrr")
        print(mrr)
        precisionk, recallk, fmeasurek = RecommendMetricUtils.precisionK(recommendList, answer, recommendNum)
        print("precision:")
        print(precisionk)
        print("recall:")
        print(recallk)
        print("fmeasure:")
        print(fmeasurek)

        return topk, mrr, precisionk, recallk, fmeasurek

    @staticmethod
    def saveResult(filename, sheetName, topk, mrr, precisionk, recallk, fmeasurek, date):
        """时间和准确率"""
        content = None
        if date[3] == 1:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2] - 1}.{12}", "TopKAccuracy"]
        else:
            content = [f"{date[2]}.{date[3]}", f"{date[0]}.{date[1]} - {date[2]}.{date[3] - 1}", "TopKAccuracy"]

        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + topk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'MRR']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + mrr
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'precisionK']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + precisionk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'recallk']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + recallk
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'F-Measure']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + fmeasurek
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def saveFinallyResult(filename, sheetName, topks, mrrs, precisionks, recallks, fmeasureks):
        """用于最后的几个月结果算平均做汇总"""

        """时间和准确率"""
        content = ['', '', "AVG_TopKAccuracy"]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(topks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_MRR']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(mrrs)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_precisionK']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(precisionks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_recallk']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(recallks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 'AVG_F-Measure']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', '', 1, 2, 3, 4, 5]
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['', ''] + DataProcessUtils.getAvgScore(fmeasureks)
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())
        content = ['']
        ExcelHelper().appendExcelRow(filename, sheetName, content, style=ExcelHelper.getNormalStyle())

    @staticmethod
    def getAvgScore(scores):
        """计算平均得分"""
        avg = []
        for i in range(0, scores[0].__len__()):
            avg.append(0)
        for score in scores:
            for i in range(0, score.__len__()):
                avg[i] += score[i]
        for i in range(0, scores[0].__len__()):
            avg[i] /= scores.__len__()
        return avg

    @staticmethod
    def convertFeatureDictToDataFrame(dicts, featureNum):
        """通过转换 feature的形式来让tf-idf 模型生成的数据可以转换成向量"""
        ar = numpy.zeros((dicts.__len__(), featureNum))
        result = pandas.DataFrame(ar)
        pos = 0
        for d in dicts:
            for key in d.keys():
                result.loc[pos, key] = d[key]
            pos = pos + 1

        return result

    @staticmethod
    def contactReviewCommentData(projectName):
        """用于拼接项目的数据并保存  之前在SQL语句上面跑太花时间了"""

        pr_review_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                           , f'ALL_{projectName}_data_pr_review_commit_file.tsv')
        review_comment_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                                , f'ALL_data_review_comment.tsv')

        out_put_file_name = os.path.join(projectConfig.getRootPath() + os.sep + 'data' + os.sep + 'train'
                                         , f'ALL_{projectName}_data.tsv')

        reviewData = pandasHelper.readTSVFile(pr_review_file_name, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        reviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
        print(reviewData.shape)

        commentData = pandasHelper.readTSVFile(review_comment_file_name, pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
        commentData.columns = DataProcessUtils.COLUMN_NAME_REVIEW_COMMENT
        print(commentData.shape)

        result = reviewData.join(other=commentData.set_index('review_comment_pull_request_review_id')
                                 , on='review_id', how='left')

        print(result.loc[result['review_comment_id'].isna()].shape)
        pandasHelper.writeTSVFile(out_put_file_name, result, pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)

    @staticmethod
    def splitProjectCommitFileData(projectName):
        """从总的commit file的联合数据中分出某个项目的数据，减少数据量节约时间"""

        """读取信息"""
        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        target_file_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        target_file_name = f'ALL_{projectName}_data_commit_file.tsv'

        prReviewData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'),
            pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False)
        print(prReviewData.shape)
        prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE

        commitFileData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, 'ALL_data_commit_file.tsv'), pandasHelper.INT_READ_FILE_WITHOUT_HEAD
            , low_memory=False)
        commitFileData.columns = DataProcessUtils.COLUMN_NAME_COMMIT_FILE
        print(commitFileData.shape)

        commitPRRelationData = pandasHelper.readTSVFile(
            os.path.join(pr_commit_relation_path, f'ALL_{projectName}_data_pr_commit_relation.tsv'),
            pandasHelper.INT_READ_FILE_WITHOUT_HEAD, low_memory=False
        )
        print(commitPRRelationData.shape)
        print("read file cost time:", datetime.now() - time1)

        """先收集pr相关的commit"""
        commitPRRelationData.columns = ['repo_full_name', 'pull_number', 'sha']
        commitPRRelationData = commitPRRelationData['sha'].copy(deep=True)
        commitPRRelationData.drop_duplicates(inplace=True)
        print(commitPRRelationData.shape)

        prReviewData = prReviewData['commit_sha'].copy(deep=True)
        prReviewData.drop_duplicates(inplace=True)
        print(prReviewData.shape)

        needCommits = prReviewData.append(commitPRRelationData)
        print("before drop duplicates:", needCommits.shape)
        needCommits.drop_duplicates(inplace=True)
        print("actually need commit:", needCommits.shape)
        needCommits = list(needCommits)

        """从总的commit file信息中筛选出需要的信息"""
        print(commitFileData.columns)
        commitFileData = commitFileData.loc[commitFileData['commit_sha'].
            apply(lambda x: x in needCommits)].copy(deep=True)
        print(commitFileData.shape)

        pandasHelper.writeTSVFile(os.path.join(target_file_path, target_file_name), commitFileData
                                  , pandasHelper.STR_WRITE_STYLE_WRITE_TRUNC)
        print(f"write over: {target_file_name}, cost time:", datetime.now() - time1)

    @staticmethod
    def contactFPSData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):
        """
        2020.6.23
        新增标识符label  分别区别 review comment、 issue comment 和 all的情况

        dataframe 输出统一格式：
        [repo_full_name, pull_number, pr_created_at, review_user_login, commit_sha, file_filename]
        不含信息项置 0

        对于 label = review comment
        通过 ALL_{projectName}_data_pr_review_commit_file
             ALL_{projectName}_commit_file
             ALL_{projectName}_data_pr_commit_relation 三个文件拼接出FPS所需信息量的文件

        对于 label = issue comment
        通过 ALL_{projectName}_data_pull_request
             # ALL_{projectName}_commit_file
             # ALL_{projectName}_data_pr_commit_relation
             ALL_{projectName}_data_issuecomment 四个文件拼接出FPS所需信息量的文件
             ALL_{projectName}_data_pr_change_file

        对于 label = issue and review comment
        通过  ALL_{projectName}_data_pull_request
              ALL_{projectName}_data_issuecomment
              ALL_{projectName}_data_pr_change_file
              ALL_{proejctName}_data_review
              ALL_{proejctName}_data_pr_change_trigger

        注 ：issue comment不能使用  ALL_{projectName}_data_pr_review_commit_file
        这里pr已经和review做连接 导致数量减少
        """

        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        commit_file_data_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        pr_change_file_path = projectConfig.getPRChangeFilePath()
        review_path = projectConfig.getReviewDataPath()
        change_trigger_path = projectConfig.getPRTimeLineDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)

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

        """issue commit 数据库输出 自带抬头"""
        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """pull request 数据库输出 自带抬头"""
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """pr_change_file 数据库输出 自带抬头"""
        prChangeFileData = pandasHelper.readTSVFile(
            os.path.join(pr_change_file_path, f'ALL_{projectName}_data_pr_change_file.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ review 数据库输出 自带抬头"""
        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ pr_change_trigger 自带抬头"""
        changeTriggerData = pandasHelper.readTSVFile(
            os.path.join(change_trigger_path, f'ALL_{projectName}_data_pr_change_trigger.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        print("read file cost time:", datetime.now() - time1)

        """过滤状态非关闭的pr review"""
        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)
        elif label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT or label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            pullRequestData = pullRequestData.loc[pullRequestData['state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", pullRequestData.shape)

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            """过滤pr 作者就是reviewer的情况"""
            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)

            """过滤不需要的字段"""
            prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)

            commitFileData = commitFileData[['commit_sha', 'file_filename']].copy(deep=True)
            commitFileData.drop_duplicates(inplace=True)
            commitFileData.reset_index(drop=True, inplace=True)
            print("after fliter commit_file:", commitFileData.shape)

        pullRequestData = pullRequestData[['number', 'created_at', 'closed_at', 'user_login', 'node_id']].copy(
            deep=True)
        reviewData = reviewData[["pull_number", "id", "user_login", 'submitted_at']].copy(deep=True)

        targetFileName = f'FPS_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'FPS_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'FPS_ALL_{projectName}_data'

        """按照不同类型做连接"""
        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            data = pandas.merge(prReviewData, commitPRRelationData, left_on='pr_number', right_on='pull_number')
            print("merge relation:", data.shape)
            data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
            data.reset_index(drop=True, inplace=True)
            data.drop(columns=['sha'], inplace=True)
            data.drop(columns=['pr_number'], inplace=True)
            print("交换位置")
            order = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                     'file_filename']
            data = data[order]
        elif label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            # data = pandas.merge(pullRequestData, commitPRRelationData, left_on='number', right_on='pull_number')
            # data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
            data = pandas.merge(pullRequestData, prChangeFileData, left_on='number', right_on='pull_number')
            data = pandas.merge(data, issueCommentData, left_on='number', right_on='pull_number')
            """过滤作者 发issue comment的情况"""
            data = data.loc[data['user_login_x'] != data['user_login_y']].copy(deep=True)
            data.drop(columns=['user_login_x'], axis=1, inplace=True)
            """过滤  多个commit包含一个文件的情况  重要 2020.6.3"""
            data.drop_duplicates(['number', 'filename', 'user_login_y'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['commit_sha'] = None
            """只选出感兴趣的部分"""
            data = data[['repo_full_name_x', 'pull_number_x', 'created_at_x', 'user_login_y', 'commit_sha',
                         'filename']].copy(deep=True)
            data.drop_duplicates(inplace=True)
            data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                            'file_filename']
            data.reset_index(drop=True)
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            """思路  上面两部分依次做凭借， 最后加上文件"""
            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)
            """过滤删除用户的场景"""
            data_issue.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景"""
            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            data_issue = data_issue[['node_id_x', 'number', 'created_at_x', 'user_login_y']].copy(deep=True)
            data_issue.columns = ['node_id_x', 'number', 'created_at', 'user_login']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)
            """过滤删除用户场景"""
            data_review.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景  """
            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['node_id', 'number', 'created_at', 'user_login_y']].copy(deep=True)
            data_review.columns = ['node_id', 'number', 'created_at', 'user_login']
            data_review.rename(columns={'node_id': 'node_id_x'}, inplace=True)
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 轴合并
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True)
            print(data.shape)

            """拼接 文件改动"""
            data = pandas.merge(data, prChangeFileData, left_on='number', right_on='pull_number')
            """只选出感兴趣的部分"""
            data['commit_sha'] = 0
            data = data[['repo_full_name', 'number', 'node_id_x', 'created_at', 'user_login', 'commit_sha', 'filename']].copy(
                deep=True)
            data.drop_duplicates(inplace=True)

            """删除无用review"""
            # unuseful_review_idx = []
            # for index, row in data.iterrows():
            #     change_trigger_records = changeTriggerData.loc[(changeTriggerData['pullrequest_node'] == row['node_id_x'])
            #                                                    & (changeTriggerData['user_login'] == row['user_login'])]
            #     if change_trigger_records.empty:
            #         unuseful_review_idx.append(index)
            # data = data.drop(labels=unuseful_review_idx, axis=0)
            """change_trigger只取出pr, reviewer，和data取交集"""
            changeTriggerData = changeTriggerData[changeTriggerData['change_trigger'] >= 0]
            changeTriggerData = changeTriggerData[['pullrequest_node', 'user_login']].copy(deep=True)
            changeTriggerData.drop_duplicates(inplace=True)
            changeTriggerData.rename(columns={'pullrequest_node': 'node_id_x'}, inplace=True)
            data = pandas.merge(data, changeTriggerData, how='inner')
            data = data.drop(labels='node_id_x', axis=1)

            data.columns = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha',
                            'file_filename']
            data.sort_values(by='pull_number', ascending=False, inplace=True)
            data.reset_index(drop=True, inplace=True)

        print("after merge:", data.shape)

        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getFPSDataPath(),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def convertStringTimeToTimeStrip(s):
        return int(time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S")))

    @staticmethod
    def contactMLData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):
        """
        对于 label == review comment
        通过 ALL_{projectName}_data_pr_review_commit_file
             ALL_commit_file
             ALL_data_review_comment 三个文件初步拼接出ML所需信息量的文件

        对于 label == review comment and issue comment
        通过 ALL_{projectName}_data_pullrequest
             ALL_{projectName}_data_review
             ALL_{projectName}_data_issuecomment 三个文件
        """

        """
          选择特征  
        """

        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        commit_file_data_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)

        """issue commit 数据库输出 自带抬头"""
        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """pull request 数据库输出 自带抬头"""
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ review 数据库输出 自带抬头"""
        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        targetFileName = f'ML_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'ML_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'ML_ALL_{projectName}_data'

        print("read file cost time:", datetime.now() - time1)

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:

            """过滤状态非关闭的pr review"""
            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)

            """过滤pr 作者就是reviewer的情况"""
            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)

            """过滤不需要的字段"""
            prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at',
                                         'pr_commits', 'pr_additions', 'pr_deletions',
                                         'pr_changed_files', 'pr_head_label', 'pr_base_label', 'pr_user_login']].copy(
                deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)

            """尝试添加 作者总共提交次数，作者提交时间间隔，作者review次数的特征"""
            author_push_count = []
            author_submit_gap = []
            author_review_count = []
            pos = 0
            for data in prReviewData.itertuples(index=False):
                pullNumber = getattr(data, 'pr_number')
                author = getattr(data, 'pr_user_login')
                temp = prReviewData.loc[prReviewData['pr_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                push_num = temp['pr_number'].drop_duplicates().shape[0]
                author_push_count.append(push_num)

                gap = DataProcessUtils.convertStringTimeToTimeStrip(prReviewData.loc[prReviewData.shape[0] - 1,
                                                                                     'pr_created_at']) - DataProcessUtils.convertStringTimeToTimeStrip(
                    prReviewData.loc[0, 'pr_created_at'])
                if push_num != 0:
                    last_num = list(temp['pr_number'])[-1]
                    this_created_time = getattr(data, 'pr_created_at')
                    last_created_time = list(prReviewData.loc[prReviewData['pr_number'] == last_num]['pr_created_at'])[
                        0]
                    gap = int(time.mktime(time.strptime(this_created_time, "%Y-%m-%d %H:%M:%S"))) - int(
                        time.mktime(time.strptime(last_created_time, "%Y-%m-%d %H:%M:%S")))
                author_submit_gap.append(gap)

                temp = prReviewData.loc[prReviewData['review_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                review_num = temp.shape[0]
                author_review_count.append(review_num)
            prReviewData['author_push_count'] = author_push_count
            prReviewData['author_review_count'] = author_review_count
            prReviewData['author_submit_gap'] = author_submit_gap

            data = prReviewData

        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            """先找出所有参与 reivew的人选"""
            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)
            """过滤删除用户的场景"""
            data_issue.dropna(subset=['user_login_y'], inplace=True)
            """"过滤 head_label 为nan的场景"""
            data_issue.dropna(subset=['head_label'], inplace=True)
            """过滤机器人的场景"""
            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            data_issue = data_issue[['number', 'user_login_y',
                                     'created_at_x', 'commits', 'additions', 'deletions',
                                     'changed_files', 'head_label', 'base_label', 'user_login_x']].copy(deep=True)

            data_issue.columns = ['pr_number', 'review_user_login', 'pr_created_at', 'pr_commits',
                                  'pr_additions', 'pr_deletions', 'pr_changed_files', 'pr_head_label',
                                  'pr_base_label', 'pr_user_login']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)
            """过滤删除用户场景"""
            data_review.dropna(subset=['user_login_y'], inplace=True)
            """"过滤 head_label 为nan的场景"""
            data_review.dropna(subset=['head_label'], inplace=True)
            """过滤机器人的场景  """
            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['number', 'user_login_y',
                                     'created_at', 'commits', 'additions', 'deletions',
                                     'changed_files', 'head_label', 'base_label', 'user_login_x']].copy(deep=True)

            data_review.columns = ['pr_number', 'review_user_login', 'pr_created_at', 'pr_commits',
                                  'pr_additions', 'pr_deletions', 'pr_changed_files', 'pr_head_label',
                                  'pr_base_label', 'pr_user_login']
            data_review.drop_duplicates(inplace=True)

            rawData = pandas.concat([data_issue, data_review], axis=0)  # 0 轴合并
            rawData.drop_duplicates(inplace=True)
            rawData.reset_index(drop=True, inplace=True)
            print(rawData.shape)

            "pr_number, review_user_login, pr_created_at, pr_commits, pr_additions, pr_deletions" \
            "pr_changed_files, pr_head_label, pr_base_label, pr_user_login, author_push_count," \
            "author_review_count, author_submit_gap"

            """尝试添加 作者总共提交次数，作者提交时间间隔，作者review次数的特征"""
            author_push_count = []
            author_submit_gap = []
            author_review_count = []
            pos = 0
            for data in rawData.itertuples(index=False):
                pullNumber = getattr(data, 'pr_number')
                author = getattr(data, 'pr_user_login')
                temp = rawData.loc[rawData['pr_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                push_num = temp['pr_number'].drop_duplicates().shape[0]
                author_push_count.append(push_num)

                gap = DataProcessUtils.convertStringTimeToTimeStrip(rawData.loc[rawData.shape[0] - 1,
                                                                                     'pr_created_at']) - DataProcessUtils.convertStringTimeToTimeStrip(
                    rawData.loc[0, 'pr_created_at'])
                if push_num != 0:
                    last_num = list(temp['pr_number'])[-1]
                    this_created_time = getattr(data, 'pr_created_at')
                    last_created_time = list(rawData.loc[rawData['pr_number'] == last_num]['pr_created_at'])[
                        0]
                    gap = int(time.mktime(time.strptime(this_created_time, "%Y-%m-%d %H:%M:%S"))) - int(
                        time.mktime(time.strptime(last_created_time, "%Y-%m-%d %H:%M:%S")))
                author_submit_gap.append(gap)

                temp = rawData.loc[rawData['review_user_login'] == author].copy(deep=True)
                temp = temp.loc[temp['pr_number'] < pullNumber].copy(deep=True)
                review_num = temp.shape[0]
                author_review_count.append(review_num)
            rawData['author_push_count'] = author_push_count
            rawData['author_review_count'] = author_review_count
            rawData['author_submit_gap'] = author_submit_gap
            data = rawData

        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getMLDataPath(),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactCAData(projectName):
        """
        通过 ALL_{projectName}_data_pr_review_commit_file
             ALL_{projectName}_commit_file
             ALL_data_review_comment 三个文件拼接出CA需信息量的文件
        """

        """读取信息   只需要commit_file和pr_review和relation的信息"""
        time1 = datetime.now()
        data_train_path = projectConfig.getDataTrainPath()
        commit_file_data_path = projectConfig.getCommitFilePath()
        pr_commit_relation_path = projectConfig.getPrCommitRelationPath()
        prReviewData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
        prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
        print("raw pr review :", prReviewData.shape)

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

        """过滤状态非关闭的pr review"""
        prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", prReviewData.shape)

        """过滤pr 作者就是reviewer的情况"""
        prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                        != prReviewData['review_user_login']].copy(deep=True)
        print("after fliter author:", prReviewData.shape)

        """过滤不需要的字段"""
        prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at']].copy(deep=True)
        prReviewData.drop_duplicates(inplace=True)
        prReviewData.reset_index(drop=True, inplace=True)
        print("after fliter pr_review:", prReviewData.shape)

        commitFileData = commitFileData[['commit_sha', 'file_filename']].copy(deep=True)
        commitFileData.drop_duplicates(inplace=True)
        commitFileData.reset_index(drop=True, inplace=True)
        print("after fliter commit_file:", commitFileData.shape)

        """做三者连接"""
        data = pandas.merge(prReviewData, commitPRRelationData, left_on='pr_number', right_on='pull_number')
        print("merge relation:", data.shape)
        data = pandas.merge(data, commitFileData, left_on='sha', right_on='commit_sha')
        data.reset_index(drop=True, inplace=True)
        data.drop(columns=['sha'], inplace=True)
        data.drop(columns=['pr_number'], inplace=True)
        print("交换位置")
        order = ['repo_full_name', 'pull_number', 'pr_created_at', 'review_user_login', 'commit_sha', 'file_filename']
        data = data[order]
        # print(data.columns)
        print("after merge:", data.shape)

        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getCADataPath(),
                                          targetFileName=f'CA_{projectName}_data', dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def convertLabelListToDataFrame(label_data, pull_list, maxNum):
        # maxNum 为候选者的数量，会有答案不在名单的可能
        ar = numpy.zeros((label_data.__len__(), maxNum), dtype=int)
        pos = 0
        for pull_num in pull_list:
            labels = label_data[pull_num]
            for label in labels:
                if label <= maxNum:
                    ar[pos][label - 1] = 1
            pos += 1
        return ar

    @staticmethod
    def convertLabelListToListArray(label_data, pull_list):
        # maxNum 为候选者的数量，会有答案不在名单的可能
        answerList = []
        for pull_num in pull_list:
            answer = []
            labels = label_data[pull_num]
            for label in labels:
                answer.append(label)
            answerList.append(answer)
        return answerList

    @staticmethod
    def getListFromProbable(probable, classList, k):  # 推荐k个
        recommendList = []
        for case in probable:
            max_index_list = list(map(lambda x: numpy.argwhere(case == x), heapq.nlargest(k, case)))
            caseList = []
            pos = 0
            while pos < k:
                item = max_index_list[pos]
                for i in item:
                    caseList.append(classList[i[0]])
                pos += item.shape[0]
            recommendList.append(caseList)
        return recommendList

    @staticmethod
    def convertMultilabelProbaToDataArray(probable):  # 推荐k个
        """这个格式是sklearn 多标签的可能性预测结果 转换成通用格式"""
        result = numpy.empty((probable[0].shape[0], probable.__len__()))
        y = 0
        for pro in probable:
            x = 0
            for p in pro[:, 1]:
                result[x][y] = p
                x += 1
            y += 1
        return result

    @staticmethod
    def contactIRData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):
        """
        对于 label == review comment

        通过 ALL_{projectName}_data_pr_review_commit_file
             ALL_{projectName}_commit_file
             ALL_data_review_comment 三个文件拼接出IR所需信息量的文件

        对于 label == review comment and issue comment
             ALL_{projectName}_data_pullrequest
             ALL_{projectName}_data_issuecomment
             ALL_{projectName}_data_review
             三个文件拼出IR所需的信息量文件
        """

        targetFileName = f'IR_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'IR_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'IR_ALL_{projectName}_data'

        """读取信息  IR 只需要pr 的title和body的信息"""
        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            prReviewData = pandasHelper.readTSVFile(
                os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
            prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
            print("raw pr review :", prReviewData.shape)

        """issue commit 数据库输出 自带抬头"""
        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """pull request 数据库输出 自带抬头"""
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ review 数据库输出 自带抬头"""
        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_REVIEW_COMMENT:
            """过滤状态非关闭的pr review"""
            prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
            print("after fliter closed pr:", prReviewData.shape)

            """过滤pr 作者就是reviewer的情况"""
            prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                            != prReviewData['review_user_login']].copy(deep=True)
            print("after fliter author:", prReviewData.shape)

            """过滤不需要的字段"""
            prReviewData = prReviewData[
                ['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(
                deep=True)
            prReviewData.drop_duplicates(inplace=True)
            prReviewData.reset_index(drop=True, inplace=True)
            print("after fliter pr_review:", prReviewData.shape)
            data = prReviewData
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            """思路  上面两部分依次做凭借， 最后加上文件"""
            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)
            """过滤删除用户的场景"""
            data_issue.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景"""
            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "IR数据行： pr_number, review_user_login, pr_title, pr_body, pr_created_at"
            data_issue = data_issue[['number', 'title', 'body_x', 'created_at_x', 'user_login_y']].copy(deep=True)
            data_issue.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)
            """过滤删除用户场景"""
            data_review.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景  """
            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            data_review = data_review[['number', 'title', 'body_x', 'created_at', 'user_login_y']].copy(deep=True)
            data_review.columns = ['pr_number', 'pr_title', 'pr_body', 'pr_created_at', 'review_user_login']
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 轴合并
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            """只选出感兴趣的部分"""
            data = data[['pr_number', 'review_user_login', 'pr_title', 'pr_body', 'pr_created_at']].copy(deep=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)

        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getIRDataPath(),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def contactPBData(projectName, label=StringKeyUtils.STR_LABEL_REVIEW_COMMENT):
        """
        对于 label == review comment and issue comment
             ALL_{projectName}_data_pullrequest
             ALL_{projectName}_data_issuecomment
             ALL_{projectName}_data_review
             ALL_{projectName}_data_review_comment
             三个文件拼出PB所需的信息量文件
        """

        targetFileName = f'PB_{projectName}_data'
        if label == StringKeyUtils.STR_LABEL_ISSUE_COMMENT:
            targetFileName = f'PB_ISSUE_{projectName}_data'
        elif label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            targetFileName = f'PB_ALL_{projectName}_data'

        """读取信息对应的信息"""
        data_train_path = projectConfig.getDataTrainPath()
        issue_comment_path = projectConfig.getIssueCommentPath()
        pull_request_path = projectConfig.getPullRequestPath()
        review_path = projectConfig.getReviewDataPath()
        review_comment_path = projectConfig.getReviewCommentDataPath()

        """issue commit 数据库输出 自带抬头"""
        issueCommentData = pandasHelper.readTSVFile(
            os.path.join(issue_comment_path, f'ALL_{projectName}_data_issuecomment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """pull request 数据库输出 自带抬头"""
        pullRequestData = pandasHelper.readTSVFile(
            os.path.join(pull_request_path, f'ALL_{projectName}_data_pullrequest.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ review 数据库输出 自带抬头"""
        reviewData = pandasHelper.readTSVFile(
            os.path.join(review_path, f'ALL_{projectName}_data_review.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        """ review comment 数据库输出 自带抬头"""
        reviewCommentData = pandasHelper.readTSVFile(
            os.path.join(review_comment_path, f'ALL_{projectName}_data_review_comment.tsv'),
            pandasHelper.INT_READ_FILE_WITH_HEAD, low_memory=False
        )

        if label == StringKeyUtils.STR_LABEL_ALL_COMMENT:
            """思路  上面两部分依次做凭借， 最后加上文件"""
            data_issue = pandas.merge(pullRequestData, issueCommentData, left_on='number', right_on='pull_number')
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_issue = data_issue.loc[data_issue['closed_at'] >= data_issue['created_at_y']].copy(deep=True)
            data_issue = data_issue.loc[data_issue['user_login_x'] != data_issue['user_login_y']].copy(deep=True)
            """过滤删除用户的场景"""
            data_issue.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景"""
            data_issue['isBot'] = data_issue['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_issue = data_issue.loc[data_issue['isBot'] == False].copy(deep=True)
            "PR数据行： repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_issue = data_issue[['repo_full_name_x', 'number', 'title', 'body_x',
                                     'created_at_x', 'user_login_y', 'body_y']].copy(deep=True)
            data_issue.columns = ['repo_full_name', 'pr_number', 'pr_title', 'pr_body',
                                  'pr_created_at', 'review_user_login', 'comment_body']
            data_issue.drop_duplicates(inplace=True)

            data_review = pandas.merge(pullRequestData, reviewData, left_on='number', right_on='pull_number')
            data_review = pandas.merge(data_review, reviewCommentData, left_on='id_y', right_on='pull_request_review_id')
            data_review = data_review.loc[data_review['user_login_x'] != data_review['user_login_y']].copy(deep=True)
            """过滤 comment 在closed 后面的场景 2020.6.28"""
            data_review = data_review.loc[data_review['closed_at'] >= data_review['submitted_at']].copy(deep=True)
            """过滤删除用户场景"""
            data_review.dropna(subset=['user_login_y'], inplace=True)
            """过滤机器人的场景  """
            data_review['isBot'] = data_review['user_login_y'].apply(lambda x: BotUserRecognizer.isBot(x))
            data_review = data_review.loc[data_review['isBot'] == False].copy(deep=True)
            "PR数据行： repo_full_name, number, review_user_login, pr_title, pr_body, pr_created_at, comment_body"
            data_review = data_review[['repo_full_name_x', 'number', 'title', 'body_x',
                                       'created_at_x', 'user_login_y', 'body']].copy(deep=True)
            data_review.columns = ['repo_full_name', 'pr_number', 'pr_title', 'pr_body',
                                   'pr_created_at', 'review_user_login', 'comment_body']
            data_review.drop_duplicates(inplace=True)

            data = pandas.concat([data_issue, data_review], axis=0)  # 0 轴合并
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            print(data.shape)

            """只选出感兴趣的部分"""
            data = data[['repo_full_name', 'pr_number', 'review_user_login', 'pr_title',
                         'pr_body', 'pr_created_at', 'comment_body']].copy(deep=True)
            data.sort_values(by='pr_number', ascending=False, inplace=True)
            data.reset_index(drop=True)

        """按照时间分成小片"""
        DataProcessUtils.splitDataByMonth(filename=None, targetPath=projectConfig.getPBDataPath(),
                                          targetFileName=targetFileName, dateCol='pr_created_at',
                                          dataFrame=data)

    @staticmethod
    def getReviewerFrequencyDict(projectName, date):
        """获得某个项目某个时间段的reviewer
        的review次数字典
        用于后面reviewer推荐排序
        date 处理时候不含最后一个月  作为兼容 [y1,m1,y2,m2)
        通过 ALL_{projectName}_data_pr_review_commit_file
        """

        data_train_path = projectConfig.getDataTrainPath()
        prReviewData = pandasHelper.readTSVFile(
            os.path.join(data_train_path, f'ALL_{projectName}_data_pr_review_commit_file.tsv'), low_memory=False)
        prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
        print("raw pr review :", prReviewData.shape)

        """过滤状态非关闭的pr review"""
        prReviewData = prReviewData.loc[prReviewData['pr_state'] == 'closed'].copy(deep=True)
        print("after fliter closed pr:", prReviewData.shape)

        """过滤pr 作者就是reviewer的情况"""
        prReviewData = prReviewData.loc[prReviewData['pr_user_login']
                                        != prReviewData['review_user_login']].copy(deep=True)
        print("after fliter author:", prReviewData.shape)

        """只留下 pr reviewer created_time"""
        prReviewData = prReviewData[['pr_number', 'review_user_login', 'pr_created_at']].copy(
            deep=True)
        prReviewData.drop_duplicates(inplace=True)
        prReviewData.reset_index(drop=True, inplace=True)
        print(prReviewData.shape)

        minYear, minMonth, maxYear, maxMonth = date
        """过滤不在的时间"""
        start = minYear * 12 + minMonth
        end = maxYear * 12 + maxMonth
        prReviewData['label'] = prReviewData['pr_created_at'].apply(lambda x: (time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        prReviewData['label_y'] = prReviewData['label'].apply(lambda x: x.tm_year)
        prReviewData['label_m'] = prReviewData['label'].apply(lambda x: x.tm_mon)
        data = None
        for i in range(start, end):
            y = int((i - i % 12) / 12)
            m = i % 12
            if m == 0:
                m = 12
                y = y - 1
            print(y, m)
            subDf = prReviewData.loc[(prReviewData['label_y'] == y) & (prReviewData['label_m'] == m)].copy(deep=True)
            if data is None:
                data = subDf
            else:
                data = pandas.concat([data, subDf])

        # print(data)
        reviewers = data['review_user_login'].value_counts()
        return dict(reviewers)

    @staticmethod
    def getStopWordList():
        stopwords = SplitWordHelper().getEnglishStopList()  # 获取通用英语停用词
        allStr = '['
        len = 0
        for word in stopwords:
            len += 1
            allStr += '\"'
            allStr += word
            allStr += '\"'
            if len != stopwords.__len__():
                allStr += ','
        allStr += ']'
        print(allStr)

    @staticmethod
    def dunn():
        # dunn 检验
        filename = os.path.join(projectConfig.getDataPath(), "compare.xlsx")
        data = pandas.read_excel(filename, sheet_name="Top-1_1")
        print(data)
        data.columns = [0, 1, 2, 3, 4]
        data.index = [0, 1, 2, 3, 4, 5, 6, 7]
        print(data)
        x = [[1, 2, 3, 5, 1], [12, 31, 54, 12], [10, 12, 6, 74, 11]]
        print(data.values.T)
        result = scikit_posthocs.posthoc_nemenyi_friedman(data.values)
        print(result)
        print(data.values.T[1])
        print(data.values.T[3])
        data1 = []
        for i in range(0, 5):
            data1.append([])
            for j in range(0, 5):
                if i == j:
                    data1[i].append(numpy.nan)
                    continue
                statistic, pvalue = wilcoxon(data.values.T[i], data.values.T[j])
                print(pvalue)
                data1[i].append(pvalue)
        data1 = pandas.DataFrame(data1)
        print(data1)
        import matplotlib.pyplot as plt
        name = ['FPS', 'IR', 'SVM', 'RF', 'CB']
        # scikit_posthocs.sign_plot(result, g=name)
        # plt.show()
        for i in range(0, 5):
            data1[i][i] = numpy.nan
        ax = seaborn.heatmap(data1, annot=True, vmax=1, square=True, yticklabels=name, xticklabels=name, cmap='GnBu_r')
        ax.set_title("Mann Whitney U test")
        plt.show()

    @staticmethod
    def compareDataFrameByPullNumber():
        """计算 两个 dataframe 的差异"""
        file2 = 'FPS_ALL_opencv_data_2017_10_to_2017_10.tsv'
        file1 = 'FPS_SEAA_opencv_data_2017_10_to_2017_10.tsv'
        df1 = pandasHelper.readTSVFile(projectConfig.getFPSDataPath() + os.sep + file1,
                                       header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        df1.drop(columns=['pr_created_at', 'commit_sha'], inplace=True)
        df2 = pandasHelper.readTSVFile(projectConfig.getFPSDataPath() + os.sep + file2,
                                       header=pandasHelper.INT_READ_FILE_WITH_HEAD)
        df2.drop(columns=['pr_created_at', 'commit_sha'], inplace=True)

        df1 = pandas.concat([df1, df2])
        df1 = pandas.concat([df1, df2])
        df1.drop_duplicates(inplace=True, keep=False)
        print(df1)

    @staticmethod
    def changeTriggerAnalyzer(repo):
        """对change trigger 数据做统计"""
        change_trigger_filename = projectConfig.getPRTimeLineDataPath() + os.sep + f'ALL_{repo}_data_pr_change_trigger.tsv'
        change_trigger_df = pandasHelper.readTSVFile(fileName=change_trigger_filename, header=0)

        prs = list(set(change_trigger_df['pullrequest_node']))
        print("prs nums:", prs.__len__())

        """"依照 issue comment 和  review comment 划分"""
        df_issue = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_issue_comment']
        print("issue all:", df_issue.shape[0])
        issue_is_change_count = df_issue.loc[df_issue['change_trigger'] == 1].shape[0]
        issue_not_change_count = df_issue.loc[df_issue['change_trigger'] == -1].shape[0]
        print("issue is count:", issue_is_change_count, " not count:", issue_not_change_count)
        plt.subplot(121)
        x = ['useful', 'useless']
        plt.bar(x=x, height=[issue_is_change_count, issue_not_change_count])
        plt.title(f'issue comment({repo})')
        for a, b in zip(x, [issue_is_change_count, issue_not_change_count]):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        # plt.show()

        df_review = change_trigger_df.loc[change_trigger_df['comment_type'] == 'label_review_comment']
        print("review all:", df_review.shape[0])
        x = range(-1, 11)
        y = []
        for i in x:
            y.append(df_review.loc[df_review['change_trigger'] == i].shape[0])
        plt.subplot(122)
        plt.bar(x=x, height=y)
        plt.title(f'review comment({repo})')
        for a, b in zip(x, y):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.show()





if __name__ == '__main__':
    # DataProcessUtils.splitDataByMonth(projectConfig.getRootPath() + r'\data\train\ALL_rails_data.tsv',
    #                                   projectConfig.getRootPath() + r'\data\train\all' + os.sep, hasHead=True)
    #
    # print(pandasHelper.readTSVFile(
    #     projectConfig.getRootPath() + r'\data\train\all\ALL_scala_data_2012_6_to_2012_6.tsv', ))
    #
    # DataProcessUtils.contactReviewCommentData('rails')
    #

    # """从总的commit file文件中分割树独立的commit file文件"""
    # DataProcessUtils.splitProjectCommitFileData('infinispan')

    """分割不同算法的训练集"""
    # DataProcessUtils.contactCAData('cakephp')

    # projects = ['opencv', 'adobe', 'angular', 'bitcoin', 'cakephp']
    # projects = ['bitcoin']
    # for p in projects:
    #     DataProcessUtils.contactMLData(p, label=StringKeyUtils.STR_LABEL_ALL_COMMENT)

    # DataProcessUtils.contactMLData('xbmc')
    # DataProcessUtils.contactFPSData('cakephp', label=StringKeyUtils.STR_LABEL_ALL_COMMENT)
    #
    # DataProcessUtils.getReviewerFrequencyDict('rails', (2019, 4, 2019, 6))
    # DataProcessUtils.getStopWordList()
    # DataProcessUtils.dunn()
    #
    # DataProcessUtils.compareDataFrameByPullNumber()

    DataProcessUtils.changeTriggerAnalyzer('cakephp')
