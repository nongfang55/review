# coding=gbk
import asyncio
import os

from source.config.projectConfig import projectConfig
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.database.AsyncSqlExecuteHelper import getMysqlObj, AsyncSqlExecuteHelper
from source.scikit.ML.MLTrain import MLTrain
from source.scikit.service.DataProcessUtils import DataProcessUtils
from source.utils.pandas.pandasHelper import pandasHelper
import matplotlib.pyplot as plt


def query(project, startYear, startMonth, endYear, endMonth):
    loop = asyncio.get_event_loop()
    task = [asyncio.ensure_future(queryByMonth2(loop, project, startYear, startMonth, endYear, endMonth))]
    tasks = asyncio.gather(*task)
    loop.run_until_complete(tasks)


async def queryByMonth(loop, project, startYear, startMonth, endYear, endMonth):
    mysql = await getMysqlObj(loop)

    result = []
    y1 = []
    y2 = []
    y3 = []
    x = []
    xx = []

    for year in range(startYear, endYear + 1):
        month1 = 1
        month2 = 12
        if startYear == year:
            month1 = startMonth
        if endYear == year:
            month2 = endMonth
        for month in range(month1, month2 + 1):
            x.append(f"{year}, {month}")
            xx.append(xx.__len__() + 1)
            sql_pr_number = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s"
            sql_pr_number_closed = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s and state = 'closed'"
            sql_review_closed = "select count(*) from pullRequest,review where review.repo_full_name = %s and YEAR(created_at) = %s " \
                                "and MONTH(created_at) = %s and pullRequest.state = 'closed' and review.repo_full_name = pullRequest.repo_full_name" \
                                " and review.pull_number = pullRequest.number"
            pr_number = await AsyncSqlHelper.query(mysql, sql_pr_number, (project, year, month))
            pr_number = pr_number[0][0]
            print(f" year:{year}, month:{month}")
            print(f"pr_number:{pr_number}")
            y1.append(pr_number)
            pr_number_closed = await AsyncSqlHelper.query(mysql, sql_pr_number_closed, (project, year, month))
            pr_number_closed = pr_number_closed[0][0]
            print(f"pr_number_closed:{pr_number_closed}")
            y2.append(pr_number_closed)
            review_number_closed = await AsyncSqlHelper.query(mysql, sql_review_closed, (project, year, month))
            review_number_closed = review_number_closed[0][0]
            print(f"review_number_closed:{review_number_closed}")
            y3.append(review_number_closed)
            result.append([year, month, pr_number, pr_number_closed, review_number_closed])

    plt.plot(x, y1, label="pr number")
    plt.plot(x, y2, label="closed pr number")
    plt.plot(x, y3, label="closed PR review number")
    plt.xticks(xx, x, rotation='90', size='small')
    plt.title('')
    plt.margins(0)
    plt.ylabel('number')
    plt.xlabel('time')
    plt.show()

    for i in result:
        print(i)


# async def pr_review_ratio_plot(loop, project, startYear, startMonth, endYear, endMonth):
#     """用于绘画pr和reivew的数量比例"""
#     mysql = await getMysqlObj(loop)
#
#     result = []
#     y1 = []
#     y2 = []
#     y3 = []
#     x = []
#     xx = []
#
#     data_train_path = projectConfig.getDataTrainPath()
#     prReviewData = pandasHelper.readTSVFile(
#         os.path.join(data_train_path, f'ALL_{project}_data_pr_review_commit_file.tsv'), low_memory=False)
#     prReviewData.columns = DataProcessUtils.COLUMN_NAME_PR_REVIEW_COMMIT_FILE
#     print("raw pr review :", prReviewData.shape)
#
#     """只留下 pr reviewer created_time"""
#     prReviewData = prReviewData[['pr_number', 'pr_created_at', 'review_id']].copy(
#         deep=True)
#     prReviewData.drop_duplicates(inplace=True)
#     prReviewData.reset_index(drop=True, inplace=True)
#     print(prReviewData.shape)
#
#     """过滤对应年的pr   这里月暂时不做考虑"""
#     prReviewData['label'] = prReviewData['pr_created_at'].apply(
#         lambda x: (prReviewData.strptime(x, "%Y-%m-%d %H:%M:%S").tm_year == date[2] and
#                    time.strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon == date[3]))
#
#
#     for year in range(startYear, endYear + 1):
#         month1 = 1
#         month2 = 12
#         if startYear == year:
#             month1 = startMonth
#         if endYear == year:
#             month2 = endMonth
#         for month in range(month1, month2 + 1):
#             x.append(f"{year}, {month}")
#             xx.append(xx.__len__() + 1)
#             sql_pr_number = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s"
#             sql_pr_number_closed = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s and state = 'closed'"
#             sql_review_closed = "select count(*) from pullRequest,review where review.repo_full_name = %s and YEAR(created_at) = %s " \
#                                 "and MONTH(created_at) = %s and pullRequest.state = 'closed' and review.repo_full_name = pullRequest.repo_full_name" \
#                                 " and review.pull_number = pullRequest.number"
#             pr_number = await AsyncSqlHelper.query(mysql, sql_pr_number, (project, year, month))
#             pr_number = pr_number[0][0]
#             print(f" year:{year}, month:{month}")
#             print(f"pr_number:{pr_number}")
#             y1.append(pr_number)
#             pr_number_closed = await AsyncSqlHelper.query(mysql, sql_pr_number_closed, (project, year, month))
#             pr_number_closed = pr_number_closed[0][0]
#             print(f"pr_number_closed:{pr_number_closed}")
#             y2.append(pr_number_closed)
#             review_number_closed = await AsyncSqlHelper.query(mysql, sql_review_closed, (project, year, month))
#             review_number_closed = review_number_closed[0][0]
#             print(f"review_number_closed:{review_number_closed}")
#             y3.append(review_number_closed)
#             result.append([year, month, pr_number, pr_number_closed, review_number_closed])
#
#     plt.plot(x, y1, label="pr number")
#     plt.plot(x, y2, label="closed pr number")
#     plt.plot(x, y3, label="closed PR review number")
#     plt.xticks(xx, x, rotation='90', size='small')
#     plt.title('')
#     plt.margins(0)
#     plt.ylabel('number')
#     plt.xlabel('time')
#     plt.show()
#
#     for i in result:
#         print(i)


async def queryByMonth2(loop, project, startYear, startMonth, endYear, endMonth):
    mysql = await getMysqlObj(loop)

    result = []
    y1 = []
    y2 = []
    y3 = []
    x = []
    xx = []

    for year in range(startYear, endYear + 1):
        month1 = 1
        month2 = 12
        if startYear == year:
            month1 = startMonth
        if endYear == year:
            month2 = endMonth
        for month in range(month1, month2 + 1):
            x.append(f"{year}, {month}")
            xx.append(xx.__len__() + 1)
            sql_pr_number = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s"
            # sql_pr_number_closed = "select count(*) from pullRequest where repo_full_name = %s and YEAR(created_at) = %s and MONTH(created_at) = %s and state = 'closed'"
            sql_commit_number = "select count(*) from pullRequest, commitPRRelation where pullRequest.repo_full_name = commitPRRelation.repo_full_name " \
                                "and pullRequest.repo_full_name = %s and" \
                                " pullRequest.number = commitPRRelation.pull_number and year(pullRequest.created_at) = %s" \
                                " and month(pullRequest.created_at)= %s"
            sql_review_closed = "select count(*) from pullRequest,review where review.repo_full_name = %s and YEAR(created_at) = %s " \
                                "and MONTH(created_at) = %s and pullRequest.state = 'closed' and review.repo_full_name = pullRequest.repo_full_name" \
                                " and review.pull_number = pullRequest.number"
            pr_number = await AsyncSqlHelper.query(mysql, sql_pr_number, (project, year, month))
            pr_number = pr_number[0][0]
            print(f" year:{year}, month:{month}")
            print(f"pr_number:{pr_number}")
            y1.append(pr_number)
            pr_number_closed = await AsyncSqlHelper.query(mysql, sql_commit_number, (project, year, month))
            pr_number_closed = pr_number_closed[0][0]
            print(f"pr_number_closed:{pr_number_closed}")
            y2.append(pr_number_closed)
            review_number_closed = await AsyncSqlHelper.query(mysql, sql_review_closed, (project, year, month))
            review_number_closed = review_number_closed[0][0]
            print(f"review_number_closed:{review_number_closed}")
            y3.append(review_number_closed)
            result.append([year, month, pr_number, pr_number_closed, review_number_closed])

    plt.plot(x, y1, label="pull-request number")
    plt.plot(x, y2, label="commit number")
    plt.plot(x, y3, label="review number")
    plt.legend()

    plt.xticks(xx, x, rotation='90', size='small')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(f'项目rails的评审信息走势图')
    plt.margins(0)
    plt.ylabel('数量')
    plt.xlabel('时间')
    plt.show()

    for i in result:
        print(i)


def pr_review_ratio():
    """绘制某个pr和对应review数量的分布"""
    train_path = projectConfig.getDataTrainPath()
    filename = os.path.join(train_path, 'pr_review_ratio_akka.tsv')
    df = pandasHelper.readTSVFile(filename)
    # MLTrain.getSeriesBarPlot(df[1])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    # fig.add_subplot(2, 1, 1)
    counts = df[1].value_counts()
    print(counts)
    counts.sort_index().plot(kind='bar')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('项目akka每一个pull-request对应的review数量')
    plt.xlabel('pull-request数量')
    plt.ylabel('review数量')
    plt.show()


if __name__ == '__main__':
    # raw = pandasHelper.readTSVFile(projectConfig.getRootPath() + r'\data\train\pullrequest_bitcoin.tsv')
    # data = raw.as_matrix()
    # y1 = data[:, 1]
    # plt.plot(y1)
    # plt.show()

    query('rails/rails', 2018, 1, 2019, 12)
    # pr_review_ratio()
