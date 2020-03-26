# coding=gbk
import asyncio

from source.config.projectConfig import projectConfig
from source.data.service.AsyncSqlHelper import AsyncSqlHelper
from source.database.AsyncSqlExecuteHelper import getMysqlObj, AsyncSqlExecuteHelper
from source.utils.pandas.pandasHelper import pandasHelper
import matplotlib.pyplot as plt


def query(project, startYear, startMonth, endYear, endMonth):
    loop = asyncio.get_event_loop()
    task = [asyncio.ensure_future(queryByMonth(loop, project, startYear, startMonth, endYear, endMonth))]
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


if __name__ == '__main__':
    raw = pandasHelper.readTSVFile(projectConfig.getRootPath() + r'\data\train\pullrequest_bitcoin.tsv')
    data = raw.as_matrix()
    y1 = data[:, 1]
    plt.plot(y1)
    plt.show()

    query('bitcoin/bitcoin', 2017, 2, 2020, 3)
