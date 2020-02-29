# coding=gbk
import asyncio
import time

from source.config.configPraser import configPraser
from source.data.service.ApiHelper import ApiHelper
from source.data.service.AsyncApiHelper import AsyncApiHelper
from source.database.AsyncSqlExecuteHelper import getMysqlObj


class AsyncProjectAllDataFetcher:
    # 获取项目的所有信息 主题信息采用异步获取

    @staticmethod
    def getDataForRepository(owner, repo, limit=-1, start=-1):

        if start == -1:
            # 获取项目pull request的数量 这里使用同步方法获取
            requestNumber = ApiHelper(owner, repo).getMaxSolvedPullRequestNumberForProject()
            print("total pull request number:", requestNumber)

            startNumber = requestNumber
        else:
            startNumber = start

        if limit == -1:
            limit = startNumber

        AsyncApiHelper.setRepo(owner, repo)
        t1 = time.time()

        loop = asyncio.get_event_loop()
        task = [asyncio.ensure_future(AsyncProjectAllDataFetcher.preProcess(loop, limit, start))]
        tasks = asyncio.gather(*task)
        loop.run_until_complete(tasks)

        t2 = time.time()
        print('cost time:', t2 - t1)

    @staticmethod
    async def preProcess(loop, limit, start):

        semaphore = asyncio.Semaphore(configPraser.getSemaphore())  # 对速度做出限制
        mysql = await getMysqlObj(loop)

        if configPraser.getPrintMode():
            print("mysql init success")

        tasks = [asyncio.ensure_future(AsyncApiHelper.downloadInformation(pull_number, semaphore, mysql))
                 for pull_number in range(start, max(start - limit, 0), -1)]
        await asyncio.wait(tasks)


if __name__ == '__main__':
    AsyncProjectAllDataFetcher.getDataForRepository(owner=configPraser.getOwner(), repo=configPraser.getRepo()
                                                    , start=configPraser.getStart(), limit=configPraser.getLimit())
