# coding=gbk
from source.config.configPraser import configPraser
from source.data.bean.Beanbase import BeanBase
from source.data.bean.Blob import Blob
from source.data.bean.Branch import Branch
from source.data.bean.CommentRelation import CommitRelation
from source.data.bean.Commit import Commit
from source.data.bean.CommitComment import CommitComment
from source.data.bean.CommitPRRelation import CommitPRRelation
from source.data.bean.File import File
from source.data.bean.HeadRefForcePushedEvent import HeadRefForcePushedEvent
from source.data.bean.IssueComment import IssueComment
from source.data.bean.PRChangeFile import PRChangeFile
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.data.bean.PullRequest import PullRequest
from source.data.bean.PullRequestCommit import PullRequestCommit
from source.data.bean.Repository import Repository
from source.data.bean.Review import Review
from source.data.bean.ReviewChangeRelation import ReviewChangeRelation
from source.data.bean.ReviewComment import ReviewComment
from source.data.bean.TreeEntry import TreeEntry
from source.data.bean.User import User
from source.data.bean.UserFollowRelation import UserFollowRelation
from source.database.SqlExecuteHelper import SqlExecuteHelper
from source.database.SqlUtils import SqlUtils


class AsyncSqlHelper:

    """异步数据库操作辅助类"""

    @staticmethod
    def getInsertTableName(bean):
        """用于获得不同bean类的插入表名字"""
        if isinstance(bean, Branch):
            return SqlUtils.STR_TABLE_NAME_BRANCH
        elif isinstance(bean, CommitRelation):
            return SqlUtils.STR_TABLE_NAME_COMMIT_RELATION
        elif isinstance(bean, Commit):
            return SqlUtils.STR_TABLE_NAME_COMMIT
        elif isinstance(bean, CommitComment):
            return SqlUtils.STR_TABLE_NAME_COMMIT_COMMENT
        elif isinstance(bean, CommitPRRelation):
            return SqlUtils.STR_TABLE_NAME_COMMIT_PR_RELATION
        elif isinstance(bean, File):
            return SqlUtils.STR_TABLE_NAME_FILE
        elif isinstance(bean, IssueComment):
            return SqlUtils.STR_TABLE_NAME_ISSUE_COMMENT
        elif isinstance(bean, PullRequest):
            return SqlUtils.STR_TABLE_NAME_PULL_REQUEST
        elif isinstance(bean, Repository):
            return SqlUtils.STR_TABLE_NAME_REPOS
        elif isinstance(bean, Review):
            return SqlUtils.STR_TABLE_NAME_REVIEW
        elif isinstance(bean, ReviewComment):
            return SqlUtils.STR_TABLE_NAME_REVIEW_COMMENT
        elif isinstance(bean, User):
            return SqlUtils.STR_TABLE_NAME_USER
        elif isinstance(bean, PRTimeLineRelation):
            return SqlUtils.STR_TABLE_NAME_PR_TIME_LINE
        elif isinstance(bean, HeadRefForcePushedEvent):
            return SqlUtils.STR_TABLE_NAME_HEAD_REF_FORCE_PUSHED_EVENT
        elif isinstance(bean, PullRequestCommit):
            return SqlUtils.STR_TABLE_NAME_PULL_REQUEST_COMMIT
        elif isinstance(bean, PRChangeFile):
            return SqlUtils.STR_TABLE_NAME_PR_CHANGE_FILE
        elif isinstance(bean, ReviewChangeRelation):
            return SqlUtils.STR_TABLE_NAME_REVIEW_CHANGE_RELATION
        elif isinstance(bean, TreeEntry):
            return SqlUtils.STR_TABLE_NAME_TREE_ENTRY
        elif isinstance(bean, Blob):
            return SqlUtils.STR_TABLE_NAME_GIT_BLOB
        elif isinstance(bean, UserFollowRelation):
            return SqlUtils.STR_TABLE_NAME_USER_FOLLOW_RELATION
        else:
            return None

    @staticmethod
    async def storeBeanData(bean, mysql):
        if bean is not None and isinstance(bean, BeanBase):
            await mysql.insertValuesIntoTable(AsyncSqlHelper.getInsertTableName(bean),
                                              bean.getItemKeyList(),
                                              bean.getValueDict(),
                                              bean.getIdentifyKeys())

            print("insert success")

    @staticmethod
    async def storeBeanDateList(beans, mysql):
        """一次性存储多个bean对象 讲道理结构是被破坏的，但是可以吧所有数据库请求压缩为一次"""

        conn, cur = await  mysql.getDatabaseConnected()

        try:
            for bean in beans:
                if isinstance(bean, BeanBase):
                    tableName = AsyncSqlHelper.getInsertTableName(bean)
                    items = bean.getItemKeyList()
                    valueDict = bean.getValueDict()

                    format_table = SqlUtils.getInsertTableFormatString(tableName, items)
                    format_values = SqlUtils.getInsertTableValuesString(items.__len__())

                    sql = SqlUtils.STR_SQL_INSERT_TABLE_UTILS.format(format_table, format_values)
                    if configPraser.getPrintMode():
                        print(sql)

                    values = ()
                    for item in items:
                        values = values + (valueDict.get(item, None),)  # 元组相加
                    try:
                        await cur.execute(sql, values)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
        finally:
            if cur:
                await cur.close()
            await mysql.pool.release(conn)

    @staticmethod
    async def queryBeanData(beans, mysql, defineItems=None):
        """一次性查询多个bean对象  define 为[[key1,key2], [key3,key4] ...]
        返回多个元组  [((),),((),())...]"""

        conn, cur = await  mysql.getDatabaseConnected()

        resultBeans = []

        try:
            pos = 0
            for bean in beans:
                if isinstance(bean, BeanBase):
                    tableName = AsyncSqlHelper.getInsertTableName(bean)
                    items = defineItems[pos]
                    if items is None:
                        items = bean.getIdentifyKeys()
                    pos += 1
                    valueDict = bean.getValueDict()

                    format_values = SqlUtils.getQueryTableConditionString(items)
                    sql = SqlUtils.STR_SQL_QUERY_TABLE_UTILS.format(tableName, format_values)

                    if configPraser.getPrintMode():
                        print(sql)

                    values = ()
                    for item in items:
                        values = values + (valueDict.get(item, None),)  # 元组相加
                    try:
                        await cur.execute(sql, values)
                        r = await cur.fetchall()
                        resultBeans.append(r)
                    except Exception as e:
                        print(e)
                        resultBeans.append(None)
        except Exception as e:
            print(e)
        finally:
            if cur:
                await cur.close()
            await mysql.pool.release(conn)

        return resultBeans

    @staticmethod
    async def updateBeanDateList(beans, mysql):
        """一次性更新多个bean对象 讲道理结构是被破坏的，但是可以吧所有数据库请求压缩为一次"""

        conn, cur = await  mysql.getDatabaseConnected()

        try:
            for bean in beans:
                if isinstance(bean, BeanBase):
                    tableName = AsyncSqlHelper.getInsertTableName(bean)
                    valueDict = bean.getValueDict()

                    format_target = SqlUtils.getUpdateTableSetString(bean.getItemKeyList())
                    format_condition = SqlUtils.getQueryTableConditionString(bean.getIdentifyKeys())
                    sql = SqlUtils.STR_SQL_UPDATE_TABLE_UTILS.format(tableName, format_target, format_condition)
                    if configPraser.getPrintMode():
                        print(sql)

                    values = ()
                    for item in bean.getItemKeyList():
                        values = values + (valueDict.get(item, None),)

                    for item in bean.getIdentifyKeys():
                        values = values + (valueDict.get(item, None),)  # 元组相加
                    try:
                        await cur.execute(sql, values)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
        finally:
            if cur:
                await cur.close()
            await mysql.pool.release(conn)

    @staticmethod
    async def query(mysql, sql, values):  # 纯粹通过sql语句执行返回结果
        conn, cur = await  mysql.getDatabaseConnected()
        r = None
        try:
            try:
                await cur.execute(sql,values)
                r = await cur.fetchall()
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
        finally:
            if cur:
                await cur.close()
            await mysql.pool.release(conn)
        return r
