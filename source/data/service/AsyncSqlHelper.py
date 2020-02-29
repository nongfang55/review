# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.data.bean.Branch import Branch
from source.data.bean.CommentRelation import CommitRelation
from source.data.bean.Commit import Commit
from source.data.bean.CommitComment import CommitComment
from source.data.bean.CommitPRRelation import CommitPRRelation
from source.data.bean.File import File
from source.data.bean.IssueComment import IssueComment
from source.data.bean.PullRequest import PullRequest
from source.data.bean.Repository import Repository
from source.data.bean.Review import Review
from source.data.bean.ReviewComment import ReviewComment
from source.data.bean.User import User
from source.database.SqlExecuteHelper import SqlExecuteHelper
from source.database.SqlUtils import SqlUtils


class AsyncSqlHelper:
    """用于获得不同bean类的插入表名字"""

    @staticmethod
    def getInsertTableName(bean):
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
