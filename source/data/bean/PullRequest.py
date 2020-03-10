# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from  source.data.bean.User import  User
from source.data.bean.Branch import  Branch


class PullRequest(BeanBase):
    """用于pull request 的数据类  共计23个属性
    """

    def __init__(self):
        self.repo_full_name = None
        self.number = None
        self.id = None
        self.node_id = None
        self.state = None
        self.title = None
        self.user = None
        self.body = None
        self.created_at = None
        self.updated_at = None
        self.closed_at = None
        self.merged_at = None
        self.merge_commit_sha = None
        self.author_association = None
        self.merged = None
        self.comments = None
        self.review_comments = None
        self.commits = None
        self.additions = None
        self.deletions = None
        self.changed_files = None
        self.head = None
        self.base = None

        self.user_login = None
        self.head_label = None
        self.base_label = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_NUMBER]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_REPO_FULL_NAME, StringKeyUtils.STR_KEY_NUMBER, StringKeyUtils.STR_KEY_ID,
                 StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_STATE,
                 StringKeyUtils.STR_KEY_TITLE, StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_BODY,
                 StringKeyUtils.STR_KEY_CREATE_AT, StringKeyUtils.STR_KEY_UPDATE_AT, StringKeyUtils.STR_KEY_CLOSED_AT,
                 StringKeyUtils.STR_KEY_MERGED_AT, StringKeyUtils.STR_KEY_MERGE_COMMIT_SHA,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, StringKeyUtils.STR_KEY_MERGED,
                 StringKeyUtils.STR_KEY_COMMENTS, StringKeyUtils.STR_KEY_REVIEW_COMMENTS,
                 StringKeyUtils.STR_KEY_COMMITS, StringKeyUtils.STR_KEY_ADDITIONS, StringKeyUtils.STR_KEY_DELETIONS,
                 StringKeyUtils.STR_KEY_CHANGED_FILES, StringKeyUtils.STR_KEY_HEAD_LABEL,
                 StringKeyUtils.STR_KEY_BASE_LABEL]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NUMBER, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_STATE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_TITLE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BODY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_CLOSED_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_MERGED_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_MERGE_COMMIT_SHA, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_MERGED, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_COMMENTS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_REVIEW_COMMENTS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_COMMITS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ADDITIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_DELETIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_CHANGED_FILES, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_HEAD_LABEL, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BASE_LABEL, BeanBase.DATA_TYPE_STRING)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name,StringKeyUtils.STR_KEY_NUMBER: self.number,
                 StringKeyUtils.STR_KEY_ID: self.id, StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_STATE: self.state, StringKeyUtils.STR_KEY_TITLE: self.title,
                 StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login, StringKeyUtils.STR_KEY_BODY: self.body,
                 StringKeyUtils.STR_KEY_CREATE_AT: self.created_at, StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at,
                 StringKeyUtils.STR_KEY_CLOSED_AT: self.closed_at, StringKeyUtils.STR_KEY_MERGED_AT: self.merged_at,
                 StringKeyUtils.STR_KEY_MERGE_COMMIT_SHA: self.merge_commit_sha,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION: self.author_association,
                 StringKeyUtils.STR_KEY_MERGED: self.merged, StringKeyUtils.STR_KEY_COMMENTS: self.comments,
                 StringKeyUtils.STR_KEY_REVIEW_COMMENTS: self.review_comments,
                 StringKeyUtils.STR_KEY_COMMITS: self.commits, StringKeyUtils.STR_KEY_ADDITIONS: self.additions,
                 StringKeyUtils.STR_KEY_DELETIONS: self.deletions,
                 StringKeyUtils.STR_KEY_CHANGED_FILES: self.changed_files,
                 StringKeyUtils.STR_KEY_HEAD_LABEL: self.head_label, StringKeyUtils.STR_KEY_BASE_LABEL: self.base_label}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = PullRequest()
                res.number = src.get(StringKeyUtils.STR_KEY_NUMBER, None)
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)
                res.state = src.get(StringKeyUtils.STR_KEY_STATE, None)
                res.title = src.get(StringKeyUtils.STR_KEY_TITLE, None)
                # user
                # user_id
                res.body = src.get(StringKeyUtils.STR_KEY_BODY, None)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT, None)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT, None)
                res.closed_at = src.get(StringKeyUtils.STR_KEY_CLOSED_AT, None)
                res.merged_at = src.get(StringKeyUtils.STR_KEY_MERGED_AT, None)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.closed_at is not None:
                    res.closed_at = datetime.strptime(res.closed_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.merged_at is not None:
                    res.merged_at = datetime.strptime(res.merged_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.merge_commit_sha = src.get(StringKeyUtils.STR_KEY_MERGE_COMMIT_SHA, None)
                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, None)
                res.merged = src.get(StringKeyUtils.STR_KEY_MERGED, None)
                res.comments = src.get(StringKeyUtils.STR_KEY_COMMENTS, None)
                res.review_comments = src.get(StringKeyUtils.STR_KEY_REVIEW_COMMENTS, None)
                res.commits = src.get(StringKeyUtils.STR_KEY_COMMITS, None)
                res.additions = src.get(StringKeyUtils.STR_KEY_ADDITIONS, None)
                res.deletions = src.get(StringKeyUtils.STR_KEY_DELETIONS, None)
                res.changed_files = src.get(StringKeyUtils.STR_KEY_CHANGED_FILES, None)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    user = User.parser.parser(userData)
                    res.user = user
                    res.user_login = user.login

                # res.head
                # res.base

                headData = src.get(StringKeyUtils.STR_KEY_HEAD, None)
                if headData is not None and isinstance(headData, dict):
                    head = Branch.parser.parser(headData)
                    res.head = head
                    res.head_label = head.label

                baseData = src.get(StringKeyUtils.STR_KEY_BASE, None)
                if baseData is not None and isinstance(baseData, dict):
                    base = Branch.parser.parser(baseData)
                    res.base = base
                    res.base_label = base.label

            return res


