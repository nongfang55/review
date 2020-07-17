# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.data.service.TextCompareUtils import TextCompareUtils
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User


class ReviewComment(BeanBase):
    """github 中review comment数据类 共21个"""

    def __init__(self):
        self.id = None
        self.user = None
        self.body = None
        self.pull_request_review_id = None
        self.diff_hunk = None
        self.path = None
        self.commit_id = None
        self.position = None
        self.original_position = None
        self.original_commit_id = None
        self.created_at = None
        self.updated_at = None
        self.author_association = None
        self.start_line = None
        self.original_start_line = None
        self.start_side = None
        self.line = None
        self.original_line = None
        self.side = None
        self.in_reply_to_id = None
        self.node_id = None
        self.repo_full_name = None

        self.user_login = None
        self.change_trigger = None  # comment 之后一系列改动中距离comment所指的line最近的距离
        self.pull_request_review_node_id = None  # 考虑Review Thread没有id，只有node_id 只能通过以上方法连接
        self.temp_original_line = None # 用于处理 LEFT RIGHT 的转换，中间变量

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_ID, StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_BODY,
                 StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID, StringKeyUtils.STR_KEY_DIFF_HUNK,
                 StringKeyUtils.STR_KEY_PATH, StringKeyUtils.STR_KEY_COMMIT_ID, StringKeyUtils.STR_KEY_POSITION,
                 StringKeyUtils.STR_KEY_ORIGINAL_POSITION, StringKeyUtils.STR_KEY_ORIGINAL_COMMIT_ID,
                 StringKeyUtils.STR_KEY_CREATE_AT, StringKeyUtils.STR_KEY_UPDATE_AT,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, StringKeyUtils.STR_KEY_START_LINE,
                 StringKeyUtils.STR_KEY_ORIGINAL_START_LINE, StringKeyUtils.STR_KEY_START_SIDE,
                 StringKeyUtils.STR_KEY_LINE, StringKeyUtils.STR_KEY_ORIGINAL_LINE, StringKeyUtils.STR_KEY_SIDE,
                 StringKeyUtils.STR_KEY_IN_REPLY_TO_ID, StringKeyUtils.STR_KEY_NODE_ID,
                 StringKeyUtils.STR_KEY_CHANGE_TRIGGER, StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_NODE_ID,
                 StringKeyUtils.STR_KEY_REPO_FULL_NAME]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_USER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BODY, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_DIFF_HUNK, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_PATH, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_COMMIT_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_POSITION, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ORIGINAL_POSITION, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ORIGINAL_COMMIT_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CREATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_UPDATE_AT, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_START_LINE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ORIGINAL_START_LINE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_START_SIDE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_LINE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ORIGINAL_LINE, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_SIDE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_IN_REPLY_TO_ID, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHANGE_TRIGGER, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_ID: self.id, StringKeyUtils.STR_KEY_USER_LOGIN: self.user_login,
                 StringKeyUtils.STR_KEY_BODY: self.body,
                 StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID: self.pull_request_review_id,
                 StringKeyUtils.STR_KEY_DIFF_HUNK: self.diff_hunk, StringKeyUtils.STR_KEY_PATH: self.path,
                 StringKeyUtils.STR_KEY_COMMIT_ID: self.commit_id, StringKeyUtils.STR_KEY_POSITION: self.position,
                 StringKeyUtils.STR_KEY_ORIGINAL_POSITION: self.original_position,
                 StringKeyUtils.STR_KEY_ORIGINAL_COMMIT_ID: self.original_commit_id,
                 StringKeyUtils.STR_KEY_CREATE_AT: self.created_at, StringKeyUtils.STR_KEY_UPDATE_AT: self.updated_at,
                 StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION: self.author_association,
                 StringKeyUtils.STR_KEY_START_LINE: self.start_line,
                 StringKeyUtils.STR_KEY_ORIGINAL_START_LINE: self.original_start_line,
                 StringKeyUtils.STR_KEY_START_SIDE: self.start_side, StringKeyUtils.STR_KEY_LINE: self.line,
                 StringKeyUtils.STR_KEY_ORIGINAL_LINE: self.original_line, StringKeyUtils.STR_KEY_SIDE: self.side,
                 StringKeyUtils.STR_KEY_IN_REPLY_TO_ID: self.in_reply_to_id,
                 StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_CHANGE_TRIGGER: self.change_trigger,
                 StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_NODE_ID: self.pull_request_review_node_id,
                 StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name}
        return items

    class parser(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = ReviewComment()
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)
                res.repo_full_name = src.get(StringKeyUtils.STR_KEY_REPO_FULL_NAME)
                res.body = src.get(StringKeyUtils.STR_KEY_BODY)
                res.pull_request_review_id = src.get(StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID)
                res.diff_hunk = src.get(StringKeyUtils.STR_KEY_DIFF_HUNK)
                res.path = src.get(StringKeyUtils.STR_KEY_PATH)
                res.commit_id = src.get(StringKeyUtils.STR_KEY_COMMIT_ID)
                res.position = src.get(StringKeyUtils.STR_KEY_POSITION)
                res.original_position = src.get(StringKeyUtils.STR_KEY_ORIGINAL_POSITION)
                res.original_commit_id = src.get(StringKeyUtils.STR_KEY_ORIGINAL_COMMIT_ID)
                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT)
                res.change_trigger = src.get(StringKeyUtils.STR_KEY_CHANGE_TRIGGER)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION)
                res.start_line = src.get(StringKeyUtils.STR_KEY_START_LINE)
                res.original_start_line = src.get(StringKeyUtils.STR_KEY_ORIGINAL_START_LINE)
                res.start_side = src.get(StringKeyUtils.STR_KEY_START_SIDE)
                res.line = src.get(StringKeyUtils.STR_KEY_LINE)
                res.original_line = src.get(StringKeyUtils.STR_KEY_ORIGINAL_LINE)
                res.side = src.get(StringKeyUtils.STR_KEY_SIDE)
                res.in_reply_to_id = src.get(StringKeyUtils.STR_KEY_IN_REPLY_TO_ID)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID)

                userData = src.get(StringKeyUtils.STR_KEY_USER, None)
                if userData is not None and isinstance(userData, dict):
                    res.user = User.parser.parser(userData)
                    res.user_login = res.user.login

            return res

    class parserV4(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = ReviewComment()
                res.id = src.get(StringKeyUtils.STR_KEY_DATABASE_ID, None)

                res.body = src.get(StringKeyUtils.STR_KEY_BODY)
                """pull_request_review_id 无法获取"""
                res.pull_request_review_id = src.get(StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_ID)
                res.diff_hunk = src.get(StringKeyUtils.STR_KEY_DIFF_HUNK_V4)
                res.path = src.get(StringKeyUtils.STR_KEY_PATH)
                """获取 commit id"""
                commit = src.get(StringKeyUtils.STR_KEY_COMMIT, None)
                if commit is not None and isinstance(commit, dict):
                    res.commit_id = commit.get(StringKeyUtils.STR_KEY_OID, None)

                res.position = src.get(StringKeyUtils.STR_KEY_POSITION)
                res.original_position = src.get(StringKeyUtils.STR_KEY_ORIGINAL_POSITION_V4)
                """获取 original commit id"""
                originalCommit = src.get(StringKeyUtils.STR_KEY_ORIGINAL_COMMIT, None)
                if originalCommit is not None and isinstance(originalCommit, dict):
                    res.original_commit_id = originalCommit.get(StringKeyUtils.STR_KEY_OID)

                res.created_at = src.get(StringKeyUtils.STR_KEY_CREATE_AT_V4)
                res.updated_at = src.get(StringKeyUtils.STR_KEY_UPDATE_AT_V4)
                res.change_trigger = src.get(StringKeyUtils.STR_KEY_CHANGE_TRIGGER)

                if res.created_at is not None:
                    res.created_at = datetime.strptime(res.created_at, StringKeyUtils.STR_STYLE_DATA_DATE)
                if res.updated_at is not None:
                    res.updated_at = datetime.strptime(res.updated_at, StringKeyUtils.STR_STYLE_DATA_DATE)

                res.author_association = src.get(StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION_V4)

                """关于 comment 行数的信息都无法获取"""

                """尝试拼接出 line 和 origin_line 两个属性"""
                res.start_line = src.get(StringKeyUtils.STR_KEY_START_LINE)
                res.original_start_line = src.get(StringKeyUtils.STR_KEY_ORIGINAL_START_LINE)
                res.start_side = src.get(StringKeyUtils.STR_KEY_START_SIDE)
                res.side = src.get(StringKeyUtils.STR_KEY_SIDE)

                # line, original_line = TextCompareUtils.getStartLine(res.diff_hunk, res.position,
                # res.original_position) res.line = line res.original_line = original_line
                """上面信息拼接需要review comment对应original_commit的patch 信息，这个地方无法获取"""
                res.line = None
                res.original_line = None

                """获取 in_replay_to_id 信息"""
                replyTo = src.get(StringKeyUtils.STR_KEY_IN_REPLY_TO_ID_V4)
                if replyTo is not None and isinstance(replyTo, dict):
                    res.in_reply_to_id = replyTo.get(StringKeyUtils.STR_KEY_DATABASE_ID)

                res.node_id = src.get(StringKeyUtils.STR_KEY_ID)

                """获取 user_login 信息"""
                userData = src.get(StringKeyUtils.STR_KEY_AUTHOR, None)
                if userData is not None and isinstance(userData, dict):
                    res.user = None
                    res.user_login = userData.get(StringKeyUtils.STR_KEY_LOGIN, None)

            return res
