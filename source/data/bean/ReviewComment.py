# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
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

        self.user_login = None
        self.change_trigger = None  # comment 之后一系列改动中距离comment所指的line最近的距离

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
                 StringKeyUtils.STR_KEY_CHANGE_TRIGGER]

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
                 (StringKeyUtils.STR_KEY_CHANGE_TRIGGER, BeanBase.DATA_TYPE_INT)]

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
                 StringKeyUtils.STR_KEY_CHANGE_TRIGGER: self.change_trigger}
        return items

    class parser(BeanBase.parser):
        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = ReviewComment()
                res.id = src.get(StringKeyUtils.STR_KEY_ID, None)

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
