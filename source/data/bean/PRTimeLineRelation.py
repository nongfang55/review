# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.data.bean.HeadRefForcePushedEvent import HeadRefForcePushedEvent
from source.data.bean.IssueComment import IssueComment
from source.data.bean.PullRequestCommit import PullRequestCommit
from source.utils.StringKeyUtils import StringKeyUtils
import json


class PRTimeLineRelation(BeanBase):
    """github中pull request的timeline 关系"""

    def __init__(self):
        self.pull_request_node = None
        self.timeline_item_node = None
        self.typename = None
        self.position = None
        self.origin = None

        """可选属性 做简化使用的 实际不进入存储"""
        self.headRefForcePushedEventAfterCommit = None
        self.headRefForcePushedEventBeforeCommit = None
        self.pull_request_review_commit = None
        self.pull_request_review_original_commit = None
        self.pull_request_commit = None
        self.merge_commit = None
        self.user_login = None
        self.comments = []

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE,
                 StringKeyUtils.STR_KEY_TYPE_NAME, StringKeyUtils.STR_KEY_POSITION, StringKeyUtils.STR_KEY_ORIGIN]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_TYPE_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_POSITION, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_ORIGIN, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_PULL_REQUEST_NODE: self.pull_request_node,
                 StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE: self.timeline_item_node,
                 StringKeyUtils.STR_KEY_TYPE_NAME: self.typename,
                 StringKeyUtils.STR_KEY_POSITION: self.position,
                 StringKeyUtils.STR_KEY_ORIGIN: self.origin}

        return items

    class Parser(BeanBase.parser):

        @staticmethod
        def parser(item):
            relation = PRTimeLineRelation()  # 返回结果为一系列关系

            """依据每个Item的TypeName来判断Item的具体类型"""
            """item的类型种类可以参考 https://developer.github.com/v4/union/pullrequesttimelineitems/"""
            relation.typename = item.get(StringKeyUtils.STR_KEY_TYPE_NAME_JSON, None)
            relation.timeline_item_node = item.get(StringKeyUtils.STR_KEY_ID, None)
            relation.origin = json.dumps(item)

            """按照感兴趣的类型 依次做出解析"""
            # 注：可能会有疏漏的代表commit的场景没有考虑
            if relation.typename == StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT:
                """force push"""
                afterCommit = item.get(StringKeyUtils.STR_KEY_AFTER_COMMIT)
                if afterCommit is not None:
                    relation.headRefForcePushedEventAfterCommit = afterCommit.get(StringKeyUtils.STR_KEY_OID, None)
                beforeCommit = item.get(StringKeyUtils.STR_KEY_BEFORE_COMMIT)
                if beforeCommit is not None:
                    relation.headRefForcePushedEventBeforeCommit = beforeCommit.get(StringKeyUtils.STR_KEY_OID, None)
                return relation
            elif relation.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT:
                """commit"""
                commit = item.get(StringKeyUtils.STR_KEY_COMMIT)
                if commit is not None and isinstance(commit, dict):
                    relation.pull_request_commit = commit.get(StringKeyUtils.STR_KEY_OID, None)
                return relation
            elif relation.typename == StringKeyUtils.STR_KEY_MERGED_EVENT:
                """merge"""
                commit = item.get(StringKeyUtils.STR_KEY_COMMIT)
                if commit is not None and isinstance(commit, dict):
                    relation.merge_commit = commit.get(StringKeyUtils.STR_KEY_OID, None)
                return relation
            elif relation.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW:
                """review 需要获取comments, commit和original_commit"""
                comments = item.get(StringKeyUtils.STR_KEY_COMMENTS).get(StringKeyUtils.STR_KEY_NODES)
                relation.comments = comments
                commit = item.get(StringKeyUtils.STR_KEY_COMMIT)
                if commit is not None and isinstance(commit, dict):
                    relation.pull_request_review_commit = commit.get(StringKeyUtils.STR_KEY_OID, None)
                author = item.get(StringKeyUtils.STR_KEY_AUTHOR)
                if author is not None and isinstance(author, dict):
                    relation.user_login = author.get(StringKeyUtils.STR_KEY_LOGIN)
                return relation
            elif relation.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_THREAD:
                comments = item.get(StringKeyUtils.STR_KEY_COMMENTS).get(StringKeyUtils.STR_KEY_NODES)
                relation.comments = comments
                if comments is not None and len(comments) > 0 and isinstance(comments, list):
                    original_commit = comments[0].get(StringKeyUtils.STR_KEY_ORIGINAL_COMMIT)
                    relation.user_login = comments[0].get(StringKeyUtils.STR_KEY_AUTHOR).get(
                        StringKeyUtils.STR_KEY_LOGIN)
                    if original_commit is not None and isinstance(original_commit, dict):
                        relation.pull_request_review_commit = original_commit.get(StringKeyUtils.STR_KEY_OID, None)
                return relation
            elif relation.typename == StringKeyUtils.STR_KEY_ISSUE_COMMENT:
                """issueComment（也算做review的一种）"""
                relation.user_login = item.get(StringKeyUtils.STR_KEY_AUTHOR).get(StringKeyUtils.STR_KEY_LOGIN)
                return relation
            else:
                return None
