# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.data.bean.HeadRefForcePushedEvent import HeadRefForcePushedEvent
from source.data.bean.PullRequestCommit import PullRequestCommit
from source.utils.StringKeyUtils import StringKeyUtils


class PRTimeLineRelation(BeanBase):
    """github中pull request的timeline 关系"""

    def __init__(self):
        self.pullrequest_node = None
        self.timelineitem_node = None
        self.typename = None
        self.position = None

        """可选属性 做简化使用的 实际不进入存储"""
        self.headRefForcePushedEventAfterCommit = None
        self.headRefForcePushedEventBeforeCommit = None
        self.pullrequestReviewCommit = None
        self.pullrequestCommitCommit = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE
            , StringKeyUtils.STR_KEY_TYPE_NAME, StringKeyUtils.STR_KEY_POSITION]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_PULL_REQUEST_NODE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_TYPE_NAME, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_POSITION, BeanBase.DATA_TYPE_INT)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_PULL_REQUEST_NODE: self.pullrequest_node,
                 StringKeyUtils.STR_KEY_TIME_LINE_ITEM_NODE: self.timelineitem_node,
                 StringKeyUtils.STR_KEY_TYPE_NAME: self.typename,
                 StringKeyUtils.STR_KEY_POSITION: self.position}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            resList = []  # 返回结果为一系列关系
            resItems = []  # 从时间线中提取有意义的信息
            if isinstance(src, dict):
                data = src.get(StringKeyUtils.STR_KEY_DATA, None)
                if data is not None and isinstance(data, dict):
                    nodes = data.get(StringKeyUtils.STR_KEY_NODES, None)
                    if nodes is not None:
                        for pr in nodes:
                            pr_id = pr.get(StringKeyUtils.STR_KEY_ID)
                            pos = 0
                            timelineitems = pr.get(StringKeyUtils.STR_KEY_TIME_LINE_ITEMS, None)
                            if timelineitems is not None:
                                edges = timelineitems.get(StringKeyUtils.STR_KEY_EDGES, None)
                                if edges is not None:
                                    for item in edges:  # 对各个item做出解析
                                        item_node = item.get(StringKeyUtils.STR_KEY_NODE, None)
                                        if item_node is not None:
                                            typename = item_node.get(StringKeyUtils.STR_KEY_TYPE_NAME_JSON, None)
                                            item_id = item_node.get(StringKeyUtils.STR_KEY_ID, None)
                                            relation = PRTimeLineRelation()
                                            relation.position = pos
                                            pos += 1
                                            relation.typename = typename
                                            relation.timelineitem_node = item_id
                                            relation.pullrequest_node = pr_id
                                            resList.append(relation)

                                            """做出解析"""
                                            if typename == StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT:
                                                bean = HeadRefForcePushedEvent()
                                                bean.node_id = item_id
                                                bean.afterCommit = item_node.get(StringKeyUtils.STR_KEY_AFTER_COMMIT
                                                                                 , None).get(StringKeyUtils.STR_KEY_OID
                                                                                             , None)
                                                relation.headRefForcePushedEventAfterCommit = bean.afterCommit
                                                bean.beforeCommit = item_node.get(StringKeyUtils.STR_KEY_BEFORE_COMMIT,
                                                                                  None).get(StringKeyUtils.STR_KEY_OID,
                                                                                            None)
                                                relation.headRefForcePushedEventBeforeCommit = bean.beforeCommit
                                                resItems.append(bean)
                                            elif typename == StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT:
                                                bean = PullRequestCommit()
                                                bean.node_id = item_id
                                                commit = item_node.get(StringKeyUtils.STR_KEY_COMMIT)
                                                if commit is not None and isinstance(commit, dict):
                                                    bean.oid = commit.get(StringKeyUtils.STR_KEY_OID, None)
                                                    relation.pullrequestCommitCommit = commit.get(StringKeyUtils.STR_KEY_OID, None)
                                                resItems.append(bean)
                                            elif typename == StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW:
                                                commit = item_node.get(StringKeyUtils.STR_KEY_COMMIT)
                                                if commit is not None and isinstance(commit, dict):
                                                    relation.pullrequestReviewCommit = \
                                                        commit.get(StringKeyUtils.STR_KEY_OID, None)

            return resList, resItems
