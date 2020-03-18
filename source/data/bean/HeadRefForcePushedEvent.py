# coding=gbk
from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class HeadRefForcePushedEvent(BeanBase):
    """github中head ref force pushed event事件"""

    def __init__(self):
        self.node_id = None
        self.afterCommit = None
        self.beforeCommit = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_NODE_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_AFTER_COMMIT,
                 StringKeyUtils.STR_KEY_BEFORE_COMMIT]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_AFTER_COMMIT, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_BEFORE_COMMIT, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_AFTER_COMMIT: self.afterCommit,
                 StringKeyUtils.STR_KEY_BEFORE_COMMIT: self.beforeCommit}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            # resList = []  # 返回结果为一系列关系
            # if isinstance(src, dict):
            #     data = src.get('data', None)
            #     if data is not None and isinstance(data, dict):
            #         nodes = data.get('nodes', None)
            #         if nodes is not None:
            #             for pr in nodes:
            #                 pr_id = pr.get('id')
            #                 pos = 0
            #                 timelineitems = pr.get('timelineItems', None)
            #                 if timelineitems is not None:
            #                     edges = timelineitems.get('edges', None)
            #                     if edges is not None:
            #                         for item in edges:
            #                             item_node = item.get('node', None)
            #                             if item_node is not None:
            #                                 typename = item_node.get('__typename', None)
            #                                 item_id = item_node.get('id', None)
            #                                 relation = PRTimeLineRelation()
            #                                 relation.position = pos
            #                                 pos += 1
            #                                 relation.typename = typename
            #                                 relation.timelineitem_node = item_id
            #                                 relation.pullrequest_node = pr_id
            #                                 resList.append(relation)
            return resList
