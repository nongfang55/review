from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation


class PRTimeLine(BeanBase):
    """github中pull request的timeline 关系"""

    def __init__(self):
        self.timeline_items = []
        self.pull_request_id = None
        self.user_login = None

    def toTSVFormat(self):
        res = []
        for item in self.timeline_items:
            res.append({
                "pullrequest_node": item.pull_request_node,
                "timelineitem_node": item.timeline_item_node,
                "typename": item.typename,
                "position": item.position,
                "origin": item.origin
            })
        return res


    class Parser(BeanBase.parser):

        @staticmethod
        def parser(node):
            if node is None:
                return None

            """初始化PRTimeLine"""
            pr_timeline = PRTimeLine()
            pr_timeline.pull_request_id = node.get(StringKeyUtils.STR_KEY_ID)
            author = node.get(StringKeyUtils.STR_KEY_AUTHOR)
            if author is not None and isinstance(author, dict):
                pr_timeline.user_login = author.get(StringKeyUtils.STR_KEY_LOGIN)
            # pr_timeline.author = node.get(StringKeyUtils.STR_KEY_ID)

            """获取TimeLineItems"""
            timeline_items = node.get(StringKeyUtils.STR_KEY_TIME_LINE_ITEMS, None)
            if timeline_items is None:
                return pr_timeline
            edges = timeline_items.get(StringKeyUtils.STR_KEY_EDGES, None)
            if edges is None:
                return pr_timeline

            """解析TimeLineItems"""
            for pos, edge in enumerate(edges):
                item = edge.get(StringKeyUtils.STR_KEY_NODE, None)

                if item is None:
                    continue
                relation = PRTimeLineRelation.Parser.parser(item)
                if relation is None:
                    continue

                relation.position = pos
                relation.pull_request_node = pr_timeline.pull_request_id
                pr_timeline.timeline_items.append(relation)
            return pr_timeline