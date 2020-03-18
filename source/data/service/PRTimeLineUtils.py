# coding=gbk
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.utils.StringKeyUtils import StringKeyUtils


class PRTimeLineUtils:
    """针对pull request的timeline做一些处理的工具类"""

    @staticmethod
    def splitTimeLine(timeLineItemRelations):
        """把一条完整的时间线分割  返回为一系列的review和相关的commit等event"""

        reviewPair = []  # review -> [nodes, nodes]

        pair_review_node = None
        pair_change_nodes = None

        pos = 0
        while pos < timeLineItemRelations.__len__():
            temp = timeLineItemRelations[pos]
            if isinstance(temp, PRTimeLineRelation):
                if temp.typename == StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW:
                    pair_review_node = temp
                    pair_change_nodes = []
                    pos += 1
                else:
                    if temp.typename in PRTimeLineUtils.getChangeType() and pair_review_node is not None:
                        pair_change_nodes.append(temp)
                        pos += 1
                    elif temp.typename not in PRTimeLineUtils.getChangeType() and pair_review_node is not None:
                        if pair_change_nodes.__len__() > 0:
                            reviewPair.append((pair_review_node, pair_change_nodes))
                        pair_review_node = None
                        pair_change_nodes = None
                        pos += 1
                    else:
                        pos += 1
        if pair_change_nodes is not None and pair_change_nodes.__len__() > 0:
            reviewPair.append((pair_review_node, pair_change_nodes))
        return reviewPair

    @staticmethod
    def getChangeType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT, StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT]
