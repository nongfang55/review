# coding=gbk
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.utils.StringKeyUtils import StringKeyUtils


class PRTimeLineUtils:
    """针对pull request的timeline做一些处理的工具类"""

    @staticmethod
    def splitTimeLine(prTimeLine):
        """把一条完整的时间线分割  返回为一系列的review和相关的commit等event"""

        reviewPair = []  # review -> [{reviewNode: changeNodes}, {}, ...]

        pair_review_node = None
        pair_change_nodes = None

        timeLineItemRelations = prTimeLine.timeline_items
        for item in timeLineItemRelations:
            if item.typename in PRTimeLineUtils.getReviewType() and item.user_login != prTimeLine.user_login:
                if pair_review_node is not None and pair_change_nodes.__len__() > 0:
                    reviewPair.append((pair_review_node, pair_change_nodes))
                pair_review_node = item
                pair_change_nodes = []
            elif item.typename in PRTimeLineUtils.getChangeType() and pair_review_node is not None:
                pair_change_nodes.append(item)
            elif item.typename not in PRTimeLineUtils.getChangeType() and pair_review_node is not None:
                if pair_change_nodes.__len__() > 0:
                    reviewPair.append((pair_review_node, pair_change_nodes))
                pair_review_node = None
                pair_change_nodes = None

        if pair_change_nodes is not None and pair_change_nodes.__len__() > 0:
            reviewPair.append((pair_review_node, pair_change_nodes))
        return reviewPair

    @staticmethod
    def getChangeType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT, StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT,
                StringKeyUtils.STR_KEY_MERGED_EVENT]

    @staticmethod
    def getReviewType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW, StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_THREAD,
                StringKeyUtils.STR_KEY_ISSUE_COMMENT]
