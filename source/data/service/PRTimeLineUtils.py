# coding=gbk
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.utils.StringKeyUtils import StringKeyUtils


class PRTimeLineUtils:
    """针对pull request的timeline做一些处理的工具类"""

    @staticmethod
    def splitTimeLine(prTimeLineItems):
        """把一条完整的时间线分割  返回为一系列的review和相关的commit等event"""
        """注：现在时间线是倒序的 2020.8.5"""

        reviewPair = []  # review -> [{(changeNode, changeNode)): reviewNodes}, {}, ...]

        pair_review_node_list = []
        pair_change_node_list = []
        last_item = None
        for item in prTimeLineItems:
            if item.typename in PRTimeLineUtils.getChangeType() and (last_item is not None and last_item.typename in PRTimeLineUtils.getReviewType()):
                """如果遇到了change类型，且上一条是comment，创建新的pair"""
                # push pair
                # 注：对于change_node_list为空的pair也保留，否则会漏掉无效评论
                if pair_change_node_list.__len__() > 0 or pair_review_node_list.__len__() > 0:
                    reviewPair.append((pair_change_node_list, pair_review_node_list))
                # 创建新pair
                pair_review_node_list = []
                pair_change_node_list = [item]
            elif item.typename in PRTimeLineUtils.getChangeType() and (last_item is None or (last_item is not None and last_item.typename in PRTimeLineUtils.getChangeType())):
                """如果遇到了change类型，且上一条是change，放入change_node_list"""
                pair_change_node_list.append(item)
            elif item.typename in PRTimeLineUtils.getReviewType() and pair_change_node_list.__len__() > 0:
                """如果遇到了comment类型，且change_list不为空，放入review_node_list"""
                pair_review_node_list.append(item)
            elif item.typename in PRTimeLineUtils.getReviewType() and pair_change_node_list.__len__() == 0:
                """如果遇到了comment类型，且change_list为空，仍然放入review_node_list"""
                pair_review_node_list.append(item)
            last_item = item

        # 注：对于change_node_list为空的pair也保留，否则会漏掉无效评论
        if pair_change_node_list.__len__() > 0 or pair_review_node_list.__len__() > 0:
            reviewPair.append((pair_change_node_list, pair_review_node_list))

        return reviewPair

    @staticmethod
    def getChangeType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT, StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT,
                StringKeyUtils.STR_KEY_MERGED_EVENT]

    @staticmethod
    def getReviewType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW, StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_THREAD,
                StringKeyUtils.STR_KEY_ISSUE_COMMENT]
