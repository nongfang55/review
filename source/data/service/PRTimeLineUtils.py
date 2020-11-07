# coding=gbk
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.utils.StringKeyUtils import StringKeyUtils


class PRTimeLineUtils:
    """针对pull request的timeline做一些处理的工具类"""

    @staticmethod
    def splitTimeLine(prTimeLineItems):
        """把一条完整的时间线分割  返回为一系列的review和相关的commit等event"""
        """注：现在时间线是倒序的 2020.8.5"""
        """注：新增对reopen的额外处理逻辑
            一次closed和reopend 中间的item都放在一个pair中，change就是reopen事件，
            对这pair中的comment认为是统统无效的  若干段的closed-reopen把时间线可能
            分成若干段  reopend的pr占比约为1.5% 还是稍微处理一下吧，问题不大  2020.10.31
        """

        """先遍历prTimeLineItems看是否需要应为reopen事件而分割， 分割剩下的相当于单独几个小pr"""
        tempPrTimeLineItems = prTimeLineItems.copy()
        prTimeLineItemsLists = []  # 可能会分割成若干次单独的pr流程
        reviewPair = []  # review -> [{(changeNode, changeNode)): reviewNodes}, {}, ...]

        pair_review_node_list = []
        pair_change_node_list = []

        prTimeLineItems = []
        isInClosedGap = False  # 用于判断是否在Closed和Reopend的间隙，如果是，则为True
        for item in tempPrTimeLineItems:
            if isInClosedGap:
                if item.typename == PRTimeLineUtils.getClosedType():  # 结束gap状态
                    isInClosedGap = False
                    if pair_change_node_list.__len__() > 0 or pair_review_node_list.__len__() > 0:
                        reviewPair.append((pair_change_node_list, pair_review_node_list))
                    pair_change_node_list = []
                    pair_review_node_list = []
                    """放入itemList中，不过滤"""
                    prTimeLineItems.append(item)
                else:
                    """gap 状态应该只有谈话，即没有change, 无脑放入review即可"""
                    """fix bug 这里需要过滤change的type
                       具体例子 https://github.com/opencv/opencv/pull/12623
                    """
                    if item.typename in PRTimeLineUtils.getReviewType():
                        pair_review_node_list.append(item)
                        """过滤"""
            else:
                if item.typename == PRTimeLineUtils.getReopenedType():  # 进入gap状态
                    isInClosedGap = True
                    """先把之前的非gap的活动列为一个单独的pr"""
                    if prTimeLineItems.__len__() > 0:
                        prTimeLineItemsLists.append(prTimeLineItems)
                    prTimeLineItems = []
                    pair_change_node_list.append(item)
                else:
                    """正常状态"""
                    prTimeLineItems.append(item)

        """收尾工作"""
        if prTimeLineItems.__len__() > 0:
            prTimeLineItemsLists.append(prTimeLineItems)

        """由于Reopend的分割，可能需要分成几个部分"""
        for prTimeLineItems in prTimeLineItemsLists:
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
                if item.typename in PRTimeLineUtils.getChangeType() or\
                        item.typename in PRTimeLineUtils.getReviewType():
                    last_item = item

            # 注：对于change_node_list为空的pair也保留，否则会漏掉无效评论
            if pair_change_node_list.__len__() > 0 or pair_review_node_list.__len__() > 0:
                reviewPair.append((pair_change_node_list, pair_review_node_list))

        return reviewPair

    @staticmethod
    def getChangeType():
        """注： reopened不加进去，因为不算是代码变更"""
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_COMMIT, StringKeyUtils.STR_KEY_HEAD_REF_PUSHED_EVENT,
                StringKeyUtils.STR_KEY_MERGED_EVENT, StringKeyUtils.STR_KEY_CLOSED_EVENT]

    @staticmethod
    def getReviewType():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW, StringKeyUtils.STR_KEY_PULL_REQUEST_REVIEW_THREAD]

    @staticmethod
    def getClosedType():
        return StringKeyUtils.STR_KEY_CLOSED_EVENT
    @staticmethod

    def getReopenedType():
        return StringKeyUtils.STR_KEY_REOPENED_EVENT
