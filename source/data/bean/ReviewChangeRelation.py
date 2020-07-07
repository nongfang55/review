from source.data.bean.Beanbase import BeanBase
from source.data.bean.PRTimeLineRelation import PRTimeLineRelation
from source.utils.StringKeyUtils import StringKeyUtils


class ReviewChangeRelation(BeanBase):
    """github中changeTrigger 中review 和 change 的关系
       关系用 node_id 标记
    """

    def __init__(self):
        self.pull_request_node_id = None
        self.review_node_id = None
        self.change_node_id = None
        self.review_position = None
        self.change_position = None
        """position 用于记录 review 和change的相对位置, 
        两个position 就直接记录 item 在 原来 pr中的pos
        保持数据的一致性
        
        注：@zhangyifan.anthony
        对于review 后面没有 change的场景
        change_node_id 设置为 -1 也存入数据库
        change_position 同样设置为 -1
        """

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE_ID,
                StringKeyUtils.STR_KEY_REVIEW_NODE_ID, StringKeyUtils.STR_KEY_CHANGE_NODE_ID]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_PULL_REQUEST_NODE_ID, StringKeyUtils.STR_KEY_REVIEW_NODE_ID,
                 StringKeyUtils.STR_KEY_CHANGE_NODE_ID, StringKeyUtils.STR_KEY_REVIEW_POSITION,
                 StringKeyUtils.STR_KEY_CHANGE_POSITION]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_PULL_REQUEST_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_REVIEW_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_CHANGE_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_REVIEW_POSITION, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_CHANGE_POSITION, BeanBase.DATA_TYPE_INT)]
        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_PULL_REQUEST_NODE_ID: self.pull_request_node_id,
                 StringKeyUtils.STR_KEY_REVIEW_NODE_ID: self.review_node_id,
                 StringKeyUtils.STR_KEY_CHANGE_NODE_ID: self.change_node_id,
                 StringKeyUtils.STR_KEY_REVIEW_POSITION: self.review_position,
                 StringKeyUtils.STR_KEY_CHANGE_POSITION: self.change_position}

        return items

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res_list = []
            """解析类型  (PRTimeLineRelation -> [PRTimeLineRelation, ....])
               返回 [ReviewChangeRelation, ....]
            """
            if isinstance(src, tuple):
                reviewItem = src[0]
                changeItemList = src[1]
                if isinstance(reviewItem, PRTimeLineRelation) and isinstance(changeItemList, list):
                    pull_request_node_id = reviewItem.pull_request_node
                    review_node_id = reviewItem.timeline_item_node
                    review_position = reviewItem.position
                    for changeItem in changeItemList:
                        if isinstance(changeItem, PRTimeLineRelation):
                            change_node_id = changeItem.timeline_item_node
                            change_position = changeItem.position

                            relation = ReviewChangeRelation()
                            relation.pull_request_node_id = pull_request_node_id
                            relation.review_node_id = review_node_id
                            relation.review_position = review_position
                            relation.change_node_id = change_node_id
                            relation.change_position = change_position
                            res_list.append(relation)

                    if changeItemList.__len__() == 0:
                        relation = ReviewChangeRelation()
                        relation.pull_request_node_id = pull_request_node_id
                        relation.review_node_id = review_node_id
                        relation.review_position = review_position

                        """对于review后面没有change的场景change_node_id
                        设置为 -1 也存入数据库change_position  同样设置为 -1 """
                        relation.change_node_id = -1
                        relation.change_position = -1
                        res_list.append(relation)

            return res_list
