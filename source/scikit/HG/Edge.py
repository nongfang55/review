class Edge:
    """超图的边，边可以包含多个顶点"""

    STR_EDGE_TYPE_PR_DIS = 'edge_pr_dis'
    STR_EDGE_TYPE_PR_REVIEW_RELATION = 'pr_review_relation'
    STR_EDGE_TYPE_PR_AUTHOR_RELATION = 'pr_author_relation'
    STR_EDGE_TYPE_AUTHOR_REVIEWER_RELATION = 'author_reviewer_relation'

    def __init__(self, key, edgeType, description, weight=0):
        self.id = key  # 每条边的唯一标识, 即图中的边的序号
        self.connectedTo = []  # 每条边所包含的顶点, 记录node的编号
        self.weight = weight  # 边的权重
        self.type = edgeType  # 边的类型
        self.description = description  # 对于边的描述

    def add_nodes(self, nodes):
        """边增加顶点id"""
        for node_id in nodes:
            if node_id not in self.connectedTo:
                self.connectedTo.append(node_id)

    def __str__(self):
        return "node id:" + str(self.id) + " type:" + self.type + "  description:" + self.description
