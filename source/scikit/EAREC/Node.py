class Node:

    STR_NODE_TYPE_PR = 'pr'
    STR_NODE_TYPE_REVIEWER = 'reviewer'

    def __init__(self, key, nodeType, contentKey, description):
        self.id = key
        self.connectedTo = []
        self.type = nodeType
        self.contentKey = contentKey
        self.description = description  # 对该点的描述

    def add_edge(self, edge_id):
        """从这个顶点添加一条边"""
        if edge_id not in self.connectedTo:
            """度的变化选择外部自己计算"""
            self.connectedTo.append(edge_id)