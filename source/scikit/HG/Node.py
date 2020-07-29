class Node:
    """超图中的点 相比普通图多了一个种类的属性"""

    STR_NODE_TYPE_PR = 'pr'
    STR_NODE_TYPE_REVIEWER = 'reviewer'
    STR_NODE_TYPE_AUTHOR = 'author'

    def __init__(self, key, nodeType, contentKey, description):
        self.id = key  # 在顶点中的唯一序号，即图中的顶点序号
        self.connectedTo = []  # 这里改为超图的Edge的序号列表集合，防止循环引用
        self.in_degree = 0  # 入度
        self.out_degree = 0  # 出度
        self.type = nodeType  # 顶点的种类
        self.contentKey = contentKey  # 原来实体的唯一表示，如人的id、pr的num等， 用于基于内容的顶点查找
        self.description = description  # 对该点的描述

    def add_edge(self, edge_id):
        """从这个顶点添加一条边"""
        if edge_id not in self.connectedTo:
            """度的变化选择外部自己计算"""
            self.connectedTo.append(edge_id)

    def __str__(self):
        return "node id:" + str(self.id) + " type:" + self.type + "  description:" + self.description

    def get_id(self):
        return self.id
