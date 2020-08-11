class Node:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.rankedEdges = []
        self.marked = []
        self.in_cnt = 0
        self.author_cnt = 0
        self.reviewer_cnt = 0

    def add_neighbor(self, nbr, weight=0):
        """从这个顶点添加一个连接到另一个"""
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + 'connectedTo' + str([x.id for x in self.connectedTo])

    def get_connections(self):
        """返回邻接表中的所有的项点"""
        return  self.connectedTo.keys()

    def get_id(self):
        return self.id

    def get_weight(self, nbr):
        """返回某条边的权重"""
        return self.connectedTo[nbr]

    def best_neighbor(self):
        """返回按权重排序后的neighbor"""
        for edge in self.rankedEdges:
            if self.marked.__contains__(edge[0]):
                continue
            return edge[0]

    def rank_edges(self):
        neighbors = sorted(self.connectedTo.items(), key=lambda x: x[1], reverse=True)
        self.rankedEdges = neighbors

    def mark_edge(self, node):
        self.marked.append(node)

    def get_neighbors(self):
        return list(map(lambda x: x.id, self.connectedTo))
