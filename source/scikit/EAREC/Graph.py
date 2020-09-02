from source.scikit.EAREC.Edge import Edge
from source.scikit.EAREC.Node import Node
import numpy as np

class Graph:
    """图 包含多种类型的顶点以及多种类型的边"""

    def __init__(self):
        self.node_list = {}  # 以顶点编号为key的字段，序号从0开始
        self.node_content_list = {}  # 以点的内容和种类作为标识的字典，减少检索时间
        self.num_nodes = 0
        self.node_id_counter = 0  # 考虑到后续可能会增加删除顶点和边的功能，顶点的id由计数器增肌
        self.node_id_map = {}  # 由于顶点的删除可能会导致序号的空缺，增加上矩阵的序号和顶点序号的映射

        self.edge_list = {}  # 以边编号从为key的字段，序号从0开始
        self.edge_content_list = {}  # 以边的内容和种类做标识的字典， 减少检索时间
        self.num_edges = 0
        self.edge_id_counter = 0  # 边同理，每一次增加边，都是获得不同的序号
        self.edge_id_map = {}  # 边同理，增加矩阵上序号和边序号的映射

        self.W = None

    def add_node(self, nodeType, contentKey, description):
        # 在超图中添加顶点，需要提供顶点的总类和描述，key由图累积计算
        node = self.get_node_by_content(nodeType=nodeType, contentKey=contentKey)  # 先依照条件查找顶点
        if node is None:
            new_node = Node(key=self.node_id_counter, contentKey=contentKey, nodeType=nodeType, description=description)
            self.node_list[self.node_id_counter] = new_node
            self.node_id_map[self.num_nodes] = self.node_id_counter
            self.node_content_list[(nodeType, contentKey)] = new_node
            self.num_nodes += 1
            self.node_id_counter += 1
            return new_node
        else:
            return node

    def get_node_by_key(self, n):  # 通过顶点的序号来查找顶点
        return self.node_list.get(n, None)

    def get_node_by_content(self, nodeType, contentKey):  # 通过顶点的内容来查找顶点
        return self.node_content_list.get((nodeType, contentKey), None)

    def add_edge(self, nodes, edgeType, description, weight, nodeObjects=None, queryBeforeAdd=True):
        """由于超图的复杂性  顶点和边分开添加，增加边的过程并不会增加新的顶点
           nodes 代表了 选定顶点在图中的编号列表  [1, 3, 5...]
           nodeObjects 代表 node实体对象的列表
           添加边之前是否查询已存在
        """
        nodes.sort()  # 编号必须从小到大排序

        # 边查询机制选用 节约时间
        if queryBeforeAdd:
            edge = self.get_edge_by_content(edgeType, nodes)
            if edge is not None:
                edge.weight += weight
                return edge

        edge = Edge(key=self.edge_id_counter, edgeType=edgeType, description=description, weight=weight)
        edge.add_nodes(nodes)  # 边添加顶点
        self.edge_list[self.edge_id_counter] = edge
        self.edge_id_map[self.num_edges] = self.edge_id_counter
        self.num_edges += 1
        self.edge_id_counter += 1
        self.edge_content_list[(edgeType, tuple(nodes))] = edge
        # 对边涉及的顶点增加边的连接
        if nodeObjects is not None:
            for node in nodeObjects:
                node.add_edge(edge.id)
        else:
            for node in nodes:
                node = self.get_node_by_key(node)
                node.add_edge(edge.id)
        return edge

    def get_edge_by_key(self, n):  # 通过边的序号来查找边
        return self.edge_list.get(n, None)

    def get_edge_by_content(self, edgeType, nodes):  # 根据边的类型和涉及的点的序号寻找边, 编号从小到大排序
        return self.edge_content_list.get((edgeType, tuple(nodes)), None)

    def get_nodes(self):
        return self.node_list.keys()

    def remove_node_by_key(self, n):
        """通过顶点的编号来删除图中的顶点，如果有包含删除目标的顶点的边，
        那么同样一起删除  如果边只有两个顶点，那么删除改边，否则边去除这个顶点即可"""

        node = self.get_node_by_key(n)
        """先找和这个顶点相关的边"""
        edges = node.connectedTo
        deleteEdgeIdList = []  # 预计需要删除的边的id
        for edge_id in edges:
            edge = self.get_edge_by_key(edge_id)
            if edge.connectedTo.__len__() > 2:
                """对于有三个点以上的边  删除这个顶点，边保留"""
                self.edge_content_list.pop((edge.type, edge.connectedTo))  # 原来的检索边需要删除
                edge.connectedTo.remove(node.id)
                self.edge_content_list[(edge.type, edge.connectedTo)] = edge
            else:
                """找到其他连接边的点， 删除该边在其他顶点上面的引用"""
                for node_id_temp in edge.connectedTo:
                    if node_id_temp != node.id:
                        node_temp = self.get_node_by_key(node_id_temp)
                        node_temp.connectedTo.remove(edge_id)
                """把这个边加入删除名单"""
                deleteEdgeIdList.append(edge_id)

        """删除在删除名单中的边"""
        for edge_id in deleteEdgeIdList:
            edge = self.edge_list.pop(edge_id)
            nodes = list(edge.connectedTo)
            nodes.sort()
            nodes = tuple(nodes)
            self.edge_content_list.pop((edge.type, nodes))

            """删除 edge 在映射表的位子"""
            edgeIdList = []
            for i in range(0, self.num_edges):
                edgeIdList.append(self.edge_id_map[i])
            edgeIdList.remove(edge_id)
            self.edge_id_map.clear()
            for index, edge_res_id in enumerate(edgeIdList):
                self.edge_id_map[index] = edge_res_id
            self.num_edges -= 1

        """删除顶点"""
        self.node_list.pop(n)
        self.node_content_list.pop((node.type, node.contentKey))

        """删除 node 在映射表的位子"""
        nodeIdList = []
        for i in range(0, self.num_nodes):
            nodeIdList.append(self.node_id_map[i])
        nodeIdList.remove(n)  # 删除该点，剩下的点重新编号
        self.node_id_map.clear()
        for index, node_res_id in enumerate(nodeIdList):
            self.node_id_map[index] = node_res_id
        self.num_nodes -= 1
        self.node_id_counter -= 1

    def updateW(self):
        """计算W'矩阵"""
        """对矩阵行规范化
           而不是全体归一化
           疑问，博客上看到矩阵是列归一化？ http://ws.nju.edu.cn/blog/2017/11/markov_chain_pagerank_rwr/
           可能矩阵定义的问题吧 如果矩阵第i行是代表其他点到点i的概率，那么应该是列归一化
        """
        N = self.num_nodes
        W = np.zeros((N, N))
        # 遍历所有边，求权重总和
        total_weight = 0
        for key, edge in self.edge_list.items():
            total_weight += edge.weight
        for key, edge in self.edge_list.items():
            nodes = edge.connectedTo
            W[nodes[0]][nodes[1]] = edge.weight
            W[nodes[1]][nodes[0]] = edge.weight
        # 列归一化
        W /= W.sum(axis=0).reshape(1, N)  # 对第2维度相加
        self.W = W