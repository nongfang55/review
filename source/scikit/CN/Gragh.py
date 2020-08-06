from source.scikit.CN.Node import Node


class Graph:
    def __init__(self):
        self.node_list = {}
        self.num_nodes = 0
        self.max_weight = 0

    def add_node(self, key):
        self.num_nodes = self.num_nodes + 1
        new_node = Node(key)
        self.node_list[key] = new_node
        return new_node

    def get_node(self, n):
        if n in self.node_list:
            return self.node_list[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.node_list

    def add_edge(self, f, t, cost=0):
        if f not in self.node_list:
            nf = self.add_node(f)
        if t not in self.node_list:
            nt = self.add_node(t)
        self.node_list[t].in_cnt += 1
        self.node_list[f].add_neighbor(self.node_list[t], cost)
        self.max_weight = max(self.max_weight, cost)

    def get_nodes(self):
        return self.node_list.keys()

    def __iter__(self):
        return iter(self.node_list.values())
