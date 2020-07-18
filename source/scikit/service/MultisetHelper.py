

class WordMultiset:
    """用于辅助计算 字符串多种集合的工具类"""

    def __init__(self):
        self.data = {}

    def copy(self):
        t = WordMultiset()
        t.data = self.data.copy()
        return t

    def equalization(self):
        sum = 0
        for v in self.data.values():
            sum += v
        for k, v in self.data.items():
            self.data[k] = v / sum

    def multiply(self, set2):
        sum = 0
        for k, v in set2.data.items():
            if self.data.get(k, None) is not None:
                sum += v * self.data.get(k)
        return sum

    def divide(self, set2):
        for k, v in set2.data.items():
            if self.data.get(k, None) is not None:
                self.data[k] /= v

    def addByTuple(self, tuple_list):
        if isinstance(tuple_list, list):
            for k, v in tuple_list:
                if self.data.get(k, None) is None:
                    self.data[k] = v
                else:
                    self.data[k] += v

    def add(self, word_list):
        if isinstance(word_list, list):
            for word in word_list:
                if self.data.get(word, None) is not None:
                    self.data[word] = self.data[word] + 1
                else:
                    self.data[word] = 1
        elif isinstance(word_list, WordMultiset):
            for k, v in word_list.data.items():
                if self.data.get(k, None) is None:
                    self.data[k] = v
                else:
                    self.data[k] = self.data[k] + v


    def TverskyIndex(self, set2, a, b):
        """计算和其他set 的Tversky 系数
           需要 a + b = 1
        """

        if a + b != 1:
            raise Exception("a + b is not 1 for Tversky index")

        """若都是空集  返回1"""
        if self.data.items().__len__() == 0 and set2.data.items().__len__() == 0:
            return 1

        """计算并集"""
        set_u = self.data.copy()
        for k, v in set2.data.items():
            if set_u.get(k, None) is not None:
                set_u[k] = set_u[k] + v
            else:
                set_u[k] = v

        """计算交集"""
        set_n = {}
        for k, v in self.data.items():
            if set2.data.get(k, None) is not None:
                set_n[k] = min(v, set2.data[k])

        """计算两个集合的大小"""
        set_u_s = 0
        for k, v in set_u.items():
            set_u_s += v
        set_n_s = 0
        for k, v in set_n.items():
            set_n_s += v

        """计算 set1 对 set 2 的差集"""
        set_e_1_to_2 = {}
        for k, v in set2.data.items():
            if self.data.get(k, None) is None:
                set_e_1_to_2[k] = v
            else:
                if v > self.data.get(k):
                    set_e_1_to_2[k] = v - self.data.get(k)

        """计算 set2 对  set1 的差集"""
        set_e_2_to_1 = {}
        for k, v in self.data.items():
            if set2.data.get(k, None) is None:
                set_e_2_to_1[k] = v
            else:
                if v > set2.data.get(k):
                    set_e_2_to_1[k] = v - set2.data.get(k)

        """计算大小"""
        set_e_1_to_2_s = 0
        for k, v in set_e_1_to_2.items():
            set_e_1_to_2_s += v
        set_e_2_to_1_s = 0
        for k, v in set_e_2_to_1.items():
            set_e_2_to_1_s += v

        return set_n_s / (set_u_s - a * set_e_2_to_1_s - b * set_e_1_to_2_s)


    def jaccardCofficient(self, set2):
        """计算和其他set的 Jaccard 系数"""

        """若都是空集  返回1"""
        if self.data.items().__len__() == 0 and set2.data.items().__len__() == 0:
            return 1

        """计算并集"""
        set_u = self.data.copy()
        for k, v in set2.data.items():
            if set_u.get(k, None) is not None:
                set_u[k] = set_u[k] + v
            else:
                set_u[k] = v

        """计算交集"""
        set_n = {}
        for k, v in self.data.items():
            if set2.data.get(k, None) is not None:
                set_n[k] = min(v, set2.data[k])

        """计算两个集合的大小"""
        set_u_s = 0
        for k, v in set_u.items():
            set_u_s += v
        set_n_s = 0
        for k, v in set_n.items():
            set_n_s += v

        return set_n_s / set_u_s



if __name__ == '__main__':

     set1 = WordMultiset()
     set2 = WordMultiset()

     set1.addByTuple([(1, 0.1), (2, 0.2)])
     set2.addByTuple([(2, 0.3), (3, 0.4)])
     # set1.add(['a', 'a', 'b'])
     # set2.add(['b', 'c', 'c'])
     print(set1.data)
     print(set2.data)
     set2.add(set1)
     print(set1.data)
     print(set2.data)
     # print(set1.multiply(set2))
     # print(set2.multiply(set1))
     set1.divide(set2)
     print(set1.data)
     set1.equalization()
     print(set1.data)
     # print(set1.jaccardCofficient(set2))
     # print(set1.TverskyIndex(set2, 1, 0))