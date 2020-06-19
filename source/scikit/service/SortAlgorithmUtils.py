# coding=gbk

class SortAlgorithmUtils:
    """一些通用的排序相关的算法的工具类"""

    @staticmethod
    def dictScoreConvertToList(scoreDict):
        sortedList = None
        if isinstance(scoreDict, dict):
            tempList = []
            candidates = scoreDict.keys()
            for candidate in candidates:
                tempList.append((candidate, scoreDict[candidate]))
            tempList.sort(key=lambda x: x[1], reverse=True)
            sortedList = []
            for item in tempList:
                sortedList.append(item[0])
        return sortedList

    @staticmethod
    def BordaCountSort(Votes):
        scores = {}
        for voteList in Votes:
            pos = 1
            for vote in voteList:
                if scores.get(vote, None) is None:
                    scores[vote] = 0
                scores[vote] += voteList.__len__() - pos
                pos += 1
        print(scores)
        return SortAlgorithmUtils.dictScoreConvertToList(scores)

    @staticmethod
    def BordaCountSortWithFreq(Votes, freqs):
        """波达计数 加上reviewer的历史review数量作为辅助判断"""
        scores = {}
        for voteList in Votes:
            pos = 1
            for vote in voteList:
                if scores.get(vote, None) is None:
                    scores[vote] = 0
                scores[vote] += voteList.__len__() - pos
                pos += 1
        # print(scores)
        return SortAlgorithmUtils.dictScoreConvertToListWithFreq(scores, freqs)

    @staticmethod
    def dictScoreConvertToListWithFreq(scoreDict, freqs):
        sortedList = None
        if isinstance(scoreDict, dict):
            tempList = []
            candidates = scoreDict.keys()
            for candidate in candidates:
                tempList.append((candidate, scoreDict[candidate]))
            print(tempList)
            tempList.sort(key=lambda x: (x[1], freqs[x[0]]), reverse=True)
            sortedList = []
            for item in tempList:
                sortedList.append(item[0])
        return sortedList
