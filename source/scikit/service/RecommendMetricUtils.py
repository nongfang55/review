# coding=gbk

class RecommendMetricUtils:
    """对于推荐的准确率的工具类"""

    @staticmethod
    def topKAccuracy(recommendCase, answerCase, k):
        """top k Accuracy  输入参数为多个预测和实际的推荐列表的列表"""
        topK = [0 for i in range(k)]
        if recommendCase.__len__() != answerCase.__len__():
            raise Exception("case is not right")
        casePos = 0
        for recommendList in recommendCase:
            # print("recommend:", recommendList)
            answerList = answerCase[casePos]
            # print("answerList:", answerList)
            casePos += 1
            listPos = 0
            firstFind = False
            for recommend in recommendList:
                if firstFind:
                    break
                index = -1
                try:
                    index = answerList.index(recommend)
                except Exception as e:
                    pass
                if index != -1:
                    for i in range(listPos, k):
                        topK[i] += 1
                    firstFind = True
                    break
                else:
                    listPos += 1
        for i in range(0, topK.__len__()):
            topK[i] /= recommendCase.__len__()
        return topK

    @staticmethod
    def MRR(recommendCase, answerCase, k=5):
        """MRR 输入参数为多个预测和实际的推荐列表的列表"""
        MMR = [0 for x in range(0, k)]
        for i in range(0, k):
            totalScore = 0
            if recommendCase.__len__() != answerCase.__len__():
                raise Exception("case is not right")
            casePos = 0
            for recommendList in recommendCase:
                # print("recommend:", recommendList)
                answerList = answerCase[casePos]
                # print("answerList:", answerList)
                recommendList = recommendList[0:i + 1]
                casePos += 1
                listPos = 0
                firstFind = False
                for recommend in recommendList:
                    if firstFind:
                        break
                    index = -1
                    try:
                        index = answerList.index(recommend)
                    except Exception as e:
                        pass
                    if index != -1:
                        totalScore += 1.0 / (listPos + 1)  # 第一个正确人员的排名倒数
                        firstFind = True
                        break
                    else:
                        listPos += 1
                # print("score:", totalScore)
            MMR[i] = totalScore / recommendCase.__len__()
        return MMR
