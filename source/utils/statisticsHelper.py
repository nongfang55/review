# coding=gbk
import threading


class statisticsHelper:
    """github 数据的统计帮助类"""

    def __init__(self):
        self.usefulRequestNumber = 0  # 有用的pull request的提取数量
        self.commentNumber = 0
        self.usefulReviewNumber = 0  # review的提取数量
        self.usefulReviewCommentNumber = 0  # review comment的提取数量
        self.usefulIssueCommentNumber = 0  # issue comment 的提取数量
        self.usefulCommitNumber = 0  # commit的提取数量
        self.usefulCommitCommentNumber = 0  # commit comment的提取数量
        self.startTime = None  # 开始时间
        self.endTime = None  # 结束时间
        self.usefulTimeLineCount = 0  # pull request的timeline线数量
        self.usefulChangeTrigger = 0  # 有用的comment的change trigger数量
        self.twoParentsNodeCase = 0  # 舍弃的双点的case数量
        self.outOfLoopCase = 0  # 迭代超出预算的case数量
        self.lock = threading.RLock()
