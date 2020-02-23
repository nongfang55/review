# coding=gbk


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
