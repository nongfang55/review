# coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils


class DataFrameColumnUtils:
    """作为提供sql查询数据后面的column提供的存储类"""

    COLUMN_REVIEW_FPS_REVIEW = [StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_AUTHOR_LOGIN, StringKeyUtils.STR_KEY_COMMITTER_LOGIN,
                                StringKeyUtils.STR_KEY_COMMIT_AUTHOR_DATE, StringKeyUtils.STR_KEY_COMMIT_COMMITTER_DATE,
                                StringKeyUtils.STR_KEY_COMMIT_MESSAGE, StringKeyUtils.STR_KEY_COMMIT_COMMENT_COUNT,
                                StringKeyUtils.STR_KEY_STATUS_TOTAL, StringKeyUtils.STR_KEY_STATUS_ADDITIONS,
                                StringKeyUtils.STR_KEY_STATUS_DELETIONS, StringKeyUtils.STR_KEY_REPO_FULL_NAME,
                                StringKeyUtils.STR_KEY_PULL_NUMBER, StringKeyUtils.STR_KEY_ID,
                                StringKeyUtils.STR_KEY_USER_LOGIN, StringKeyUtils.STR_KEY_BODY,
                                StringKeyUtils.STR_KEY_STATE, StringKeyUtils.STR_KEY_AUTHOR_ASSOCIATION,
                                StringKeyUtils.STR_KEY_SUBMITTED_AT, StringKeyUtils.STR_KEY_COMMIT_ID,
                                StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_COMMIT_SHA,
                                StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_FILENAME,
                                StringKeyUtils.STR_KEY_STATUS, StringKeyUtils.STR_KEY_ADDITIONS,
                                StringKeyUtils.STR_KEY_DELETIONS, StringKeyUtils.STR_KEY_CHANGES,
                                StringKeyUtils.STR_KEY_PATCH]

    COLUMN_REVIEW_FPS_COMMIT = [StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_NODE_ID,
                                StringKeyUtils.STR_KEY_AUTHOR_LOGIN, StringKeyUtils.STR_KEY_COMMITTER_LOGIN,
                                StringKeyUtils.STR_KEY_COMMIT_AUTHOR_DATE, StringKeyUtils.STR_KEY_COMMIT_COMMITTER_DATE,
                                StringKeyUtils.STR_KEY_COMMIT_MESSAGE, StringKeyUtils.STR_KEY_COMMIT_COMMENT_COUNT,
                                StringKeyUtils.STR_KEY_STATUS_TOTAL, StringKeyUtils.STR_KEY_STATUS_ADDITIONS,
                                StringKeyUtils.STR_KEY_STATUS_DELETIONS, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                                StringKeyUtils.STR_KEY_NOP]

    COLUMN_REVIEW_FPS_FILE = [StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_NOP,
                              StringKeyUtils.STR_KEY_NOP, StringKeyUtils.STR_KEY_COMMIT_SHA,
                              StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_FILENAME,
                              StringKeyUtils.STR_KEY_STATUS, StringKeyUtils.STR_KEY_ADDITIONS,
                              StringKeyUtils.STR_KEY_DELETIONS, StringKeyUtils.STR_KEY_CHANGES,
                              StringKeyUtils.STR_KEY_PATCH]
