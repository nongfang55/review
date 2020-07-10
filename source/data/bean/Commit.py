# coding=gbk
from datetime import datetime

from source.data.bean.Beanbase import BeanBase
from source.data.bean.CommentRelation import CommitRelation
from source.utils.StringKeyUtils import StringKeyUtils
from source.data.bean.User import User
from source.data.bean.File import File


class Commit(BeanBase):
    """github commit的数据类  15个属性"""

    def __init__(self):
        self.sha = None
        self.node_id = None
        self.author = None
        self.committer = None
        self.commit_author_date = None
        self.commit_committer_date = None
        self.commit_message = None
        self.commit_comment_count = None
        self.status_total = None
        self.status_additions = None
        self.status_deletions = None
        self.files = None
        self.parents = None

        self.author_login = None
        self.committer_login = None
        """用于判断是否保存对应的gitFile"""
        self.has_file_fetched = None
        """记录commit 指向的Tree"""
        self.tree_oid = None

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_SHA]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_SHA, StringKeyUtils.STR_KEY_NODE_ID, StringKeyUtils.STR_KEY_AUTHOR_LOGIN,
                 StringKeyUtils.STR_KEY_COMMITTER_LOGIN, StringKeyUtils.STR_KEY_COMMIT_AUTHOR_DATE,
                 StringKeyUtils.STR_KEY_COMMIT_COMMITTER_DATE, StringKeyUtils.STR_KEY_COMMIT_MESSAGE,
                 StringKeyUtils.STR_KEY_COMMIT_COMMENT_COUNT, StringKeyUtils.STR_KEY_STATUS_TOTAL,
                 StringKeyUtils.STR_KEY_STATUS_ADDITIONS, StringKeyUtils.STR_KEY_STATUS_DELETIONS,
                 StringKeyUtils.STR_KEY_HAS_FILE_FETCHED, StringKeyUtils.STR_KEY_TREE_OID]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_SHA, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_NODE_ID, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_AUTHOR_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_COMMITTER_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_COMMIT_AUTHOR_DATE, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_COMMIT_COMMITTER_DATE, BeanBase.DATA_TYPE_DATE_TIME),
                 (StringKeyUtils.STR_KEY_COMMIT_MESSAGE, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_COMMIT_COMMENT_COUNT, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_STATUS_TOTAL, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_STATUS_ADDITIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_STATUS_DELETIONS, BeanBase.DATA_TYPE_INT),
                 (StringKeyUtils.STR_KEY_HAS_FILE_FETCHED, BeanBase.DATA_TYPE_BOOLEAN),
                 (StringKeyUtils.STR_KEY_TREE_OID, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_SHA: self.sha, StringKeyUtils.STR_KEY_NODE_ID: self.node_id,
                 StringKeyUtils.STR_KEY_AUTHOR_LOGIN: self.author_login,
                 StringKeyUtils.STR_KEY_COMMITTER_LOGIN: self.committer_login,
                 StringKeyUtils.STR_KEY_COMMIT_AUTHOR_DATE: self.commit_author_date,
                 StringKeyUtils.STR_KEY_COMMIT_COMMITTER_DATE: self.commit_committer_date,
                 StringKeyUtils.STR_KEY_COMMIT_MESSAGE: self.commit_message,
                 StringKeyUtils.STR_KEY_COMMIT_COMMENT_COUNT: self.commit_comment_count,
                 StringKeyUtils.STR_KEY_STATUS_TOTAL: self.status_total,
                 StringKeyUtils.STR_KEY_STATUS_ADDITIONS: self.status_additions,
                 StringKeyUtils.STR_KEY_STATUS_DELETIONS: self.status_deletions,
                 StringKeyUtils.STR_KEY_HAS_FILE_FETCHED: self.has_file_fetched,
                 StringKeyUtils.STR_KEY_TREE_OID: self.tree_oid}

        return items

    class parser(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Commit()
                res.sha = src.get(StringKeyUtils.STR_KEY_SHA, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_NODE_ID, None)

                commitData = src.get(StringKeyUtils.STR_KEY_COMMIT, None)
                if commitData is not None and isinstance(commitData, dict):

                    authorData = commitData.get(StringKeyUtils.STR_KEY_AUTHOR, None)
                    if authorData is not None and isinstance(authorData, dict):
                        res.commit_author_date = authorData.get(StringKeyUtils.STR_KEY_DATE, None)
                        if res.commit_author_date is not None:
                            res.commit_author_date = datetime.strptime(res.commit_author_date,
                                                                       StringKeyUtils.STR_STYLE_DATA_DATE)

                    committerData = commitData.get(StringKeyUtils.STR_KEY_COMMITTER, None)
                    if committerData is not None and isinstance(committerData, dict):
                        res.commit_committer_date = committerData.get(StringKeyUtils.STR_KEY_DATE, None)
                        if res.commit_committer_date is not None:
                            res.commit_committer_date = datetime.strptime(res.commit_committer_date,
                                                                          StringKeyUtils.STR_STYLE_DATA_DATE)

                    treeData = commitData.get(StringKeyUtils.STR_KEY_TREE, None)
                    if treeData is not None and isinstance(treeData, dict):
                        res.tree_oid = treeData.get(StringKeyUtils.STR_KEY_SHA, None)

                    res.commit_message = commitData.get(StringKeyUtils.STR_KEY_MESSAGE, None)
                    res.commit_comment_count = commitData.get(StringKeyUtils.STR_KEY_COMMENT_COUNT, None)

                statusData = src.get(StringKeyUtils.STR_KEY_STATS, None)
                if statusData is not None and isinstance(statusData, dict):
                    res.status_total = statusData.get(StringKeyUtils.STR_KEY_TOTAL, None)
                    res.status_additions = statusData.get(StringKeyUtils.STR_KEY_ADDITIONS, None)
                    res.status_deletions = statusData.get(StringKeyUtils.STR_KEY_DELETIONS, None)

                authorData = src.get(StringKeyUtils.STR_KEY_AUTHOR, None)
                if authorData is not None and isinstance(authorData, dict):
                    res.author = User.parser.parser(authorData)
                    res.author_login = res.author.login

                committerData = src.get(StringKeyUtils.STR_KEY_COMMITTER, None)
                if committerData is not None and isinstance(committerData, dict):
                    res.committer = User.parser.parser(committerData)
                    res.committer_login = res.committer.login

                files = src.get(StringKeyUtils.STR_KEY_FILES, None)
                res.files = []
                if files is not None:
                    for item in files:
                        file = File.parser.parser(item)
                        file.commit_sha = res.sha
                        res.files.append(file)

                res.parents = []
                parentsData = src.get(StringKeyUtils.STR_KEY_PARENTS)
                if parentsData is not None:
                    for parent in parentsData:
                        relation = CommitRelation()
                        relation.child = res.sha
                        relation.parent = parent.get(StringKeyUtils.STR_KEY_SHA, None)
                        res.parents.append(relation)

            return res

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            res = None
            if isinstance(src, dict):
                res = Commit()
                res.sha = src.get(StringKeyUtils.STR_KEY_OID, None)
                res.node_id = src.get(StringKeyUtils.STR_KEY_ID, None)

                """获取 author_login 和 committer_login"""
                authorData = src.get(StringKeyUtils.STR_KEY_AUTHOR, None)
                if authorData is not None and isinstance(authorData, dict):
                    res.author = None
                    res.author_login = authorData.get(StringKeyUtils.STR_KEY_NAME, None)
                    res.commit_author_date = None
                committerData = src.get(StringKeyUtils.STR_KEY_COMMITTER, None)
                if committerData is not None and isinstance(committerData, dict):
                    res.committer = None
                    res.committer_login = committerData.get(StringKeyUtils.STR_KEY_NAME, None)
                    res.commit_committer_date = None
                treeData = src.get(StringKeyUtils.STR_KEY_TREE, None)
                if treeData is not None and isinstance(treeData, dict):
                    res.tree_oid = treeData.get(StringKeyUtils.STR_KEY_OID, None)

                res.commit_message = src.get(StringKeyUtils.STR_KEY_MESSAGE_BODY_V4, None)
                res.status_additions = src.get(StringKeyUtils.STR_KEY_ADDITIONS, None)
                res.status_deletions = src.get(StringKeyUtils.STR_KEY_DELETIONS, None)
                if res.status_deletions is not None and res.status_additions is not None:
                    res.status_total = res.status_additions + res.status_deletions

                res.parents = []
                parent_list = src.get(StringKeyUtils.STR_KEY_PARENTS, None)
                if parent_list is not None and isinstance(parent_list, dict):
                    parent_list_nodes = parent_list.get(StringKeyUtils.STR_KEY_NODES, None)
                    if parent_list_nodes is not None and isinstance(parent_list_nodes, list):
                        for parentData in parent_list_nodes:
                            relation = CommitRelation()
                            relation.child = res.sha
                            relation.parent = parentData.get(StringKeyUtils.STR_KEY_OID, None)
                            res.parents.append(relation)

            return res
