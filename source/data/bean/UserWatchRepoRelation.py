from source.data.bean.Beanbase import BeanBase
from source.utils.StringKeyUtils import StringKeyUtils


class UserWatchRepoRelation(BeanBase):
    """github中的记录用户关注项目"""

    def __init__(self):
        self.login = None  # 主题
        self.repo_full_name = None  # follow 的对象

    @staticmethod
    def getIdentifyKeys():
        return [StringKeyUtils.STR_KEY_LOGIN, StringKeyUtils.STR_KEY_REPO_FULL_NAME]

    @staticmethod
    def getItemKeyList():
        items = [StringKeyUtils.STR_KEY_LOGIN, StringKeyUtils.STR_KEY_REPO_FULL_NAME]

        return items

    @staticmethod
    def getItemKeyListWithType():
        items = [(StringKeyUtils.STR_KEY_LOGIN, BeanBase.DATA_TYPE_STRING),
                 (StringKeyUtils.STR_KEY_REPO_FULL_NAME, BeanBase.DATA_TYPE_STRING)]

        return items

    def getValueDict(self):
        items = {StringKeyUtils.STR_KEY_LOGIN: self.login,
                 StringKeyUtils.STR_KEY_REPO_FULL_NAME: self.repo_full_name}

        return items

    class parserV4(BeanBase.parser):

        @staticmethod
        def parser(src):
            """返回  [watchList, watchCount, lastWatchCursor] """
            watchList = []
            watchCount = 0
            lastWatchCursor = None

            if isinstance(src, dict):
                """ 从following list 来解析"""
                data = src.get(StringKeyUtils.STR_KEY_DATA, None)
                if isinstance(data, dict):
                    userData = data.get(StringKeyUtils.STR_KEY_USER, None)
                    if isinstance(userData, dict):
                        login = userData.get(StringKeyUtils.STR_KEY_LOGIN, None)

                        watchListData = userData.get(StringKeyUtils.STR_KEY_WATCHING, None)
                        if isinstance(watchListData, dict):
                            totalCount = watchListData.get(StringKeyUtils.STR_KEY_TOTAL_COUNT_V4)
                            watchCount = totalCount
                            """如果发现 totolCount 大于 50"""
                            print("login:", login, "  following:", totalCount)
                            edgeData = watchListData.get(StringKeyUtils.STR_KEY_EDGES, None)
                            if isinstance(edgeData, list):
                                for edge in edgeData:
                                    if isinstance(edge, dict):
                                        node = edge.get(StringKeyUtils.STR_KEY_NODE, None)
                                        lastWatchCursor = edge.get(StringKeyUtils.STR_KEY_CURSOR, None)
                                        if isinstance(node, dict):
                                            repo_full_name = node.get(StringKeyUtils.STR_KEY_NAME_WITH_OWNER, None)
                                            relation = UserWatchRepoRelation()
                                            relation.login = login
                                            relation.repo_full_name = repo_full_name
                                            watchList.append(relation)

                return [watchList, watchCount, lastWatchCursor]
