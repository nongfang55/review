# coding=gbk
from source.utils.pandas.pandasHelper import pandasHelper
import re


class TextCompareUtils:
    """patch的文本比较类"""

    @staticmethod
    def patchParser(text):
        """patch的文本解析"""
        changes = []  # 一个patch可能会有多个改动    [(开始行,共,版本二开始,共) -> [+, ,-,....]]
        print(text)
        print('-' * 50)

        headMatch = re.compile(r'@@(.)+@@')
        numberMatch = re.compile(r'[^0-9]+')

        status = None
        lines = []
        for t in text.split('\n'):
            head = headMatch.search(t)
            if head:
                if status is not None:
                    changes.append([status, lines])
                    status = None
                    lines = []

                print(head.group())
                numbers = [int(x) for x in numberMatch.split(head.group()) if x.__len__() > 0]
                print(numbers)
                if numbers.__len__() == 4:
                    status = tuple(numbers)
                elif numbers.__len__() == 2:
                    numbers = (numbers[0], 1, numbers[1], 1)
                status = numbers
            else:
                lines.append(t[0])
        if status is not None:
            changes.append([status, lines])
        print(changes)
        return changes

    @staticmethod
    def simulateTextChanges(patches1, patches2, targetLine):
        """通过对patch做文本模拟来获得变化最后结果"""

        changes1 = []
        changes2 = []

        # minLine = float('inf')
        maxLine = 0  # 节省空间查询涉及变化的上界限
        for patch in patches1:
            change = TextCompareUtils.patchParser(patch)  # 每一个patch可能有几个变化，是平行关系
            changes1.insert(0, change)

            for c in change:
                # minLine = min(minLine, c[0])
                maxLine = max(maxLine, c[0][0] + c[1].__len__(), c[0][2] + c[1].__len__())
        for patch in patches2:
            change = TextCompareUtils.patchParser(patch)
            changes2.insert(0, change)
            for c in change:
                # minLine = min(minLine, c[0])
                maxLine = max(maxLine, c[0][0] + c[1].__len__(), c[0][2] + c[1].__len__())

        maxLine = max(maxLine + 20, targetLine + 20)
        print(maxLine)

        text = [x for x in range(1, maxLine)]  # 生成模拟文本
        print(text)

        """对于返回路径反着来"""
        for changes in changes1:
            offset = 0
            for change in changes:
                cur = change[0][2] - offset
                print('start  offset:', change[0], offset)
                offset = 0
                for c in change[1]:
                    if c == ' ':
                        cur += 1
                    elif c == '-':
                        text.insert(cur - 1, 0)
                        cur += 1
                    elif c == '+':
                        text.pop(cur - 1)
                offset = change[1].count('+') - change[1].count('-')

        """前进路径为正"""
        for changes in changes2:
            offset = 0
            for change in changes:
                cur = change[0][0] + offset
                print('start  offset:', change[0], offset)
                offset = 0
                for c in change[1]:
                    if c == ' ':
                        cur += 1
                    elif c == '+':
                        text.insert(cur - 1, 0)
                        cur += 1
                    elif c == '-':
                        text.pop(cur - 1)
                offset = change[1].count('+') - change[1].count('-')
        print(text)
        return text

    @staticmethod
    def getClosedFileChange(patches1, patches2, commentLine):
        """获得某个评论最近的line   如果超过十则返回-1  patch的顺序是从根到最新"""

        text = TextCompareUtils.simulateTextChanges(patches1, patches2, commentLine)

        if commentLine not in text:
            """评论行不在 变化之后的文本当中，说明本行变化，返回0"""
            return 0
        else:
            curLine = text.index(commentLine)

            """分两个方向查找 先向0方向查找"""
            upChange = None
            downChange = None
            for i in range(1, min(11, curLine)):
                if text[curLine - i] != commentLine - i:
                    upChange = i
                    break
            for i in range(1, min(11, text.__len__() - curLine)):
                if text[curLine + i] != curLine + i:
                    downChange = i
                    break

            if upChange is None and downChange is None:
                return -1

            if downChange is None:
                return upChange
            elif upChange is None:
                return downChange
            else:
                return min(upChange, downChange)


if __name__ == '__main__':
    data = pandasHelper.readTSVFile(r'C:\Users\ThinkPad\Desktop\select____from_gitCommit_gitFile__where_.tsv',
                                    pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
    text = data.as_matrix()[0][18]
    # print(TextCompareUtils.patchParser(text))
    # print(text)
    # for t in text.split('\n'):
    #     print(t)
    TextCompareUtils.simulateTextChanges([text], [text], 127)
