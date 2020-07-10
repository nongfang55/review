# coding=gbk
from source.utils.pandas.pandasHelper import pandasHelper
import re


class TextCompareUtils:
    """patch的文本比较类"""

    @staticmethod
    def patchParser(text):
        """patch的文本解析"""

        """ patch 文本格式示例说明
        
         @@ -35,9 +36,8 @@ ruby <%= \"'#{RUBY_VERSION}'\" -%>
         # gem 'rack-cors'
         
         <%- end -%>
         -# The gems below are used in development, but if they cause problems it's OK to remove them
         -
         <% if RUBY_ENGINE == 'ruby' -%>
         +# The gems below are used in development, but if they cause problems it's OK to remove them
         group :development, :test do
         # Call 'byebug' anywhere in the code to stop execution and get a debugger console
         gem 'byebug', platforms: [:mri, :mingw, :x64_mingw]
         
         
         说明：  -35,9,+36,8  说明这个改动在上个版本是35行开始，下面有9行是原来版本的内容
                                           下个版本36行开始，下面8行是新版本的内容
                                           "+" 行是新版本独有的内容
                                           "-" 行是老版本独有的内容
                                           
                patch 的第一行不记内容
        注： 这是我自己摸索的理解 @张逸凡
        """

        changes = []  # 一个patch可能会有多个改动    [(开始行,共,版本二开始,共) -> [+, ,-,....]]
        # print(text)
        # print('-' * 50)

        headMatch = re.compile(r'@@(.)+@@')
        numberMatch = re.compile(r'[^0-9]+')

        status = None
        lines = []
        for t in text.split('\n'):
            """按行拆分  依次解析"""
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
                    """可能有只有两个的特殊情况"""
                    numbers = (numbers[0], 1, numbers[1], 1)
                    status = numbers
            else:
                """收集正负符号 即每一行修改状态"""
                lines.append(t[0])
        if status is not None:
            changes.append([status, lines])
        # print(changes)
        return changes

    @staticmethod
    def simulateTextChanges(patches1, patches2, targetLine):
        """通过对patch做文本模拟来获得变化最后结果
          对Patch1 的改动做负模拟
          对Patch2 的改动做正模拟

        """

        changes1 = []
        changes2 = []

        # minLine = float('inf')
        maxLine = 0  # 节省空间查询涉及变化的上界限
        for patch in patches1:
            change = TextCompareUtils.patchParser(patch)  # 每一个patch可能有几个变化，是平行关系
            changes1.insert(0, change)

            for c in change:
                # minLine = min(minLine, c[0])
                """找修改本文可能会涉及的最大行数 减少文本模拟的负担"""
                maxLine = max(maxLine, c[0][0] + c[1].__len__(), c[0][2] + c[1].__len__())
        for patch in patches2:
            change = TextCompareUtils.patchParser(patch)
            changes2.insert(0, change)
            for c in change:
                # minLine = min(minLine, c[0])
                maxLine = max(maxLine, c[0][0] + c[1].__len__(), c[0][2] + c[1].__len__())

        maxLine = max(maxLine + 20, targetLine + 20)
        print(maxLine)

        """通过一个数组来模拟文本的变化"""
        text = [x for x in range(1, maxLine)]  # 生成模拟文本
        print(text)

        """本文模拟就是 一个数据 数组中的数字代表行号  对这个数组使用Patch做增改"""

        """对于返回路径反着来  即改动中加内容为减内容   减内容为加内容"""
        for changes in changes1:
            """计算模拟时候带来的偏移"""
            offset = 0
            for change in changes:
                cur = change[0][2] - offset
                print('start  offset:', change[0], offset)
                for c in change[1]:
                    if c == ' ':
                        cur += 1
                    elif c == '-':
                        text.insert(cur - 1, 0)
                        cur += 1
                    elif c == '+':
                        text.pop(cur - 1)
                """删减行导致原来的起始行数错位  需要计算偏移补正"""

                """修正偏移未累加导致的bug"""
                offset += change[1].count('+') - change[1].count('-')

        """前进路径为正"""
        for changes in changes2:
            offset = 0
            for change in changes:
                cur = change[0][0] + offset
                print('start  offset:', change[0], offset)
                for c in change[1]:
                    if c == ' ':
                        cur += 1
                    elif c == '+':
                        text.insert(cur - 1, 0)
                        cur += 1
                    elif c == '-':
                        text.pop(cur - 1)
                offset += change[1].count('+') - change[1].count('-')
        print(text)
        return text

    @staticmethod
    def getClosedFileChange(patches1, patches2, commentLine):
        """获得某个评论最近的line   如果超过十则返回-1  patch的顺序是从根到最新"""

        text = TextCompareUtils.simulateTextChanges(patches1, patches2, commentLine)

        """text是模拟commit操作之后的文本"""

        if commentLine not in text:
            """评论行不在 变化之后的文本当中，说明本行变化，返回0"""
            return 0
        else:
            """寻找距离品论最近的改动距离"""
            curLine = text.index(commentLine)

            """分两个方向查找 先向0方向查找"""
            upChange = None
            downChange = None
            for i in range(1, min(11, curLine)):
                """发现错位或者内容为0的文本为止"""
                if text[curLine - i] != commentLine - i:
                    upChange = i
                    break
            for i in range(1, min(11, text.__len__() - curLine)):
                if text[curLine + i] != commentLine + i:
                    downChange = i
                    break

            """-1表示附近没有改动"""
            if upChange is None and downChange is None:
                return -1

            if downChange is None:
                return upChange
            elif upChange is None:
                return downChange
            else:
                return min(upChange, downChange)

    @staticmethod
    def getStartLine(text, originalPosition):
        """通过文本originalPosition来推断出
           review comment的original_StartLine 和 side

           注： 2020.7.9 发现 position 是代指最新commit上面位置，
        """

        """patch 解析"""
        """虽然 patch 可能会解析多个 但是数据库14万comment 只有37个疑似情况
           加上try catch忽略特殊情况
        """
        side = None
        original_startLine = None
        try:
            changes = TextCompareUtils.patchParser(text)
            """找到对应的改动"""
            """第一行不算行"""
            originalPosition += 1
            pos = 0
            posLine = 0
            while posLine + changes[pos][1]. __len__() + 1 < originalPosition:
                posLine += changes[pos][1].__len__() + 1
                pos += 1

            original_startLine = None
            side = None
            left_start, _, right_start, _ = changes[pos][0]
            status = changes[pos][1]
            if originalPosition is not None:
                if status[originalPosition - posLine - 2] == '-':
                    side = 'LEFT'
                    original_startLine = left_start
                else:
                    side = 'RIGHT'
                    original_startLine = right_start
                offset = 0
                if side == 'LEFT':
                    for i in range(0, max(originalPosition - posLine - 1, 0)):
                        if status[i] != '+':
                            offset += 1
                if side == 'RIGHT':
                    for i in range(0, max(originalPosition - posLine - 1, 0)):
                        if status[i] != '-':
                            offset += 1
                original_startLine = original_startLine + offset - 1
        except Exception as e:
            print(e)

        return original_startLine, side

    @staticmethod
    def ConvertLeftToRight(text, originalPosition):
        """由于 comment 可能会加在  老版本(LEFT) 和新版本 (RIGHT)两个
        上面，老版本的行数和新版本的行数并不相同。 对于comment和chnage的
        版本比较而言   是新版本（RIGHT） 和  change 版本的比较，需要把
        LEFT的行数 转化为 最近的RIGHT 行数

           注： 2020.7.9 同之前相同，当Patch由多个小patch组成的时候，
           现在拼接出来可能会有1行左右的误差  误差原因不明
           由于LEFT 的 comment 相对比较少 （LEFT:RIGHT = 1 ： 10）
           先转化查看效果
        """
        try:
            changes = TextCompareUtils.patchParser(text)
            """找到对应的改动"""
            """第一行不算行"""
            originalPosition += 1
            pos = 0
            posLine = 0
            while posLine + changes[pos][1]. __len__() + 1 < originalPosition:
                posLine += changes[pos][1].__len__() + 1
                pos += 1

            left_start, _, right_start, _ = changes[pos][0]
            status = changes[pos][1]
            if originalPosition is not None:
                """认定调用的comment都是LEFT"""
                offset = 0
                for i in range(0, max(originalPosition - posLine - 1, 0)):
                    """统计 RIGHT 版本有的行数"""
                    if status[i] != '-':
                        offset += 1

                return right_start + offset - 1
        except Exception as e:
            print(e)

        return None


if __name__ == '__main__':
    # data = pandasHelper.readTSVFile(r'C:\Users\ThinkPad\Desktop\select____from_gitCommit_gitFile__where_.tsv',
    #                                 pandasHelper.INT_READ_FILE_WITHOUT_HEAD)
    # text = data.as_matrix()[0][18]
    # print(TextCompareUtils.patchParser(text))
    # print(text)
    # for t in text.split('\n'):
    #     print(t)
    #text = "@@ -20,6 +20,7 @@ ruby <%= \"'#{RUBY_VERSION}'\" -%>\n <% end -%>\n <% end -%>\n \n+\n # Optional gems needed by specific Rails features:\n \n # Use bcrypt to encrypt passwords securely. Works with https://guides.rubyonrails.org/active_model_basics.html#securepassword\n@@ -35,9 +36,8 @@ ruby <%= \"'#{RUBY_VERSION}'\" -%>\n # gem 'rack-cors'\n \n <%- end -%>\n-# The gems below are used in development, but if they cause problems it's OK to remove them\n-\n <% if RUBY_ENGINE == 'ruby' -%>\n+# The gems below are used in development, but if they cause problems it's OK to remove them\n group :development, :test do\n   # Call 'byebug' anywhere in the code to stop execution and get a debugger console\n   gem 'byebug', platforms: [:mri, :mingw, :x64_mingw]\n@@ -75,7 +75,6 @@ group :test do\n   # Easy installation and use of web drivers to run system tests with browsers\n   gem 'webdrivers'\n end\n-\n <%- end -%>\n \n <% if depend_on_bootsnap? -%>"
    #text = "@@ -2,9 +2,9 @@ a\n b\n c\n d\n-e\n a\n b\n+aa\n c\n d\n a\n@@ -123,10 +123,8 @@ a\n d\n f\n d\n-\n+aaa\n t\n-ty\n-u\n u\n u\n y\n@@ -233,10 +231,10 @@ ter\n t\n t\n d\n-\n+zzzz\n t\n-rt\n-ert\n+zz\n+erzt\n r\n t\n rt"
    text = """@@ -22,8 +22,8 @@ export async function run(
   const lockfile = args.length ? await Lockfile.fromDirectory(config.cwd, reporter) : new Lockfile();
   let addArgs = [...args];
 
-  if (args.length) {
-    const [dependency] = args;
+  if (addArgs.length) {
+    const [dependency] = addArgs;
     const manifest = await config.readRootManifest() || {};
     const dependencies = manifest.dependencies || {};
     const remoteSource = dependencies[dependency];"""
    print(TextCompareUtils.getStartLine(text, 17))
    # print(TextCompareUtils.ConvertLeftToRight(text, 33))
