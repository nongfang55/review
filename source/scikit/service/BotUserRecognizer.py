from source.utils.StringKeyUtils import StringKeyUtils


class BotUserRecognizer:
    """用户识别用户是否是机器
       一半的机器人名字都带[bot] 如 codecov[bot]
       但是少数也有user类型但是为机器人的情况
       这里维护一个机器人表 游动加入
    """

    """手动维护的列表"""
    BOT_TABLE = ['stickler-ci', 'codecov-io', 'rails-bot', 'mention-bot',
                 'babel-bot', 'symfony-skeleton-bot', 'akka-ci', 'buildsize',
                 'stale', 'netty-bot', 'codecov', 'label-actions', 'salt-jenkins',
                 'facebook-github-bot', 'reactjs-bot', 'pull-bot', 'googlebot',
                 'mary-poppins', 'ngbot[bot]', 'ngbot', 'sklearn-lgtm', 'pep8speaks',
                 'fastlane-bot-helper', 'netkins', 'GordonTheTurtle', 'lightbend-cla-validator',
                 'sizebot']

    @staticmethod
    def isBot(name):
        if name.find(StringKeyUtils.STR_NAME_BOT) != -1:
            return True
        elif name in BotUserRecognizer.BOT_TABLE:
            return True
        else:
            return False