import os
import time
from source.config.projectConfig import projectConfig


class Logger:

    @staticmethod
    def logi(content):
        f = open(projectConfig.getLogPath() + os.sep + "fetch.log", "a")
        # 写入文件
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\tINFO\t" + content + "\n")

    @staticmethod
    def loge(content):
        f = open(projectConfig.getLogPath() + os.sep + "fetch.log", "a")
        # 写入文件
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\tERROR\t" + content + "\n")