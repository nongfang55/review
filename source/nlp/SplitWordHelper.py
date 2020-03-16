#coding=gbk
from source.config.projectConfig import projectConfig
import jieba
import jieba.posseg

class SplitWordHelper:
    '''分词助手
    
    '''
    
    def getGHDStopWordList(self):
        ''' 获取哈工大通用停用词 '''
        
        file = open(projectConfig.getStopWordHGDPath() ,mode = 'r+',encoding = 'utf-8')
        content = file.read() 
        print(content)
        return content.split('\n')

    def getEnglishStopList(self):
        file = open(projectConfig.getStopWordEnglishPath() ,mode = 'r+',encoding = 'utf-8')
        content = file.read()
        return content.split('\n')


    def getSplitWordListFromListData(self,dataList,cut_all = False,filter = False):
        '''获取给定数据列表的分词统计元组 
            cut_all 做模式区分
            filter 是否做停用词过滤
        '''
        
        stopWordList = self.getGHDStopWordList()
        tf_dict = {}
        for line in dataList:
            print(line)
            seg_list = jieba.cut(line,cut_all = cut_all)
            for w in seg_list:
#                 print(w)
                if(filter):
                    if(w in stopWordList):
                        print('filter:',w)
                        continue
                tf_dict[w] = tf_dict.get(w, 0) + 1
        print("收集分词数：",tf_dict.__len__())
        sorted_list = sorted(tf_dict.items(), key = lambda x:x[1],reverse = True)
        return sorted_list
    
    def getPartOfSpeechTaggingFromListData(self,sent):
        ''' 获取给定某个句子的词性标注
        
        '''
        seg_list = jieba.posseg.cut(sent)
        result = []
        for w,t in seg_list:
            result.append((w,t))
        return result
