#coding=gbk
import jieba
from source.nlp.SplitWordHelper import SplitWordHelper

if __name__=='__main__':
    
#     '''jieba 使用demo '''
#     sent = '中文分词是文本处理不可或缺的一部分！'
#     jieba.load_userdict(projectConfig.getUserDictPath())
#     seg_list= jieba.cut(sent, cut_all = True)
#      
#     print('cut_all:',r'/'.join(seg_list))
#      
#     seg_list = jieba.cut(sent,cut_all = False)
#      
#     print('extra:','/'.join(seg_list))
#      
#     seg_list = jieba.cut(sent)
#      
#     print('moren:','/'.join(seg_list))
#      
#     seg_list = jieba.cut_for_search(sent)
#     print('serach:','/'.join(seg_list))
#   
  
#    '''分词演示demo''' 
# 
#     source = ExcelHelper().readExcelCol(projectConfig.getTestInputExcelPath(),r'Sheet2', 3, 1,False)
#     sorted_list = SplitWordHelper().getSplitWordListFromListData(source,cut_all = False,filter = True)
#     sorted_list_key = []
#     sorted_list_value = []
#     for item in sorted_list:
#         sorted_list_key.append(item[0])
#         sorted_list_value.append(item[1])
#     ExcelHelper().initExcelFile(projectConfig.getSplitWordExcelPath(), projectConfig.TEST_OUT_PUT_SHEET_NAME
#                                 ,ExcelHelper.excel_key_list_split_word)
#     ExcelHelper().writeExcelCol(projectConfig.getSplitWordExcelPath()
#                                 , projectConfig.TEST_OUT_PUT_SHEET_NAME
#                                 , 1 , 0, sorted_list_key)
#     ExcelHelper().writeExcelCol(projectConfig.getSplitWordExcelPath()
#                                 , projectConfig.TEST_OUT_PUT_SHEET_NAME
#                                 , 1 , 1, sorted_list_value)
#      

    '''词性标注demo '''
    
    sent = '中文分词是文本处理不可或缺的一步！'
    # print(SplitWordHelper().getPartOfSpeechTaggingFromListData(sent))

    print(SplitWordHelper().getEnglishStopList())
    
    



















