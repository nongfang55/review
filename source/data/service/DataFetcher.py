#coding=gbk
from source.data.service.ApiHelper import ApiHelper
from source.utils.ExcelHelper import ExcelHelper
from source.config.projectConfig import projectConfig
from datetime import datetime
import sys
import io

class DataFetcher:
    
    DesensitizeKeyList = [0,1,4] # 需要脱敏的列
    
    def getCommentsForProject(self, owner, repo , limit = -1):
        
        ExcelHelper().initExcelFile(projectConfig.getTestoutputExcelPath(), projectConfig.TEST_OUT_PUT_SHEET_NAME)
        helper = ApiHelper(owner=owner, repo=repo)
        helper.setAuthorization(True)
        #获取项目pullrequest的数量
        #requestNumber = helper.getTotalPullRequestNumberForProject()
        requestNumber = helper.getMaxSolvedPullRequestNumberForProject()
        
        print("total pull request number:",requestNumber)
        
        language = helper.getLanguageForProject()
        print("project language:",language)
        
        resNumber = requestNumber
        rr = 0
        usefulRequestNumber = 0
        commentNumber = 0
        while(resNumber > 0):
            print("pull request:",resNumber, " now:",rr)
            comments = helper.getCommentsForPullRequest(resNumber)
            if(comments.__len__()>0):
                usefulRequestNumber = usefulRequestNumber + 1
                
            for comment in comments:
                self.writeRawCommmentIntoExcel(comment,repo,language)
                commentNumber = commentNumber + 1
                
            resNumber =  resNumber - 1
            rr =  rr + 1
            if(limit > 0  and rr> limit):
                break
        
        self.commentDesensitization()
        print("useful pull request:",usefulRequestNumber,"  total comment:", commentNumber)
        
    def writeRawCommmentIntoExcel(self,comment,repo,language):
        dataList = []
        dataList.append(repo)
        dataList.append(comment.path)
        dataList.append(language)
        dataList.append(comment.body)
        dataList.append(comment.userId)
        dataList.append(datetime.strptime(comment.createTime,ExcelHelper.STR_STYLE_DATA_DATE))
        dataList.append(comment.line)
        
        styleList = [ExcelHelper.getNormalStyle(), ExcelHelper.getNormalStyle(),ExcelHelper.getNormalStyle(),
                     ExcelHelper.getNormalStyle(), ExcelHelper.getNormalStyle(),ExcelHelper.getDateStyle(),
                     ExcelHelper.getNormalStyle()]
        
        ExcelHelper().appendExcelRowWithDiffStyle(projectConfig.getTestoutputExcelPath(), projectConfig.TEST_OUT_PUT_SHEET_NAME
                                                  , dataList, styleList)
        print(dataList)
        
    def commentDesensitization(self):
        for col in self.DesensitizeKeyList:
            ExcelHelper().desensitizationExcelCol(projectConfig.getTestoutputExcelPath(), projectConfig.TEST_OUT_PUT_SHEET_NAME\
                                                  , col, 1)
            


if __name__=="__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    fetcher = DataFetcher()
    fetcher.getCommentsForProject('ctripcorp','apollo',2000)