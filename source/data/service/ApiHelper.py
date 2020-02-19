#coding=gbk
import requests
import sys
import json
import io
import time

import os
from source.config import projectConfig
from source.config import configPraser
from source.data.bean.CommentPraser import CommentPraser
from source.utils.TableItemHelper import TableItemHelper
from source.utils.StringKeyUtils import StringKeyUtils
from _datetime import datetime
from math import ceil

class ApiHelper:
    
    API_GITHUB = 'https://api.github.com'
    API_REVIEWS_FOR_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number/reviews'
    API_PULL_REQUEST_FOR_PROJECT = '/repos/:owner/:repo/pulls'
    API_COMMENTS_FOR_REVIEW = '/repos/:owner/:repo/pulls/:pull_number/reviews/:review_id/comments'
    API_COMMENTS_FOR_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number/comments'
    API_PROJECT = '/repos/:owner/:repo'
    
    
    #用于替换的字符串
    STR_HEADER_AUTHORIZAITON = 'Authorization'
    STR_HEADER_TOKEN = 'token ' #有空格
    STR_HEADER_ACCEPT = 'Accept'
    STR_HEADER_MEDIA_TYPE = 'application/vnd.github.comfort-fade-preview+json' 
    STR_HEADER_RATE_LIMIT_REMIAN = 'X-RateLimit-Remaining'
    STR_HEADER_RATE_LIMIT_RESET = 'X-RateLimit-Reset'

    STR_OWNER = ':owner'
    STR_REPO = ':repo'
    STR_PULL_NUMBER = ':pull_number'
    STR_REVIEW_ID = ':review_id'

        
    STR_PARM_STARE = 'state'
    STR_PARM_ALL = 'all'
    STR_PARM_OPEN = 'open'
    STR_PARM_CLOSED = 'closed'
    
    RATE_LIMIT = 5
    

    
    
    def __init__(self,owner,repo): #设置对应的仓库和所属
        self.owner = owner
        self.repo = repo
        self.isUseAuthorization = False
        
    def setOwner(self,owner):
        self.owner = owner
        
    def setRepo(self,repo):
        self.repo = repo
        
    def setAuthorization(self, isUseAuthorization):
        self.isUseAuthorization = isUseAuthorization
        
    def getAuthorizationHeaders(self, header):
        if(header != None and isinstance(header, dict)):
            if(self.isUseAuthorization):
                if (configPraser.configPraser.getAuthorizationToken()) :
                    header[self.STR_HEADER_AUTHORIZAITON] = (self.STR_HEADER_TOKEN 
                                        + configPraser.configPraser.getAuthorizationToken())
        
        return header
    
    def getMediaTypeHeaders(self,header):
        if(header != None and isinstance(header, dict)):
            header[self.STR_HEADER_ACCEPT] = self.STR_HEADER_MEDIA_TYPE
            
        return header
        
    def getPullRequestsForPorject(self,state = STR_PARM_OPEN): 
        '''获取一个项目的pull reuqest的列表，但是 只能获取前30个  没参数的时候默认是open
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers, params = {self.STR_PARM_STARE : state})
        self.printCommon(r)
        self.judgeLimit(r)
        if(r.status_code != 200):
            return list()
        
        res = list()
        for request in r.json():
            res.append(request.get(self.STR_KET_NUMBER))
            print(request.get(self.STR_KET_NUMBER))
            
        print(res.__len__())       
        return res
    
    
    def getLanguageForPorject(self): 
        '''获取一个项目的pull reuqest的列表，但是 只能获取前30个  没参数的时候默认是open
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_PROJECT
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers)
        self.printCommon(r)
        self.judgeLimit(r)
        if(r.status_code != 200):
            return StringKeyUtils.STR_KEY_LANG_OTHER
        
        return r.json().get(StringKeyUtils.STR_KEY_LANG, StringKeyUtils.STR_KEY_LANG_OTHER)
    
    def getTotalPullRequestNumberForProject(self):
        '''通过获取最新的pull request的编号来获取总数量  获取参数为all
         
        '''
        
        if(self.owner == None or self.repo == None):
            return -1
        
        api = self.API_GITHUB + self.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers, params = {self.STR_PARM_STARE : self.STR_PARM_ALL})
        self.printCommon(r)
        self.judgeLimit(r)
        if(r.status_code != 200):
            return -1
        
        list = r.json()
        if(list.__len__() > 0):
            request = list[0]
            return request.get(self.STR_KET_NUMBER, -1)
        else:
            return -1
        
        
        
    def getMaxSolvedPullRequestNumberForProject(self):
        '''通过获取最新的pull request的编号来获取总数量  获取参数为all
         
        '''
        
        if(self.owner == None or self.repo == None):
            return -1
        
        api = self.API_GITHUB + self.API_PULL_REQUEST_FOR_PROJECT
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers, params = {self.STR_PARM_STARE : self.STR_PARM_CLOSED})
        self.printCommon(r)
        self.judgeLimit(r)
        if(r.status_code != 200):
            return -1
        
        list = r.json()
        if(list.__len__() > 0):
            request = list[0]
            return request.get(self.STR_KET_NUMBER, -1)
        else:
            return -1
        
    def getCommentsForPullRequest(self, pull_number):
        '''获取一个pullrequest的comemnts  可以获取行号
        
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_COMMENTS_FOR_PULL_REQUEST
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
        api = api.replace(self.STR_PULL_NUMBER,str(pull_number))
        
        #print(api)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        headers = self.getMediaTypeHeaders(headers)
        r = requests.get(api, headers = headers)
        self.printCommon(r) 
        self.judgeLimit(r)
        if(r.status_code != 200):
            return list()
        
        res = list()
        for review in r.json():
            #print(review)
            #print(type(review))
            praser = CommentPraser()
            res.append(praser.praser(review))
            
        return res
        
        
        
        
    def getCommentsForReview(self, pull_number, review_id):
        '''获取一个review的相关comments  这个接口无法获取行号
        
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_COMMENTS_FOR_REVIEW
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
        api = api.replace(self.STR_PULL_NUMBER,str(pull_number))
        api = api.replace(self.STR_REVIEW_ID,str(review_id))
        
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers)
        self.printCommon(r) 
        self.judgeLimit(r)
        if(r.status_code != 200):
            return list()
        
        
        res = list()
        for review in r.json():
            #print(review)
            #print(type(review))
            praser = CommentPraser()
            res.append(praser.praser(review))
        return res
        
                
    def getReviewForPullRequest(self,pull_number):    
        '''获取一个pull_reuqest的review的id列表
        
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_REVIEWS_FOR_PULL_REQUEST
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
        api = api.replace(self.STR_PULL_NUMBER,str(pull_number))
        
                
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers)
        self.printCommon(r) 
        self.judgeLimit(r)
        if(r.status_code != 200):
            return list()
        
        
        res = list()
        for review in r.json():
            #print(review)
            res.append(review.get(StringKeyUtils.STR_KEY_ID))
            
        return res
        
        
    def printCommon(self,r):
        if(isinstance(r, requests.models.Response)):
            print(type(r))
            print(r.json())
            print(r.text.encode(encoding='utf_8', errors='strict'))
            print(r.headers)
            print("status:" , r.status_code.__str__())
            print("remaining:" , r.headers.get(self.STR_HEADER_RATE_LIMIT_REMIAN))
            print("rateLimit:" , r.headers.get(self.STR_HEADER_RATE_LIMIT_RESET))
             
    def judgeLimit(self,r):
        if(isinstance(r, requests.models.Response)):
            remaining = int(r.headers.get(self.STR_HEADER_RATE_LIMIT_REMIAN))
            rateLimit = int(r.headers.get(self.STR_HEADER_RATE_LIMIT_RESET))
            if(remaining < self.RATE_LIMIT):
                print("start sleep:",ceil(rateLimit - datetime.now().timestamp() + 1))
                time.sleep(ceil(rateLimit - datetime.now().timestamp() + 1))
                print("sleep end")
            
            
        
    def getInformationForPorject(self): 
        '''获取一个项目的信息  返回一个字典
        '''
        if(self.owner == None or self.repo == None):
            return list()
        
        api = self.API_GITHUB + self.API_PROJECT
        api = api.replace(self.STR_OWNER,self.owner)
        api = api.replace(self.STR_REPO,self.repo)
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
        
        headers = {}
        headers = self.getAuthorizationHeaders(headers)
        r = requests.get(api, headers = headers)
        self.printCommon(r)
        self.judgeLimit(r)
        if(r.status_code != 200):
            return None
        
        rawData = r.json()
        res = {}
        for item in TableItemHelper.getProjectTableItem():
            res[item] = rawData.get(item, None) 
            
        #print(res)
        return res
        
        
            
            
        
        
if __name__=="__main__":
    helper = ApiHelper('rails','rails')
    helper.setAuthorization(True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    #print(helper.getReviewForPullRequest(38211))
    #helper.getPullRequestsForPorject(state = ApiHelper.STR_PARM_ALL)
#     print("total:" + helper.getTotalPullRequestNumberForProject().__str__())
#     print(helper.getCommentsForReview(38211,341374357))
#     print(helper.getCommentsForPullRequest(38211))
#     print(helper.getCommentsForPullRequest(38211))
#     print(helper.getMaxSolvedPullRequestNumberForProject())
#     print(helper.getLanguageForPorject())
    print(helper.getInformationForPorject())
    
        
    