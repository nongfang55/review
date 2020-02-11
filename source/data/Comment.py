#coding=gbk
from source.data.User import User
class Comment:
    '''
       Comment 数据类，为从api中提取的具体实现
    
    '''
    def __init__(self,commentId = None,userId = None,path = None, commitId = None
                 ,body = None,createTime = None,updateTime = None,association = None,
                 line = None, user = None):
        self.commentId = commentId
        self.userId = userId
        self.path = path
        self.commitId = commitId
        self.body = body
        self.createTime = createTime
        self.updateTime = updateTime
        self.association = association
        self.line = line
        self.user = user
        
    def printData(self):
        print()
        print("commentId:" + str(self.commentId))
        print("userId:" + str(self.userId))
        print("path:" + self.path)
        print("commitId:" + str(self.commitId))
        print("body:" + self.body) 
        print("createTime:" + str(self.createTime))
        print("updateTime:" + str(self.updateTime))
        print("association:" + str(self.association))
        print("line:" + str(self.line))
        if(isinstance(self.user, User)):
            self.user.printData()
        
        
    def setCommentId(self,commentId):
        self.commentId = commentId
        return self
    
    def setUserId(self,userId):
        self.userId = userId
        return self
    
    def setPath(self,path):
        self.path = path
        return self
    
    def setCommitId(self,commitId):
        self.commitId = commitId
        return self
    
    def setBody(self,body):
        self.body = body
        return self
    
    def setCreateTime(self,createTime):
        self.createTime = createTime
        return self
    
    def setUpdateTime(self,updateTime):
        self.updateTime = updateTime
        return self
    
    def setAssociation(self, association):
        self.association = association
        return self
    
    def  setLine(self, line):
        self.line = line
        return self     
    
    def setUser(self, user):
        if(user != None and isinstance(user,User)):
            self.user = user
        return self

    
            