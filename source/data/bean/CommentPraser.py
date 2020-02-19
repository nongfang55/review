#coding=gbk
from source.data.bean.Comment import Comment 
from source.data.bean.User import User
from source.data.bean.UserPraser import UserPraser

class CommentPraser:
    
    STR_KEY_COMMENT_ID = 'id'
    STR_KEY_USER = 'user'
    STR_KEY_USER_ID = 'login'
    STR_KEY_PATH = 'path'
    STR_KEY_COMMIT_ID= 'commit_id'
    STR_KEY_BODY = 'body'
    STR_KEY_CREATE_TIME = 'created_at'
    STR_KEY_UPDATE_TIME = 'updated_at'
    STR_KEY_LINE = 'original_line'
    STR_KEY_ASSOCATION = 'author_association'
    
    
    def praser(self, src):
        
        comment = None
        if(isinstance(src,dict)):
            self.commentId = src.get(self.STR_KEY_COMMENT_ID, None)
            self.path = src.get(self.STR_KEY_PATH, None)
            self.commitId = src.get(self.STR_KEY_COMMIT_ID, None)
            self.body = src.get(self.STR_KEY_BODY, None)
            self.createTime = src.get(self.STR_KEY_CREATE_TIME, None)
            self.updateTime = src.get(self.STR_KEY_UPDATE_TIME , None)
            self.association = src.get(self.STR_KEY_ASSOCATION , None)
            self.line = src.get(self.STR_KEY_LINE, None)
            self.user = src.get(self.STR_KEY_USER, None)
            
            comment = Comment(commentId = self.commentId, path = self.path,
                              commitId = self.commitId, body = self.body,
                              createTime = self.createTime, updateTime = self.updateTime,
                              association = self.association, line = self.line)
            
            if(self.user != None and isinstance(self.user,dict)):
                user = UserPraser().praser(self.user)
                comment.setUserId(user.userId)
                comment.setUser(user)
            
            self.printData()
            comment.printData()
        else:
            comment = Comment()
            
        return comment
        
        
            
            
    def printData(self):
        print()
        print("commentId:" + str(self.commentId))
        print("path:" + self.path)
        print("commitId:" + str(self.commitId))
        print("body:" + self.body)
        print("createTime:" + str(self.createTime))
        print("updateTime:" + str(self.updateTime))
        print("association:" + str(self.association))
        print("line:" + str(self.line))
        print("user:" + str(self.user))
            
