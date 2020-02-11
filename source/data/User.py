#coding=gbk
class User:
    ''' github用户类数据 
     
    '''
    def __init__(self, userId = None, name = None, userType = None, email = None):
        self.userId = userId
        self.name = name
        self.userType = userType
        self.email = email
        
    def printData(self):
        print()
        print("name:" + str(self.name))
        print("userId:" + str(self.userId))
        print("userType:" + str(self.userType))
        print("email:" + str(self.email))
        
    def setUserId(self,userId):
        self.userId = userId
        return self
    
    def setName(self,name):
        self.name = name
        return self
    
    def setUserType(self,userType):
        self.userType = userType
        return self
    
    def setEmail(self,email):
        self.email = email
        return self