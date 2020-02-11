#coding=gbk
from source.data.User import User

class UserPraser:
    
    STR_KEY_NAME = 'login'
    STR_KEY_USER_ID = 'id'
    STR_KEY_USER_TYPE = 'type'
    STR_KEY_USER_EMAIL = 'email'
    
    
    def praser(self, src):
        
        user = None
        if(isinstance(src, dict)):
            self.name = src.get(self.STR_KEY_NAME, None)
            self.userId = src.get(self.STR_KEY_USER_ID, None)
            self.userType = src.get(self.STR_KEY_USER_TYPE, None)
            self.email = src.get(self.STR_KEY_USER_EMAIL, None)
            user = User(name = self.name, userId=self.userId,userType = self.userType, email = self.email)
        else:
            user = User()   
        self.printData()
        return user
        
        
    def printData(self):
        print()
        print("name:" + str(self.name))
        print("userId:" + str(self.userId))
        print("userType:" + str(self.userType))
        print("email:" + str(self.email))
        