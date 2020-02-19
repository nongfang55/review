#coding=gbk
import uuid
class DesensitizationHelper:
    ''' 用于信息的脱敏
    
    '''
    
    @staticmethod
    def desensitization(rawData):
        dataDict = {}
        resList = []
        for item in rawData:
            if(dataDict.get(item,None) == None):
                dataDict[item] = uuid.uuid4().__str__()
            resList.append(dataDict[item])
        
        print(dataDict)
            
        return resList
    
    
if __name__=="__main__":
    
    print(DesensitizationHelper.desensitization([1,2,3,1]))
    