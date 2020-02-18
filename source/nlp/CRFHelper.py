#coding=gbk
import os
from source.config.projectConfig import projectConfig

class CRFHelper:
    
    
    def tag_line(self,words):
        chars = []
        tags = []
        temp_word = ''
        for word in words:
            word = word.strip('\t ')
            if temp_word == '':
                bracket_pos = word.find('[')
                w,h = word.split('/')
                if bracket_pos == -1:
                    if len(w) == 0: continue
                    chars.extend(w)
                    if h == 'ns':
                        tags += ['S'] if len(w) == 1 else ['B'] +\
                         ['M'] * (len(w) - 2) + ['E']
                    else:
                        tags += ['O'] * len(w)
                else:   
                    w = w[bracket_pos + 1:]
                    temp_word += w
            else:
                bracket_pos = word.find(']')
                w,h = word.split('/')
                if bracket_pos == -1:
                    temp_word += w
                else:
                    w = temp_word + w
                    h = word[bracket_pos + 1:]
                    temp_word = ''
                    if len(w) == 0: continue
                    chars.extend(w)
                    if h == 'ns':
                        tags += ['S'] if len(w) == 1 else ['B']\
                         + ['M']* (len(w) - 2) + ['E']
                    else:
                        tags += ['O']*len(w)
        
        assert temp_word == ''
        return (chars,tags)
                
        
    
    
    def corpusHandler(self,corpusPath):
        root = os.path.dirname(corpusPath)
        with open(corpusPath,encoding = 'utf-8') as corpus_f,\
        open(os.path.join(root,'train.txt'),'w',encoding = 'utf-8') as train_f,\
        open(os.path.join(root,'test.txt',),'w',encoding = 'utf-8') as test_f:
            pos = 0
            for line in corpus_f:
                line = line.strip('\r\n\t')
                if line == '':
                    continue
                isTest = True if pos % 5 == 0 else False
                words = line.split()[1:]
                if len(words) == 0:continue
                line_chars, line_tags = self.tag_line(words)
                saveObj =  test_f if isTest else train_f
                for k,v in enumerate(line_chars):
                    saveObj.write(v + '\t' + line_tags[k] + '\n')
                saveObj.write('\n')
                pos +=1
                
        
    def judge(self,path):
        with open(path,encoding='utf-8') as f:
            all_tag = 0
            loc_tag = 0
            pred_loc_tag=  0
            correct_tag = 0
            correct_loc_tag = 0
            
            status = ['B','M','E','S']
            for line in f:
                line = line.strip()
                if line == '': continue
                #print(line.split())
                _, r, p = line.split()
                all_tag +=1
                if r == p:
                    correct_tag += 1
                    if r in status:
                        correct_loc_tag +=1
                if r in status: loc_tag +=1
                if p in status: pred_loc_tag +=1
        
        loc_P = 1.0 *correct_loc_tag/pred_loc_tag
        loc_R = 1.0 *correct_loc_tag/loc_tag
        print('loc_P:{0},loc_R:{1},loc_F1:{2}'.format(loc_P\
                                        , loc_R,(2*loc_P*loc_R)/(loc_P+loc_R)))
                
                
if __name__=='__main__':
    
    #CRFHelper().corpusHandler(projectConfig.getCRFInputData())
    CRFHelper().judge(projectConfig.getCRFTestDataResult())