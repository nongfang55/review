'''
@author: ThinkPad
'''


import sys
import os
import nltk
from nltk.probability import FreqDist
from nltk.book import *


def lexical_diversity(text):
    return len(text)/len(set(text))


def percentage(count,total):
    return 100*count/total



def debug():
#     print(lexical_diversity('1323234242'))
#     print(percentage(1, 10))
#     
# #     sent2 = ['The', 'family', 'of', 'Dashwood', 'had', 'long',
# # 'been', 'settled', 'in', 'Sussex', '.'] 
# #     
# #     print(sent2)  
# #     print(sorted(sent2))
# #     print(sent2)
# #     print(len(sent2))
# #     print(sent2.count('The'))
# #     
#     
    saying = ['After', 'all', 'is', 'said', 'and', 'done','more', 'is', 'said', 'than', 'done']
    tokens = set(text1)
    tokens = sorted(tokens)
    print(tokens[-2:])
    print(FreqDist(text1).items())
 #   FreqDist(text1).plot()
if __name__=="__main__":
    debug()