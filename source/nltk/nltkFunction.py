
def stemList(wordList):
    """用于过滤单词还原到词干的状态，来减少类似的单词数"""

    """基于SNOWball 的词干提取"""
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    res = []
    for word in wordList:
        stemmedWord = stemmer.stem(word)
        res.append(stemmedWord)
    return res


if __name__ == "__main__":
    print(stemList(['one', 'ones', 'apple', 'apples']))
