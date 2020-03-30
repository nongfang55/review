# coding=gbk
import math
import re

import numpy
import pronouncing
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from pandas import Series

from source.config.projectConfig import projectConfig
from source.nlp import LanguageKeyWordHelper
from source.nlp.SplitWordHelper import SplitWordHelper
from source.utils.pandas.pandasHelper import pandasHelper


class FleshReadableUtils:

    # @staticmethod
    # def word_list(comment):
    #
    #     word_re = re.compile(r'[^A-Za-z\']+')
    #     words = [x for x in word_re.split(comment.lower()) if x.__len__() != 0]
    #     # print("单词数:" + str(len(words)))
    #     # print(words)
    #     return words

    @staticmethod
    def word_list(comment):
        """分词正则去除了带引号的场景"""
        word_re = re.compile(r'[^A-Za-z]+')
        words = [x for x in word_re.split(comment.lower()) if x.__len__() != 0]
        # print("单词数:" + str(len(words)))
        # print(words)
        return words

    @staticmethod
    def sentence(comment):
        point_re = re.compile(r'\.|\?|\!')
        point = [x for x in point_re.split(comment) if x.__len__() != 0]
        print("句子长度:" + str(len(point)))
        print(point)
        return point

    @staticmethod
    def sentenceQuestionCount(comment):
        point_re = re.compile(r'\?+')
        point = point_re.findall(comment)
        print("问句数量:" + str(len(point)))
        print(point)
        return len(point)

    @staticmethod
    def CodeElement(comment):
        code_re = re.compile("```[^`]*```|`[^`]+`")
        codes = re.findall(code_re, comment)
        print("codes:")
        print(codes)
        return codes

    @staticmethod
    def get_pronouncing_num(word):
        try:
            pronunciating_list = pronouncing.phones_for_word(word)
            num = pronouncing.syllable_count(pronunciating_list[0])
        except Exception as e:
            print("音节计算异常，异常单词：" + word)
            return math.ceil(2)
        else:
            return num

    @staticmethod
    def get_pronouncing_nums(words):
        counts = 0
        for word in words:
            counts += FleshReadableUtils.get_pronouncing_num(word)
        print('音节总数：', str(counts))
        return counts


if __name__ == "__main__":

    data = pandasHelper.readTSVFile(projectConfig.getReviewCommentTestData())
    comments = data.as_matrix()[:, (2, 4)]
    print(comments.shape)

    readable = []  # 可读性
    stopWordRate = []  # 停句率
    questionRatio = []  # 问题率
    codeElementRatio = []  # 代码元素率
    stopKeyRatio = []  # 关键字率
    conceptualSimilarity = []  # 概念相似度
    badCase = []

    stopwords = SplitWordHelper().getEnglishStopList()
    languageKeyWords = LanguageKeyWordHelper.LanguageKeyWordLanguage.getRubyKeyWordList()

    for line in comments:

        # if '??' in comment:
        #     print(comment)
        #     print("-" * 200)
        code_diff = line[1].strip()
        comment = line[0].strip()
        print("comment:")
        print(comment)

        word_list = FleshReadableUtils.word_list(comment)
        sentence = FleshReadableUtils.sentence(comment)
        # ASL 数单词/句子数
        word_num = len(word_list)
        sentence_num = len(sentence)

        if word_num == 0 or sentence_num == 0:  # 对于特殊句子舍弃
            # readable.append(None)
            # stopWordRate.append(None)
            # questionRatio.append(None)
            print("bad case")
            print(comment)
            continue

        """做可读性的计算"""
        ASL = word_num / sentence_num

        # ASW 音节数/单词数
        pronouncing_nums = FleshReadableUtils.get_pronouncing_nums(word_list)
        ASW = pronouncing_nums / word_num

        RE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
        print("RE:", RE)
        readable.append(RE)

        if RE > 100 or RE < 0:
            badCase.append((comment, RE))

        """做stop word ratio的计算"""
        stopwordsInComments = [x for x in word_list if x in stopwords]
        print("word list:")
        print(word_list.__len__())
        print("stop words:")
        print(stopwordsInComments.__len__())
        stopWordRate.append(stopwordsInComments.__len__() / word_list.__len__())

        """做question ratio计算"""  # 通过正则表达式句子拆分减去不带问号的句子拆分数量
        questionsCount = FleshReadableUtils.sentenceQuestionCount(comment)
        print("questions count")
        questionRatio.append(questionsCount / sentence.__len__())

        """做Code Element Ratio计算"""
        codes = FleshReadableUtils.CodeElement(comment)
        codeElementCount = 0
        for code in codes:
            codeElementCount += len(FleshReadableUtils.word_list(code))
        print("code Element count")
        print(codeElementCount)
        codeElementRatio.append(codeElementCount / word_num)

        """做概念相似度计算"""
        print("diff")
        print(code_diff)

        """把改动代码的每一行作为一个文本 分词，去停用词"""
        diff_word_list = []
        for code in code_diff.split('\n'):
            diff_word_list.append([x for x in FleshReadableUtils.word_list(code) if x not in stopwords])
            # print([x for x in FleshReadableUtils.word_list(code) if x not in stopwords])
        diff_word_list = [x for x in diff_word_list if x.__len__() != 0]

        if diff_word_list.__len__() == 0:
            conceptualSimilarity.append(0)  # 对改动文本为空特殊处理
        else:
            """建立词典  获得特征数"""
            dictionary = corpora.Dictionary(diff_word_list)
            feature_cnt = len(dictionary.token2id.keys())
            """基于词典  分词列表转稀疏向量集"""
            corpus = [dictionary.doc2bow(codes) for codes in diff_word_list]
            # print("key")
            # print([x for x in word_list if x not in stopwords])
            kw_vector = dictionary.doc2bow([x for x in word_list if x not in stopwords])
            """创建tf-idf模型   传入语料库训练"""
            tfidf = TfidfModel(corpus)
            """训练好的tf-idf模型处理检索文本和搜索词"""
            tf_texts = tfidf[corpus]
            tf_kw = tfidf[kw_vector]
            """相似度计算"""
            sparse_matrix = SparseMatrixSimilarity(tf_texts, feature_cnt)
            similarities = sparse_matrix.get_similarities(tf_kw)
            # print("similarities")
            # print(similarities)
            # for e, s in enumerate(similarities, 1):
            #     print('kw 与 text%d 相似度为：%.2f' % (e, s))
            conceptualSimilarity.append(max(similarities))

        """key word ratio"""
        keywordsInComments = [x for x in word_list if x in languageKeyWords]
        stopKeyRatio.append(keywordsInComments.__len__() / word_list.__len__())


    print(readable)
    print(max(readable), min(readable))

    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(readable)
    s.plot(kind='kde')
    plt.title('readable')
    plt.show()

    # for case in badCase:
    #     print(case[1], case[0])
    print(badCase.__len__() / comments.shape[0])

    print(stopWordRate)
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(stopWordRate)
    s.plot(kind='kde')
    plt.title('stopwordRate')
    plt.show()

    print(questionRatio)
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(questionRatio)
    s.plot(kind='kde')
    plt.title('questionRatio')
    plt.show()

    print(codeElementRatio)
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(codeElementRatio)
    plt.title('codeElementRatio')
    s.plot(kind='kde')
    plt.show()

    print(conceptualSimilarity)
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(conceptualSimilarity)
    plt.title('conceptualSimilarity')
    s.plot(kind='kde')
    plt.show()

    print(stopKeyRatio)
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    s = Series(stopKeyRatio)
    plt.title('stopKeyRatio')
    s.plot(kind='kde')
    plt.show()
