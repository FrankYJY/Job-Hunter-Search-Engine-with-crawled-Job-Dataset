import csv
import sys

import nltk
import string
from collections import Counter
import jieba
import re
import math
from nltk.stem.porter import *
import numpy as np
import copy

def tokenize(text):
    '''tokenize chinese english number'''
    lowers = text.lower()
    lowers = lowers.replace("-"," ")
    no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
    seg_list = jieba.cut(no_punctuation, cut_all=False)
    tokens = nltk.word_tokenize(" ".join(seg_list))
    # count = Counter(tokens)
    # print('no_punctuation: '+str(no_punctuation))
    # print('tokens: '+str(tokens))
    # print("document : tokens: %d, types: %d" % (len(tokens), len(count)))
    return tokens#,count

from nltk.corpus import stopwords

def remove_stopwords(tokens,stopwords1, stopwords2=None):
    filtered = [w for w in tokens if not w in stopwords1]
    if stopwords2!=None:
        filtered = [w for w in filtered if not w in stopwords2]
    count_filtered = Counter(filtered)
    # print("filtered:"+str(filtered))
    # print("document 1 : tokens: %d, types: %d" % (len(filtered), len(count_filtered)))
    return filtered, count_filtered

def tf(term):
    tf = []
    record_max_value=[]
    for i in range(len(term)):
        dict = term[i]
        # print(dict)
        max_value = max(dict.values())
        record_max_value.append(max_value)
        temptf = dict.copy()
        for i in temptf.keys():
            temptf[i] /=max_value
        # print(temptf)
        tf.append(temptf)
    # print('')
    # dict = term[-2]
    # # print(dict)
    # max_value = max(dict.values())
    # temptf = dict.copy()
    # for i in temptf.keys():
    #     temptf[i] /= sum(record_max_value)/len(record_max_value)
    # print(temptf)
    # tf.append(temptf)
    #
    # dict = term[-1]
    # # print(dict)
    # max_value = max(dict.values())
    # temptf = dict.copy()
    # for i in temptf.keys():
    #     temptf[i] /= max_value
    # print(temptf)
    # tf.append(temptf)
    return tf

def idf(term):
    number_of_docs=term.__len__()-1
    # print("number_of_docs: "+str(number_of_docs))
    idf = term[-1].copy()
    for w in term[number_of_docs].keys():
        count_word_in_index = 0
        for i in range(number_of_docs):
            if term[i][w]>0:
                count_word_in_index += 1
        idf[w]=math.log((number_of_docs/count_word_in_index),2)
    return idf

def tfidf(tf,idf):
    tfidf = tf.copy()
    for i in range(len(tfidf)):
        for w in tfidf[i].keys():
            tfidf[i][w] *= idf[w]
    return tfidf

def read_file(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # print(type(reader))
        # for row in reader:
        #     print(row)
        # columns = list(reader)
        # for i in range(1,columns.__len__()):
        #     # print(columns[i])# show and check
        #     print(columns[i][3])
        return list(reader)

def all_cos_similarity(term):
    number_of_docs=term.__len__()-1
    similarity =  [([0] * number_of_docs) for i in range(number_of_docs)]
    for i in range(number_of_docs):
        for j in range(number_of_docs):
            numerator = 0
            denominatorpart1 = 0
            denominatorpart2 = 0
            for wi in term[i]:
                if wi in term[j]:
                    numerator += term[i][wi] * term[j][wi]
                denominatorpart1 += term[i][wi]**2
            for wj in term[j]:
                denominatorpart2 += term[j][wj]**2
            similarity[i][j] =  numerator/((denominatorpart1**0.5)*(denominatorpart2**0.5))
    return similarity

def cos_similarity(weight_matrix, weighted_query=None):
    number_of_docs= weight_matrix.__len__() - 1#include query
    similarity =  [([0] * 2)for i in range(number_of_docs-1)]#exclude query
    # print(np.array(similarity).shape)
    denominatorpart2 = 0
    if weighted_query == None:
        real_query = weight_matrix[-2]
    else:
        real_query = weighted_query
    # print(weight_matrix[-2])
    for i in weight_matrix[-2]:
        denominatorpart2 += real_query[i] ** 2
    for i in range(number_of_docs-1):
        numerator = 0
        denominatorpart1 = 0
        for wi in weight_matrix[i]:
            if wi in real_query:
                numerator += weight_matrix[i][wi] * real_query[wi]
            denominatorpart1 += weight_matrix[i][wi] ** 2
        similarity[i][0] = i+1
        similarity[i][1] =  numerator/((denominatorpart1**0.5)*(denominatorpart2**0.5))
        #[[1,x1],[2,x2]...]
    return similarity

def quicksort(li, start, end):
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = li[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and li[right] >= mid:
            right -= 1
        li[left] = li[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and li[left] < mid:
            left += 1
        li[right] = li[left]
    # while结束后，把mid放到中间位置，left=right
    li[left] = mid
    # 递归处理左边的数据
    quicksort(li, start, left - 1)
    # 递归处理右边的数据
    quicksort(li, left + 1, end)

def special_quicksort(list, start, end):
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = list[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and list[right][1] <= mid[1]:
            right -= 1
        list[left] = list[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and list[left][1] > mid[1]:
            left += 1
        list[right] = list[left]
    # while结束后，把mid放到中间位置，left=right
    list[left] = mid
    # 递归处理左边的数据
    special_quicksort(list, start, left - 1)
    # 递归处理右边的数据
    special_quicksort(list, left + 1, end)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def find_similarity(raw_text_list,query,record=None):
    text_list = copy.deepcopy(raw_text_list)
    text_list.append(query)
    temp = ' '.join(text_list)
    text_list.append(temp)
    # print("after append all"+str(len(text_list)))
    '''text_list:   [text0,text1,...textn,query,all]'''

    term_list=[]
    stopwords1 = [line.strip() for line in open('../hit_stopwords.txt', encoding='UTF-8').readlines()]  # chinese
    stopwords2 = stopwords.words('english')  # if contain english stopword, add it into parameters of remove_stopwords
    stopwords_all = stopwords1+stopwords2#remove all chinese and english stopwords
    stemmer = PorterStemmer()
    for i in range(text_list.__len__()):
        '''Step1: tokenize'''
        tokens = tokenize(text_list[i])
        # token_list.append(tokenize(text_list[i]))
        '''Step2: remove stopwords'''
        # stopwords2 = stopwords.words('english')# if contain english stopword, add it into parameters of remove_stopwords
        filtered, count_filtered = remove_stopwords(tokens, stopwords1)
        '''Step3: stem'''
        '''can only stem english words if there are any'''
        stemmed = stem_tokens(filtered,stemmer)
        '''Step4: TF-IDF'''
        term = Counter(stemmed)
        # print("document %d : terms: %d" % ((i + 1), len(term)))
        term_list.append(term)
    # for i in term_list:
    #     print(i)
    # print("\n")
    '''TF-IDF'''
    TF = tf(term_list)
    # for i in TF:
    #     print(i)
    IDF = idf(term_list)
    # print(IDF)
    TFIDF = tfidf(TF, IDF)
    # print("\n")
    # for i in TFIDF:
    #     print(i)
    # print("\n")
    '''Step5: cosine similarity'''
    # for i in term_list:
    #     print(i)
    if record!=None:
        record_list =[]
        '''record_list: [CounterQ,Counter1,Counter2,Counterall]'''
        temp = ''
        for i in range(record.__len__()):
            temp += record[i][1]+" "
        record.append([True,temp])
        '''record: [[T,query],[T/F,doc1],[T/F,doc2],[True,all]]'''
        for i in range(record.__len__()):
            '''Step1: tokenize'''
            tokens = tokenize(record[i][1])
            # token_list.append(tokenize(text_list[i]))
            '''Step2: remove stopwords'''
            filtered, count_filtered = remove_stopwords(tokens, stopwords_all)
            '''Step3: stem'''
            '''can only stem english words if there are any'''
            stemmed = stem_tokens(filtered, stemmer)
            '''Step4: TF-IDF'''
            term = Counter(stemmed)
            # print("document %d : terms: %d" % ((i + 1), len(term)))
            record_list.append(term)
        '''TF-IDF'''
        TF2 = tf(record_list)
        # print(TF2)
        # print(IDF)
        #skip IDF, calculated
        TFIDF2 = tfidf(TF2, IDF)
        # for i in TFIDF2:
        #     print(i)
        new_query = standard_rocchio_calculate(record, TFIDF2)
        # print(query)
        print(new_query)
        similarity = cos_similarity(TFIDF,new_query)
    else:
        similarity = cos_similarity(TFIDF)
    # print(similarity)
    sys.setrecursionlimit(20000)
    special_quicksort(similarity,0,similarity.__len__()-1)
    # print(similarity)
    # print([x[0] for x in similarity])
    #rank starts from 1
    #rank = [x[0] for x in similarity]
    return similarity

def standard_rocchio_calculate(record, TFIDF2, a=1, b=0.75, y=0.25):
    #record only use its 0 row which indicates T/F
    result = Counter(TFIDF2[-1])#deep copy, avoid modification on raw data
    # print(result)
    # contain = [x for x in query]
    # print(contain)
    for key in result:
        # denominator = len(TFIDF2[0])
        if key in TFIDF2[0]:
            # print(key)
            result[key] = TFIDF2[0][key]*a
        else:
            result[key] = 0
        # if key == '阿里巴巴':
        #     print(key, result[key])
    # print(len(record)-1)
    # print('rocchio1', result)
    for i in range(1,len(record)-1):
        for key in result:
            if key in TFIDF2[i]:
                # print(record[i][0])
                if record[i][0]==True:
                    # print(result[key])
                    result[key] += TFIDF2[i][key] * b
                else:
                    result[key] -= TFIDF2[i][key] * y
                # if key == '阿里巴巴':
                #     print(key,result[key],TFIDF2[i])
        # print('rocchion', result)
    # print(result)
    return result

'''
MAIN
'''
def run():
    doc_path = '../lagou_ITjobs.csv'
    query = 'java杭州'
    columns = read_file(doc_path)
    # for i in columns:
    #     print(type(i))
    # print("records(include description column):"+str(columns.__len__()))
    text_list =[]
    for i in range(1,columns.__len__()):
        text_list.append(' '.join(columns[i]))
        # text = ' '.join(columns[i])
        # print(text)
    record_list = [[True, query],[True, 'python java 苏州']]
    use_record = True
    if use_record:
        ranked_similarity = find_similarity(text_list, query, record_list)
    else:
        ranked_similarity = find_similarity(text_list, query)
    # show:
    for i in range(10):
        #text_list 去头取序号需要减一，columns不去头，直接取从1开始的rank
        print('index:',ranked_similarity[i][0],'score:',ranked_similarity[i][1],'text:', columns[ranked_similarity[i][0]])

run()