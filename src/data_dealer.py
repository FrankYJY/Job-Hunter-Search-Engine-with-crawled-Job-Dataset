import csv
import os
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
import pickle

def tokenize(text):
    '''tokenize chinese english number'''
    lowers = text.lower()
    lowers = lowers.replace("-"," ")
    no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
    '''
    精确模式	jieba.lcut(s)	返回一个列表类型的分词结果，不存在冗余
    Precise Mode  jieba.lcut (s)  Returns a list-type participle result with no edundancy
    全模式	jieba.lcut(s,cut_all=True)	返回一个列表类型的分词结果，存在冗余
    Full mode  jieba.lcut (s, cut_all =True)  Returns a list type word segmentation result with redundancy
    搜索引擎模式	jieba.lcut_for_search(s)	返回一个列表类型的分词结果，存在冗余，会对较长的单词再次细分
    Search engine mode  jieba.lcut_for_search (s)  Returns a list-type word segmentation result with redundancy that subdivides longer words
    '''
    seg_list = jieba.cut(no_punctuation, cut_all=False)#chinese
    tokens = nltk.word_tokenize(" ".join(seg_list))
    # count = Counter(tokens)
    # print('no_punctuation: '+str(no_punctuation))
    # print('tokens: '+str(tokens))
    # print("document : tokens: %d, types: %d" % (len(tokens), len(count)))
    return tokens#,count

from nltk.corpus import stopwords

def remove_stopwords(tokens, stopwords):
    filtered = [w for w in tokens if not w in stopwords]
    # count_filtered = Counter(filtered)
    # print("filtered:"+str(filtered))
    # print("document 1 : tokens: %d, types: %d" % (len(filtered), len(count_filtered)))
    return filtered

def tf(term_frequency_matrix):
    tf = []
    # record_max_value=[]
    for i in range(len(term_frequency_matrix)):
        dict = term_frequency_matrix[i]
        # print(dict)
        max_value = max(dict.values())
        # record_max_value.append(max_value)
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

def idf(term_count_matrix, whole_count):
    number_of_docs= len(term_count_matrix)
    # print("number_of_docs: "+str(number_of_docs))
    idf = whole_count.copy()
    for w in idf.keys():
        number_of_docs_have_this_word = 0
        for i in range(number_of_docs):
            if term_count_matrix[i][w]>0:
                number_of_docs_have_this_word += 1
        idf[w]=math.log((number_of_docs/number_of_docs_have_this_word),2)
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
    '''can generate cosine similarity on every pair of documents, we do not use here'''
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

def cos_similarity(TFIDF, weighted_query):
    similarity =  [([0] * 2)for i in range(len(TFIDF))]
    # print(np.array(similarity).shape)
    denominatorpart2 = 0
    # print(weight_matrix[-2])
    for i in weighted_query:
        denominatorpart2 += weighted_query[i] ** 2
    for i in range(len(TFIDF)):
        numerator = 0
        denominatorpart1 = 0
        for wi in TFIDF[i]:
            if wi in weighted_query:
                numerator += TFIDF[i][wi] * weighted_query[wi]
            denominatorpart1 += TFIDF[i][wi] ** 2
        similarity[i][0] = i+1 #this document index, start from 1
        similarity[i][1] =  numerator/((denominatorpart1**0.5)*(denominatorpart2**0.5)) #similarity score
        #[[1,x1],[2,x2]...]
    return similarity

def quicksort(li, start, end):
    '''standard quick sort, for reference'''
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
    # Divide and divide into two
    # start=end ,证明要处理的数据只有一个demonstrate that there is only one data to be processed
    # start>end ,证明右边没有数据No data on the right
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    # Defines two cursors, pointing to positions 0 and end, respectively
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    # takes the 0 position data as an intermediate value
    mid = list[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        # Move the right cursor to the left in order to find a value less than mid and place
        while left < right and list[right][1] <= mid[1]:
            right -= 1
        list[left] = list[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        # Move the left cursor to the right to find a value greater than the mid and place it in right cursor position
        while left < right and list[left][1] > mid[1]:
            left += 1
        list[right] = list[left]
    # while结束后，把mid放到中间位置，left=right
    # Put the mid in the middle after the while
    list[left] = mid
    # 递归处理左边的数据
    # Recursive processing of data on the left
    special_quicksort(list, start, left - 1)
    # 递归处理右边的数据
    # Recursive processing of data on the right
    special_quicksort(list, left + 1, end)

def stem_tokens(tokens, stemmer):
    '''can only stem english words if there are any'''
    '''if we need chinese synonym mergence, we can refer to HIT IR-Lab Tongyici Cilin (Extended)'''
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def calculate_text_list_TFIDF(raw_text_list, IDF=None):
    print('calculating TFIDF...')
    text_list = copy.deepcopy(raw_text_list)
    '''text_list:   [text0,text1,...textn]'''

    '''here we take whole as one text to count later, another way to count whole is sum counts of every record in the end'''
    whole_text = ' '.join(text_list)

    '''our original solution was to add together and manipulate'''
    # text_list.append(query)
    # text_list.append(temp)
    # print("after append all"+str(len(text_list)))
    # '''text_list:   [text0,text1,...textn,query,all]'''

    term_count_matrix=[]
    stopwords1 = [line.strip() for line in open('hit_stopwords.txt', encoding='UTF-8').readlines()]  # chinese
    stopwords2 = stopwords.words('english')  # if contain english stopword, add it into parameters of remove_stopwords
    stopwords_all = stopwords1+stopwords2#remove all chinese and english stopwords
    stemmer = PorterStemmer()

    def count_one_text(text,stopwords,stemmer):
        '''Step1: tokenize'''
        tokens = tokenize(text)
        # print(tokens)
        # token_list.append(tokenize(text_list[i]))
        '''Step2: remove stopwords'''
        filtered = remove_stopwords(tokens, stopwords)
        '''Step3: stem'''
        stemmed = stem_tokens(filtered, stemmer)

        term_count = Counter(stemmed)
        # print("document %d : terms: %d" % ((i + 1), len(term)))
        return  term_count

    for i in range(len(text_list)):
        term_count = count_one_text(text_list[i],stopwords_all,stemmer)
        term_count_matrix.append(term_count)
    whole_count = count_one_text(whole_text,stopwords_all,stemmer)
    # for i in term_count_matrix:
    #     print(i)
    # print("")
    '''Step4: TF-IDF'''
    TF = tf(term_count_matrix)
    # for i in TF:
    #     print(i)
    if IDF==None:
        IDF = idf(term_count_matrix,whole_count)
    # print(whole_count)
    # print(TF)
    # print(IDF)
    TFIDF = tfidf(TF, IDF)
    # print("\n")
    # for i in TFIDF:
    #     print(i)
    # print("\n")
    '''tfidf only calculate once for database'''
    '''we need idf when calculate query in an other method when running'''
    return TFIDF,IDF,whole_count

def standard_rocchio_calculate(rel_irrel_list, record, whole_count, a=1, b=0.75, y=0.25):
    #record only use its 0 row which indicates T/F
    result = Counter(whole_count)#deep copy, avoid modification on raw data
    # print(result)
    # contain = [x for x in query]
    # print(contain)
    for key in result:
        # denominator = len(TFIDF2[0])
        # print(key,record[0])
        if key in record[0]:
            # print(key)
            result[key] = record[0][key] * a
        else:
            result[key] = 0
        # if key == '阿里巴巴':
        #     print(key, result[key])
    # print('rocchio1',result)
    for i in range(1,len(record)):
        for key in result:
            # print(key, record[i])
            if key in record[i]:
                # print('ya',record[i][0])
                if rel_irrel_list[i]==True:
                    # print(result[key])
                    result[key] += record[i][key] * b
                else:
                    result[key] -= record[i][key] * y
                # if key == '阿里巴巴':
                #     print(key,result[key],TFIDF2[i])
        # print('rocchion',result)
    # print(result)
    return result


def find_similarity(database_TFIDF, database_IDF,record_list):
    '''record list is [[T/F,text1],[T/F,Text2]...]'''
    query_text_list = [i[1] for i in record_list]

    query_TFIDF, IDF, query_whole_count = calculate_text_list_TFIDF(query_text_list, database_IDF)
    # print(query_TFIDF)
    # print(query_whole_count)

    rel_irrel_list = [i[0] for i in record_list]
    weighted_query = standard_rocchio_calculate(rel_irrel_list, query_TFIDF, query_whole_count)
    # print(weighted_query)

    #     query = [True, query]

    ranked_similarity = cos_similarity(database_TFIDF, weighted_query)
    sys.setrecursionlimit(20000)  # we have to expand recursion limit for quick sort
    special_quicksort(ranked_similarity, 0, len(ranked_similarity) - 1)

    # print(ranked_similarity)
    return ranked_similarity

'''
MAIN
'''
def example():
    database_name = 'lagou_ITjobs'
    csv_file_name = database_name+'.csv'
    database_records = read_file(csv_file_name)

    text_list =[]
    for i in range(1,database_records.__len__()):
        text_list.append(' '.join(database_records[i]))
        # text = ' '.join(columns[i])
        # print(text)

    # print("\n")
    # for i in text_list:
    #     print(i)
    # print("\n")

    filename = database_name+'_TFIDF_storage.pickle'
    # is_first_time = False
    is_first_time = not os.path.exists(filename)
    if is_first_time:
        print('is the first time')
        database_TFIDF, database_IDF = calculate_text_list_TFIDF(text_list)
        print('store calculated TFIDF')
        with open(filename, 'wb') as f:
            pickle.dump(database_TFIDF, f)
            pickle.dump(database_IDF, f)
    with open(filename, 'rb')as f:
        print('load calculated TFIDF')
        database_TFIDF=pickle.load(f)
        database_IDF=pickle.load(f)
        # print("")
        # for i in TFIDF_matrix:
        #     print(i)
        # print("")
        # print(IDF)

    query = 'java杭州阿里巴巴'
    use_record = False
    if use_record:
        record_list = [[True, query], [True, 'python java 苏州']]
    else:
        record_list = [[True, query]]
    ranked_similarity = find_similarity(database_TFIDF,database_IDF, record_list)

    # show:
    # for i in range(10):
    #     #text_list 去头取序号需要减一，columns不去头，直接取从1开始的rank
    #     print('index:',ranked_similarity[i][0],'score:',ranked_similarity[i][1],'text:', database_records[ranked_similarity[i][0]])

# example()