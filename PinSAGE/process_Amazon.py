"""

    User table ``users``

    * reviewerID : userID
    * meanRating : 평균 평점
    * ReviewCount : 리뷰 개수

    * meanReviewLength : 평균 리뷰 길이
    * meanSummaryLength : 평균 Summary 길이
    * meanReviewWord : 평균 리뷰 단어 개수
    * meanSummaryWord : 평균 Summary 단어 개수

    =========== ============== =============== ==================== ==================== =================== ===================
    ``user_ID`` ``meanRating`` ``ReviewCount`` ``meanReviewLength`` ``meanSummaryLength`` ``meanReviewWord`` ``meanSummaryWord``
    =========== ============== =============== ==================== ==================== =================== ===================
         0           5.0              1                95.0                 16.0                 21.0                3.0
        ...
       749232        5.0              1                4.0                  4.0                  1.0                 1.0
    =========== ============== =============== ==================== ==================== =================== ===================

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========


"""

import os, sys
import re
import argparse

import json
import gzip
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    # parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    # output_path = args.output_path

    # Data Load
    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.loads(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    # USER
    raw_user = getDF(os.path.join(directory, 'AMAZON_FASHION.json.gz'))

    # user matrix
    def WordCount(x):
        try:
            return len(x.split())
        except:
            return 0

    def user_dataframe(df):

        '''
        user_ID : user ID
        meanRating : 평균 평점
        ReviewCount : 리뷰 개수

        meanReviewLength : 평균 리뷰 길이
        meanSummaryLength : 평균 Summary 길이
        meanReviewWord : 평균 리뷰 단어 개수
        meanSummaryWord : 평균 Summary 단어 개수
        '''

        le = LabelEncoder()
        df["reviewerID"] = le.fit_transform(df["reviewerID"])

        # preprocess
        df["reviewTextLength"] = df["reviewText"].apply(lambda x: len(str(x)))
        df["summaryLength"] = df["summary"].apply(lambda x: len(str(x)))
        df["reviewTextCount"] = df["reviewText"].apply(lambda x: WordCount(x))
        df["summaryCount"] = df["summary"].apply(lambda x: WordCount(x))

        # user dataframe
        user = df.groupby('reviewerID').agg({
            'overall': [('meanRating', np.mean)],
            'reviewTextLength': [('meanReviewLength', np.mean)],
            'summaryLength': [('meanSummaryLength', np.mean)],
            'reviewTextCount': [('meanReviewWord', np.mean), ('ReviewCount', 'count')],
            'summaryCount': [('meanSummaryWord', np.mean)],
        }).reset_index()
        user.columns = user.columns.get_level_values(level=1)
        user.columns = ["user_ID", 'meanRating', 'meanReviewLength', 'meanSummaryLength', 'meanReviewWord',
                        'ReviewCount', 'meanSummaryWord']

        user = user[["user_ID", 'meanRating', 'ReviewCount', 'meanReviewLength', 'meanSummaryLength', 'meanReviewWord', 'meanSummaryWord']]
        return user

    user = user_dataframe(raw_user)
    print(user)
    # ITEM
    raw_item = getDF(os.path.join(directory, 'meta_AMAZON_FASHION.json.gz'))

    # ITEM_LIST
    item = raw_item['asin'].drop_duplicates().reset_index(drop = True).reset_index()
    
    # Preprocess Category
    def prepCategory(x):
    
        if type(x) == list:
            x = x[0]
        
        # 숫자 삭제
        x = re.sub('[0-9(#]+', '', x)
        
        # split 기호 변환
        x = re.sub('[-=+,#/\?:^$.@*\"※~%ㆍ!』\;\‘|\(\)\[\]\<\>`\'…》]', '', x)
        
        # 처음 ,, 삭제
        if len(x) > 1:
            while x[0] == ',':
                x = x[1:]
                
        # 처음 in 삭제
        if len(x) > 1:
            if x[:2] == 'in':
                x = x[2:]
        
        result = ''
        for elem in x:
            
            # 공백 무조건 지우기
            if elem == ' ' or elem == '':
                continue
                
            try:
                if result[-1] == '&':
                    result += elem
                    continue
            except:
                pass
            
            if ord(elem) < 97 and ord(elem) >= 65:
                result += ','
                result += elem
            else:
                result += elem
        
        category = ''
        for word in result.split(','):
            if word == 'in':
                continue
            word = ',' + word
            category += word
        
        if len(category) > 1:
            while category[0] == ',':
                category = category[1:]
        
        category_dict = {}
        for c in category.split(','):
            category_dict[c] = True
        
        return category_dict
    
    item['rank'] = raw_item['rank'].fillna('null').astype(str).apply(lambda x : prepCategory(x))
    
    for i, arg in enumerate(item.values):
        item.iloc[i, 2]['item_id'] = arg[0]

    item = pd.DataFrame(item['rank'].to_list())
    idx = item[['item_id']]
    item = item.drop('item_id', axis = 1)
    item = pd.concat([idx, item], axis = 1)
    print(item)

    sys.exit(1)
    ## Build heterogeneous graph




    # Graph Build
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user, 'user_id', 'user')
    graph_builder.add_entities(item, 'item_id', 'item')
    graph_builder.add_binary_relations(rating, 'user_id', 'item_id', 'buy')
    graph_builder.add_binary_relations(rating, 'item_id', 'user_id', 'purchased-by')

    g = graph_builder.build()


    # Assign

