"""

    User table ``user``

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

    Item table ``item``:

    ===========  ============  ========  
    ``item_id``  ``Category``  ``...`` 
    ===========  ============  ========  
    1            True           True            
    2            False          False           
    ===========  ============  ========

    Play relationship table ``rating``:

    ===========  ===========  =========
    ``user_id``  ``item_id``  ``ratings``
    ===========  ===========  =========
    1            1            5
    2            1            2
    3            2            1
    4            2            2
    ===========  ===========  =========

"""

import os, sys
import re
import argparse

import json
import gzip
import pickle
import joblib

import numpy as np
import pandas as pd

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

    # ======================================================================
    #   USER
    # ======================================================================
    raw_df = getDF(os.path.join(directory, 'AMAZON_FASHION.json.gz'))

    # 리뷰 단어 count
    def WordCount(x):
        try:
            return len(x.split())
        except:
            return 0

    def prep_user(df):

        '''
        user_id : user ID
        meanRating : 평균 평점
        ReviewCount : 리뷰 개수

        meanReviewLength : 평균 리뷰 길이
        meanSummaryLength : 평균 Summary 길이
        meanReviewWord : 평균 리뷰 단어 개수
        meanSummaryWord : 평균 Summary 단어 개수
        '''

        # preprocess
        df["reviewTextLength"] = df["reviewText"].apply(lambda x: len(str(x)))
        df["summaryLength"] = df["summary"].apply(lambda x: len(str(x)))
        df["reviewTextCount"] = df["reviewText"].apply(lambda x: WordCount(x))
        df["summaryCount"] = df["summary"].apply(lambda x: WordCount(x))

        # user aggregation
        user = df.groupby('reviewerID').agg({
            'overall': [('meanRating', np.mean)],
            'reviewTextLength': [('meanReviewLength', np.mean)],
            'summaryLength': [('meanSummaryLength', np.mean)],
            'reviewTextCount': [('meanReviewWord', np.mean), ('ReviewCount', 'count')],
            'summaryCount': [('meanSummaryWord', np.mean)],
        }).reset_index()
        user.columns = user.columns.get_level_values(level=1)
        user.columns = ["user_id", 'meanRating', 'meanReviewLength', 'meanSummaryLength', 'meanReviewWord',
                        'ReviewCount', 'meanSummaryWord']

        user = user[["user_id", 'meanRating', 'ReviewCount', 'meanReviewLength', 'meanSummaryLength', 'meanReviewWord', 'meanSummaryWord']]
        return user
    
    # USER_LIST
    users = prep_user(raw_df)
    user2idx = {k : v for v, k in enumerate(users['user_id'].unique())}
    users['user_id'] = users['user_id'].map(user2idx)


    # ======================================================================
    #   ITEM
    # ======================================================================
    raw_item = getDF(os.path.join(directory, 'meta_AMAZON_FASHION.json.gz'))

    # Preprocess Category
    def prep_category(x):
        
        # list value 처리
        if type(x) == list:
            x = x[0]
        
        # 숫자 삭제
        x = re.sub('[0-9(#]+', '', x)
        
        # 특수문자 삭제
        x = re.sub('[-=+,#/\?:^$.@*\"※~%ㆍ!』\;\‘|\(\)\[\]\<\>`\'…》]', '', x)
        
        # 처음 ,, 삭제
        if len(x) > 1:
            while x[0] == ',':
                x = x[1:]
                
        # 처음 in 삭제
        if len(x) > 1:
            if x[:2] == 'in':
                x = x[2:]
        
        # 대문자를 기준으로 카테고리 나누기
        result = ''
        for elem in x:
            
            # 공백 무조건 지우기
            if elem == ' ' or elem == '':
                continue
            
            # &가 있으면 앞뒤 같은 카테고리로
            try:
                if result[-1] == '&':
                    result += elem
                    continue
            except:
                pass
            
            # 대문자 발견되면 , 로 카테고리 구분하기
            if ord(elem) < 97 and ord(elem) >= 65:
                result += ','
                result += elem
            else:
                result += elem
        
        # in 이라는 단어 삭제
        category = ''
        for word in result.split(','):
            if word.strip() == 'in':
                continue
            word = ',' + word.strip()
            category += word
        
        # 처음 , 삭제
        if len(category) > 1:
            while category[0] == ',':
                category = category[1:]
        
        # 딕셔너리로 반환
        category_dict = {}
        for c in category.split(','):
            category_dict[c] = True
        
        return category_dict
    

    # ITEM_LIST
    items = (raw_item[['asin', 'rank']].drop_duplicates('asin')
                                      .rename(columns = {'asin' : 'item_id'})
                                      ) # 중복 제거
    item2idx = {k : v for v, k in enumerate(items['item_id'].unique())}
    items['item_id'] = items['item_id'].map(item2idx) 

    items['rank'] = items['rank'].fillna('null').astype(str).apply(lambda x : prep_category(x))
    
    for i, arg in enumerate(items.values):
        items.iloc[i, 1]['item_id'] = arg[0]

    items = pd.DataFrame(items['rank'].to_list())
    idx = items[['item_id']]
    items = pd.concat([idx, items.drop('item_id', axis = 1)], axis = 1)


    # ======================================================================
    #   RATING
    # ======================================================================
    def prep_rating(df, user2idx, item2idx):
        '''
        df (user-item 행렬), user2idx & item2idx -> encoding rating matrix 
        '''
        rating = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        rating['reviewerID'] = rating['reviewerID'].map(user2idx)
        rating['asin'] = rating['asin'].map(item2idx)
        rating.columns = ['user_id', 'item_id', 'rating', 'timestamp']

        return rating
    
    ratings = prep_rating(raw_df, user2idx, item2idx)
    
    # ======================================================================
    #   Build heterogeneous graph
    # ======================================================================
   
    # 평가를 한, 평점을 받은 유저와 아이템만 선별
    distinct_users_in_ratings = ratings['user_id'].unique()
    distinct_items_in_ratings = ratings['item_id'].unique()
    users = users[users['user_id'].isin(distinct_users_in_ratings)]
    items = items[items['item_id'].isin(distinct_items_in_ratings)]

    # Group the movie features into genres (a vector), year (a category), title (a string)
    category_columns = items.columns.drop('item_id')
    items[category_columns] = items[category_columns].fillna(False).astype('bool')
    # movies_categorical = items.drop('title', axis=1)


    # Graph Build
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(items, 'item_id', 'item')
    graph_builder.add_binary_relations(ratings, 'user_id', 'item_id', 'purchased')
    graph_builder.add_binary_relations(ratings, 'item_id', 'user_id', 'purchased-by')

    g = graph_builder.build()

    # Assign features -> feature 수정중 0529
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.nodes['user'].data['meanRating'] = torch.LongTensor(users['meanRating'].values)    # .cat.codes.values
    g.nodes['user'].data['ReviewCount'] = torch.LongTensor(users['ReviewCount'].values)
    g.nodes['user'].data['meanSummaryLength'] = torch.LongTensor(users['meanSummaryLength'].values)
    g.nodes['user'].data['meanReviewWord'] = torch.LongTensor(users['meanReviewWord'].values)
    g.nodes['user'].data['meanSummaryWord'] = torch.LongTensor(users['meanSummaryWord'].values)

    # g.nodes['item'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    g.nodes['item'].data['cat'] = torch.FloatTensor(items[category_columns].values)

    g.edges['purchased'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['purchased'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['purchased-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['purchased-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    
    # ======================================================================
    #   Train-validation-test split -> 수정아직 안함
    # ======================================================================
    # rating의 timestamp 기준, 각 유저의 마지막 평점을 test로 / 마지막 바로 전 평점을 valid로
    # from data_utils.py
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    assert train_g.out_degrees(etype='watched').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

        ## Build title set

        # movie_textual_dataset = {'title': movies['title'].values}

        # The model should build their own vocabulary and process the texts.  Here is one example
        # of using torchtext to pad and numericalize a batch of strings.
        #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
        #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
        #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
        #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
        #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

        ## Dump the graph and the datasets

        # dataset = {
        #     'train-graph': train_g,
        #     'val-matrix': val_matrix,
        #     'test-matrix': test_matrix,
        #     'item-texts': movie_textual_dataset,
        #     'item-images': None,
        #     'user-type': 'user',
        #     'item-type': 'movie',
        #     'user-to-item-type': 'watched',
        #     'item-to-user-type': 'watched-by',
        #     'timestamp-edge-column': 'timestamp'}

    #     with open(output_path, 'wb') as f:
    #         pickle.dump(dataset, f)

