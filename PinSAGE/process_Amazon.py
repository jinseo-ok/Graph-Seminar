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
import gzip
import pickle
import pandas as pd
import numpy as np
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
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

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
    raw_user = getDF(os.path.join(directory, 'AMAZON_FASHION.json.gz'))

    # user matrix
    def WordCount(x):
        try:
            return len(x.split())
        except:
            return 0

    le = LabelEncoder()  # label encoder 정의
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
    # print(user)

    # ======================================================================
    #   ITEM
    # ======================================================================
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
    item2idx = {k:v for k, v in item[['item_id', 'asin']].values}    # item2idx 추가 -> rating에 사용
    
    for i, arg in enumerate(item.values):
        item.iloc[i, 2]['item_id'] = arg[0]

    item = pd.DataFrame(item['rank'].to_list())
    idx = item[['item_id']]
    item = item.drop('item_id', axis = 1)
    item = pd.concat([idx, item], axis = 1)
    # print(item)

    sys.exit(1)

    # ======================================================================
    #   RATING
    # ======================================================================
    def rating_preb(raw_data,item2idx):
        '''
        raw_data, item2idx 받아서 rating 테이블 만든다.
        '''
        rating = raw_data[['reviewerID','asin','overall','unixReviewTime']]
        rating['user_id'] = le.transform(rating['reviewerID'])    # user2idx도 가능
        rating['asin'].map(item2idx)
        rating.columns = ['user_id','item_id','rating','timestamp']

        return rating
    
    rating = rating_preb(raw_data,item2idx)


    # ======================================================================
    #   Build heterogeneous graph
    # ======================================================================
   
    # Graph Build
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user, 'user_id', 'user')
    graph_builder.add_entities(item, 'item_id', 'item')
    graph_builder.add_binary_relations(rating, 'user_id', 'item_id', 'purchased')
    graph_builder.add_binary_relations(rating, 'item_id', 'user_id', 'purchased-by')

    g = graph_builder.build()

    # item features
    # Group the movie features into genres (a vector), year (a category), title (a string)
    cat_columns = item.columns.drop(['item_id'])
    item[cat_columns] = item[cat_columns].fillna(False).astype('bool')
    # item_categorical = item.drop('title', axis=1)

    # Assign features -> feature 수정중 0529
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.nodes['user'].data['meanRating'] = torch.LongTensor(user['meanRating'].values)    # .cat.codes.values
    g.nodes['user'].data['ReviewCount'] = torch.LongTensor(user['ReviewCount'].values)
    g.nodes['user'].data['meanSummaryLength'] = torch.LongTensor(user['meanSummaryLength'].values)
    g.nodes['user'].data['meanReviewWord'] = torch.LongTensor(user['meanReviewWord'].values)
    g.nodes['user'].data['meanSummaryWord'] = torch.LongTensor(user['meanSummaryWord'].values)

    # g.nodes['item'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    g.nodes['item'].data['cat'] = torch.FloatTensor(item[cat_columns].values)

    g.edges['purchased'].data['rating'] = torch.LongTensor(rating['rating'].values)
    g.edges['purchased'].data['timestamp'] = torch.LongTensor(rating['timestamp'].values)
    g.edges['purchased-by'].data['rating'] = torch.LongTensor(rating['rating'].values)
    g.edges['purchased-by'].data['timestamp'] = torch.LongTensor(rating['timestamp'].values)

    
    # ======================================================================
    #   Train-validation-test split -> 수정아직 안함
    # ======================================================================
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    assert train_g.out_degrees(etype='watched').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

        ## Build title set

        movie_textual_dataset = {'title': movies['title'].values}

        # The model should build their own vocabulary and process the texts.  Here is one example
        # of using torchtext to pad and numericalize a batch of strings.
        #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
        #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
        #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
        #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
        #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

        ## Dump the graph and the datasets

        dataset = {
            'train-graph': train_g,
            'val-matrix': val_matrix,
            'test-matrix': test_matrix,
            'item-texts': movie_textual_dataset,
            'item-images': None,
            'user-type': 'user',
            'item-type': 'movie',
            'user-to-item-type': 'watched',
            'item-to-user-type': 'watched-by',
            'timestamp-edge-column': 'timestamp'}

    #     with open(output_path, 'wb') as f:
    #         pickle.dump(dataset, f)

