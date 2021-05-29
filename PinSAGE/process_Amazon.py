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


    df1 = getDF('AMAZON_FASHION.json.gz')
    df2 = getDF('meta_AMAZON_FASHION.json.gz')

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

    user = user_dataframe(df1)

    ## Build heterogeneous graph



    # Graph Build
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user, 'user_id', 'user')
    graph_builder.add_entities(item, 'item_id', 'item')
    graph_builder.add_binary_relations(rating, 'user_id', 'item_id', 'buy')
    graph_builder.add_binary_relations(rating, 'item_id', 'user_id', 'purchased-by')

    g = graph_builder.build()


    # Assign

