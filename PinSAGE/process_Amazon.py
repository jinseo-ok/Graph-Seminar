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
import pickle
import pandas as pd
import numpy as np
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

    ## Build heterogeneous graph

    # Load Data
    with open('user.pickle', 'rb') as f:
        user = pickle.load(f)
    with open('item.pickle', 'rb') as f:
        item = pickle.load(f)
    with open('rating.pickle', 'rb') as f:
        rating = pickle.load(f)




    # Load Data
    def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')


    ## user


    ## item
    item = getDF(os.path.join(directory, 'meta_AMAZON_FASHION.json.gz'))
    print(item)

    sys.exit(1)



    # Graph Build
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user, 'user_id', 'user')
    graph_builder.add_entities(item, 'item_id', 'item')
    graph_builder.add_binary_relations(rating, 'user_id', 'item_id', 'buy')
    graph_builder.add_binary_relations(rating, 'item_id', 'user_id', 'purchased-by')

    g = graph_builder.build()

    # Assign

