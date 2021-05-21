import operator
import os
import re
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from wordcloud import STOPWORDS

import datacleaner

for dir_name, _, file_names in os.walk('/root/hotown/hw2_v2/data'):
    # for file_name in file_names:
    #     print(os.path.join(dir_name, file_name))
    df_train = pd.read_csv(os.path.join(dir_name, 'train.txt'))
    df_test = pd.read_csv(os.path.join(dir_name, 'test.txt'))

    print(f'Training Set Shape: {df_train.shape}')
    print(f'Traning Set Index: {list(df_train.columns)}')
    print(f'Test Set Shape: {df_test.shape}')
    print(f'Test Set Index: {list(df_test.columns)}')
    # Training Set Shape: (135000, 15)
    # Traning Set Index: ['age', 'body type', 'bust size', 'category', 'fit', 'height', 'item_id', 'rating', 'rented for', 'review_date', 'review_summary', 'review_text', 'size', 'user_id', 'weight']
    # Test Set Shape: (57544, 14)
    # Test Set Index: ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'review_date', 'review_summary', 'review_text', 'size', 'user_id', 'weight']

    missing_cols = list(df_train.columns)
    missing_test_cols = list(df_test.columns)
    fig, axes = plt.subplots(ncols=2, figsize=(45, 10), dpi=100)

    sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, ax=axes[0])
    sns.barplot(x=df_test[missing_test_cols].isnull().sum().index, y=df_test[missing_test_cols].isnull().sum().values, ax=axes[1])

    axes[0].set_ylabel('Missing Value Count')
    axes[0].set_title('Traing Set')
    axes[1].set_title('Test Set')

    plt.savefig('missing_cols')

    # fig = plt.plot()

    # sns.barplot(x=list(set(df_train['fit'])), y=[df_train['fit'].apply(lambda x: str(x) == 'small').sum(), df_train['fit'].apply(lambda x: str(x) == 'fit').sum(), df_train['fit'].apply(lambda x: str(x) == 'large').sum()])

    # plt.savefig('fit_count')

    for df in [df_train, df_test]:
        df['review_text'] = df['review_text'].fillna('')

    df_train['word_count'] = df_train['review_text'].apply(lambda x:len(str(x).split()))
    # print(df_train['word_count'])
    df_test['word_count'] = df_test['review_text'].apply(lambda x:len(str(x).split()))

    df_train['unique_word_count'] = df_train['review_text'].apply(lambda x:len(set(str(x).split())))
    df_test['unique_word_count'] = df_test['review_text'].apply(lambda x:len(set(str(x).split())))
    
    df_train['stop_word_count'] = df_train['review_text'].apply(lambda x:len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df_test['stop_word_count'] = df_test['review_text'].apply(lambda x:len([w for w in str(x).lower().split() if w in STOPWORDS]))

    df_train['mean_word_length'] = df_train['review_text'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))
    df_test['mean_word_length'] = df_test['review_text'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))

    meta_feature = ['word_count', 'unique_word_count', 'stop_word_count', 'mean_word_length']
    fit_tweets = df_train['fit'] == 'fit'
    large_tweets = df_train['fit'] == 'large'
    small_tweets = df_train['fit'] == 'small'

    fig, axes = plt.subplots(ncols=2, nrows=len(meta_feature), figsize=(20,40), dpi=100)

    for i, feature in enumerate(meta_feature):
        sns.distplot(df_train.loc[fit_tweets][feature], label='Fit', ax=axes[i][0], color='green')
        sns.distplot(df_train.loc[small_tweets][feature], label='Small', ax=axes[i][0], color='yellow')
        sns.distplot(df_train.loc[large_tweets][feature], label='Large', ax=axes[i][0], color='red')

        sns.distplot(df_train[feature], label='Train', ax=axes[i][1])
        sns.distplot(df_test[feature], label='Test', ax=axes[i][1])

        for j in range(2):
            axes[i][j].legend()

        axes[i][0].set_title(f'{feature} Fit Distribution in Training Set')
        axes[i][1].set_title(f'{feature} Train & Test Set Distribution')
    
    plt.savefig('meta_feature')

    # df_train['text_clean'] = df_train['review_text'].apply(lambda s: datacleaner.clean(s))
    # df_test['text_clean'] = df_test['review_text'].apply(lambda s: datacleaner.clean(s))

    # TODO:spell check

    # df_train.to_csv('./data/cleaned_train.csv', index=False)
    # df_test.to_csv('./data/cleaned_test.csv', index=False)

    print('finish!')
