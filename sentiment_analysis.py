import os
import re
import glob
import pickle
import numpy as np
from collections import Counter
import pickle

from modules import scraper

import pandas as pd
# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import *
# download required resources
nltk.download("wordnet")
nltk.download("stopwords")


class sentiment_analysis(scraper.hotel_scraper):

    def gen_df(self, filename):
        path = self.get_datadir() + filename
        df = self.load_csv(path)
        return df

    def clean_text(self, text):
        sw = stopwords.words('English')
        stemmer = porter.PorterStemmer()
        text = re.sub(r'[^A-Za-z ]', '', text.lower())
        text = ' '.join([stemmer.stem(r) for r in text.split() if r not in sw])
        return text

    def data_preprocessing(self):
        df = self.gen_df('review_data.csv')
        df = df[['Reviews', 'Ratings']]
        df['Reviews'] = df['Reviews'].apply(self.clean_text)
        df['Ratings'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)
        return df

    def data_preparation(self, model_name):
        df = self.data_preprocessing()
        if model_name == 'TF-IDF':
            X = df['Reviews']
            y = df['Ratings']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
            tvec = TfidfVectorizer(max_features=10000, ngram_range=(1, 5))
            tvec.fit(X_train)
            Xtrain_tfidf = tvec.transform(X_train)
            Xtest_tfidf = tvec.transform(X_test).toarray()
            return Xtrain_tfidf, Xtest_tfidf, y_train, y_test
        elif model_name == 'LSTM':
            X, y = (df['Reviews'].values, df['Ratings'].values)
            tk = Tokenizer(num_words=10000, lower=True)
            tk.fit_on_texts(X)
            X_seq = tk.texts_to_sequences(X)
            X_pad = pad_sequences(X_seq, maxlen=100, padding='post')
            X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.3, random_state=101)
            return X_train, X_test, y_train, y_test

    def build_model(self, model_name):
        X_train, X_test, y_train, y_test = self.data_preparation(model_name)
        if model_name == 'TF-IDF':
            mdl_name = 'model_tfidf.h5'
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            best_model = self.model_tuning(model, X_train, y_train)
            with open(self.get_datadir() + mdl_name, 'wb') as file:
                pickle.dump(best_model, file)
        elif model_name == 'LSTM':
            mdl_name = 'model_lstm.h5'
            model = self.lstm_model()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            batch_size = 64
            n_epoch = 10
            model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch)
            with open(self.get_datadir() + mdl_name, 'wb') as file:
                pickle.dump(model, file)

    def lstm_model(self):
        df = self.data_preprocessing()
        X, y = (df['Reviews'].values, df['Ratings'].values)
        tk = Tokenizer(num_words=10000, lower=True)
        tk.fit_on_texts(X)
        vocabulary_size = len(tk.word_counts.keys()) + 1
        max_words = 100
        embedding_size = 50
        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
        model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
        model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def model_tuning(self, model, X, y):
        param_grid = {'n_estimators': [20, 40, 60, 80, 100],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [int(x) for x in np.arange(1, 5)] + [None]
                      }
        rs_clf = RandomizedSearchCV(model, param_grid,
                                    n_jobs=-1, verbose=2, cv=5,
                                    scoring='accuracy', random_state=42)
        model_rs = rs_clf.fit(X, y)
        best_model = model_rs.best_estimator_
        return best_model

    def select_model(self):
        mdl = eval(input('Plese Select Model (1-"TF-IDF", 2-"LSTM"): '))
        while mdl != 1 and mdl != 2:
            print('Wrong Model Selected!')
            mdl = eval(input('Plese Select Model (1-"TF-IDF", 2-"LSTM"): '))
        else:
            if mdl == 1:
                print('Using TF-IDF...')
                mdl_name = 'TF-IDF'
                _, X_test, _, y_test = self.data_preparation(mdl_name)
                model_name = 'model_tfidf.h5'
                with open(self.get_datadir() + model_name, 'rb') as file:
                    model = pickle.load(file)
            elif mdl == 2:
                print('Using LSTM...')
                mdl_name = 'LSTM'
                _, X_test, _, y_test = self.data_preparation(mdl_name)
                model_name = 'model_lstm.h5'
                with open(self.get_datadir() + model_name, 'rb') as file:
                    model = pickle.load(file)
        return model, mdl_name, X_test, y_test

    def result_report(self):
        model, mdl_name, X_true, y_true = self.select_model()
        if mdl_name == 'TF-IDF':
            y_pred = model.predict(X_true)
        elif mdl_name == 'LSTM':
            y_pred = model.predict_classes(X_true)
        print('\033[1m{:10s}\033[0m'.format('The classification report is as below:\n'))
        print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    homepage = 'https://www.tripadvisor.ca'
    url = 'https://www.tripadvisor.ca/Hotels-g155019-Toronto_Ontario-Hotels.html'
    page_no = 2
    sna = sentiment_analysis(homepage, url, page_no)
    sna.build_model('TF-IDF')
    sna.result_report()
