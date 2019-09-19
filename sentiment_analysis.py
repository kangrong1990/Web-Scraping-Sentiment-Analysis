import os
import re
import glob
import pickle
import numpy as np
from collections import Counter

from modules import scraper

import pandas as pd
# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import *
# download required resources
nltk.download("wordnet")
nltk.download("stopwords")

import warnings
warnings.filterwarnings("ignore")


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
        df['Ratings'] = df['Ratings'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
        return df

    def data_preparation(self):
        df = self.data_preprocessing()
        X = df['Reviews']
        y = df['Ratings']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        tvec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
        tvec.fit(X_train)

        Xtrain_tfidf = tvec.transform(X_train)
        Xtest_tfidf = tvec.transform(X_test).toarray()

        return Xtrain_tfidf, Xtest_tfidf, y_train, y_test

    def model_selection(self, name):
        if name == 'Random Forest':
            model = RandomForestClassifier()
        elif name == 'Logistic Regression':
            model = LogisticRegression()
        elif name == 'Naive Bayes':
            model = MultinomialNB()
        else:
            print('Model Not Defined!')
        return model

    def run_model(self):
        Xtrain_tfidf, Xtest_tfidf, y_train, y_test = self.data_preparation()
        model = self.model_selection('Random Forest')
        model.fit(Xtrain_tfidf, y_train)
        ytrain_pred = model.predict(Xtrain_tfidf)
        ytest_pred = model.predict(Xtest_tfidf)
        acc_test = accuracy_score(y_test, ytest_pred)
        print('Test Accuracy: {0:.3f}'.format(acc_test))
        return y_test, ytest_pred

    def confusion_matrix(self):

        y_true, y_pred = self.run_model()
        labels = ['Positive', 'Negative']

        cm = confusion_matrix(y_pred, y_true, labels=labels)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=10, fontweight='bold', pad=10)
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, y_true.unique(), fontsize=12)
        plt.yticks(tick_marks, y_true.unique(), fontsize=12)
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                color = 'black'
                if cm[i][j] > 5:
                    color = 'white'
                plt.text(j, i, format(cm[i][j]),
                         horizontalalignment='center',
                         color=color, fontsize=15)


if __name__ == '__main__':
    homepage = 'https://www.tripadvisor.ca'
    url = 'https://www.tripadvisor.ca/Hotels-g155019-Toronto_Ontario-Hotels.html'
    page_no = 2

    sna = sentiment_analysis(homepage, url, page_no)
    sna.confusion_matrix()
