# -*- coding: utf-8 -*-


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./2017 Data')
args = parser.parse_args()

data_path = args.dataset+'/'

df = pd.read_csv(data_path + 'Data.csv')

df1 = df[['r1' , 'rec1']]
df1.rename(columns = {'r1':'ReviewText' , 'rec1' : "Recommendation"}, inplace = True)
df2 = df[['r2' , 'rec2']]
df2.rename(columns = {'r2':'ReviewText' , 'rec2' : "Recommendation"}, inplace = True)
df3 = df[['r3' , 'rec3']]
df3.rename(columns = {'r3':'ReviewText' , 'rec3' : "Recommendation"}, inplace = True)

df1 = df1.append(df2 , ignore_index = True)
df = df1.append(df3 ,ignore_index = True)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

def make_dataset(df , gram):
  corpus = []
  for i in range(0, df.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', df['ReviewText'][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review)
  vectorizer = CountVectorizer(stop_words='english', ngram_range=gram, dtype='double',)
  data = vectorizer.fit_transform(corpus)
  vectors=data
  tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
  tfidf_transformer.fit(data)
  data=tfidf_transformer.transform(data)
  pca = TruncatedSVD(n_components=64)
  data = pca.fit_transform(data)
  y = df.iloc[:, 1].values
  return train_test_split(data, y, test_size = 0.20, random_state = 0)

X_train, X_test, y_train, y_test = make_dataset(df , (1,1))

print(X_train.shape)
print(y_train.shape)

"""### Metrics"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from matplotlib import pyplot
import seaborn as sns
def metrics(expected , predicted):
  # Mean absolute error
  print("The rmse is " + str(np.sqrt(mean_squared_error(expected, predicted))))
  print("The mae is " + str(mean_absolute_error(expected, predicted)))
  print("The r2 is " + str(r2_score(expected, predicted)))
  print("Standard Deviation of predicted" + str(np.std(predicted)))
  plt.figure()
  ax1 = sns.distplot(predicted)
  ax2 = sns.distplot(expected)
  plt.axvline(np.mean(predicted) , color='b' , linestyle='dashed' , linewidth='2')
  plt.axvline(np.mean(expected) , color='orange' , linestyle='dashed' , linewidth='2')
  # pyplot.plot(err)
  # pyplot.xticks(ticks=[i for i in range(len(err))], labels=predicted)
  # pyplot.xlabel('Predicted Value')
  # pyplot.ylabel('Mean Squared Error')
  # pyplot.show()
  # mean sq error
  # r2

"""## SVM Unigram"""

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))

"""## SVM Bi-gram"""

X_train, X_test, y_train, y_test = make_dataset(df , (2,2))
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))

"""## SVM Unigram + Bi-Gram"""

X_train, X_test, y_train, y_test = make_dataset(df , (1,2))
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))