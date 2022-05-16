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

df1 = df[['r1' , 'label']]
df1.rename(columns = {'r1':'ReviewText' , 'label' : "Label"}, inplace = True)
df2 = df[['r2' , 'label']]
df2.rename(columns = {'r2':'ReviewText' , 'label' : "Label"}, inplace = True)
df3 = df[['r3' , 'label']]
df3.rename(columns = {'r3':'ReviewText' , 'label' : "Label"}, inplace = True)

df1 = df1.append(df2 , ignore_index = True)
df = df1.append(df3 ,ignore_index = True)
# df1.head()

df['Label'].value_counts()

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
  y = y=='Accept'
  return train_test_split(data, y, test_size = 0.20, random_state = 0)

X_train, X_test, y_train, y_test = make_dataset(df , (1,1))

"""### Metrics"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, accuracy_score, recall_score
from sklearn import model_selection
from matplotlib import pyplot
import seaborn as sns
def metrics(expected , predicted):
  # Mean absolute error
  print("The f1 is " + str(f1_score(expected, predicted,average='weighted')))
  print("The accuracy is " + str(accuracy_score(expected, predicted)))
  print("The recall is " + str(recall_score(expected, predicted , average='weighted')))

"""## SVM Unigram"""

from sklearn.svm import SVC
regressor = SVC(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predicted)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, predicted, labels=regressor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=regressor.classes_)
disp.plot()
plt.show()

"""## SVM Bi-gram"""

X_train, X_test, y_train, y_test = make_dataset(df , (2,2))
regressor = SVC(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predicted)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, predicted, labels=regressor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=regressor.classes_)
disp.plot()
plt.show()

"""## SVM Unigram + Bi-Gram"""

X_train, X_test, y_train, y_test = make_dataset(df , (1,2))
regressor = SVC(kernel='rbf')
regressor.fit(X_train,y_train)
expected = y_test
predicted = regressor.predict(X_test)

"""### Metrics"""

print(metrics(y_test , predicted))
print("And for training values -------------------")
predicted2 = regressor.predict(X_train)
print(metrics(y_train , predicted2))