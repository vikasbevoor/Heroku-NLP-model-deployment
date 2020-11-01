# Importing essential libraries
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Loading the dataset
df = pd.read_csv(r"D:\Data science\Assignments docs\Naive Bayes\ham_spam.csv", encoding="latin")


# Features and Labels
df['label'] = df['type'].map({'ham': 0, 'spam': 1})
X = df['text']
y = df['label']

#Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
   
pickle.dump(cv, open('tranform.pkl', 'wb'))
  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
