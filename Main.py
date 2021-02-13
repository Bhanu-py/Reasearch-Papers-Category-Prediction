import pandas as pd
import texthero as hero
import numpy as np
import string
from texthero import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB


test_df = pd.read_csv(r"E:\Hackathon\Research paper search\test_8iecVfC\test.csv", encoding='utf-8')
train_df = pd.read_csv(r"E:\Hackathon\Research paper search\train_tGmol3O\train.csv", encoding='utf-8')

pd.set_option('max_columns', None)

test_df["TITLE"] = hero.clean(test_df["TITLE"])
train_df["TITLE"] = hero.clean(train_df["TITLE"])

test_df["ABSTRACT"] = hero.clean(test_df["ABSTRACT"])
train_df["ABSTRACT"] = hero.clean(train_df["ABSTRACT"])

train_df["Content"] = train_df["TITLE"] + train_df["ABSTRACT"]
test_df["Content"] = test_df["TITLE"] + test_df["ABSTRACT"]
train_df["Content"] = hero.clean(train_df["Content"])

train_df_x = train_df["Content"]
test_df_x = test_df["Content"]
train_df_y = train_df.iloc[:, [3, 4, 5, 6, 7, 8]]

stp = string.punctuation
# stp = "!”#$%&’()*+,-./:;<=>?@[]^_`{|}~\\"
print("'Text processed'")

def text_process(mess):
    nopunc = [char for char in mess if char not in stp]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word not in stopwords.words("english")]


bow_transformer_train = CountVectorizer(analyzer=text_process).fit(train_df["Content"])
bow_transformer_test = CountVectorizer(analyzer=text_process).fit(test_df["Content"])

messages_bow_train = bow_transformer_train.transform(train_df["Content"])
messages_bow_test = bow_transformer_train.transform(test_df["Content"])
# print(bow_transformer)
# print(messages_bow)
# print
print("'Data is Preprocessed'")

classifier = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None)).fit(messages_bow_train, train_df_y) ### 0.79
# classifier = LabelPowerset(LogisticRegression())
# classifier = ClassifierChain(classifier=RandomForestClassifier(n_estimators=100), require_dense = [False, True])
# classifier = BinaryRelevance(GaussianNB())
# rf = RandomForestClassifier()
print("'Model is Created'")

print("Model is Training----------------")
classifier.fit(messages_bow_train, train_df_y)
print("'Model is Trained'")

pred_spam = classifier.predict(messages_bow_test)
np.savetxt("test.csv", pred_spam, delimiter=",")
print(pred_spam)
print(type(pred_spam))

# tab_spam = confusion_matrix(train_df_y.values.argmax(axis=1), pred_spam.argmax(axis=1))
# print(tab_spam)
#
# acc = tab_spam.diagonal().sum() / tab_spam.sum() * 100
#
# print(acc)

