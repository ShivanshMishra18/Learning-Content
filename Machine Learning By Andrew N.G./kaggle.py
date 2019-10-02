import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
from numpy.core.umath_tests import inner1d

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn
import sklearn.ensemble as ske

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("/home/soham/Desktop/codes/ml/test.csv")
test_df    = pd.read_csv("/home/soham/Desktop/codes/ml/test.csv")

# preview the data
titanic_df.head()
titanic_df.info()
print("----------------------------")
test_df.info()
titanic_df.head()
# print(titanic_df['survived'].mean())
# print(titanic_df.groupby('pclass').mean())
# class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
# print(class_sex_grouping)
# class_sex_grouping['survived'].plot.bar()
# print(titanic_df.count())

# titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
# titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
# titanic_df = titanic_df.dropna()
# print(titanic_df.count())

# def preprocess_titanic_df(df):
#     processed_df = df.copy()
#     le = preprocessing.LabelEncoder()
#     processed_df.sex = le.fit_transform(processed_df.sex)
#     processed_df.embarked = le.fit_transform(processed_df.embarked)
#     processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
#     return processed_df

# processed_df = preprocess_titanic_df(titanic_df)

# X = processed_df.drop(['survived'], axis=1).values
# y = processed_df['survived'].values

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
# clf_dt = tree.DecisionTreeClassifier(max_depth=10)
# clf_dt.fit (X_train, y_train)
# print(clf_dt.score (X_test, y_test))
# shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)
# def test_classifier(clf):
#     scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
#     print("Accuracy: %0.4f (+/- %0.3f)" % (scores.mean(), scores.std()))
# def test_classifier(clf):
#     scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
#     print("Accuracy: %0.4f (+/- %0.3f)" % (scores.mean(), scores.std()))
    
# print(test_classifier(clf_dt))
# clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
# print(test_classifier(clf_gb))

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)
random_forest = ske.RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred_1 = random_forest.predict(X_test)
print(Y_pred_1)
