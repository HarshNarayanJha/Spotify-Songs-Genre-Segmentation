import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset import load_spotify_dataset

from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv("Spotify Songs' Genre Segmentation/spotify dataset.csv")
print(dataset.head())
print(dataset.info())
print(dataset.describe())

print(dataset.nunique())

dataset = load_spotify_dataset()

# Support Vector Machine (SVM)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

# clf = svm.SVC(kernel='linear')

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))

# K-Nearest Neighbour (KNN)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Decision Trees (DT) 

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("DT Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Logistic Regression (LR)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("LR Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Random Forest (RF)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("RF Accuracy:",metrics.accuracy_score(y_test, y_pred))


