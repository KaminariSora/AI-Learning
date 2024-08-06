from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("ML/diabetes.csv")

# data
x = df.drop("Outcome", axis=1).values

# Outcome data
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

# find k to model
k_neighbor = np.arange(1,9)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)

# train
knn.fit(x_train, y_train)

# prediction
y_pred = knn.predict(x_test)
pd.crosstab(y_test, y_pred,rownames=['Actually'],colnames=['Predict'])