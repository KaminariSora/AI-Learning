import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('ML/adult.csv')

def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset, feature):
    feature=dataset.drop(feature,axis=1) # เอาข้อมูลทั้งหมดยกเว้น income
    labels=dataset[feature].copy() # เอาเฉพาะข้อมูล income
    return feature, labels

dataset = cleanData(dataset)
print(dataset.head())
# split train, test
training_set, test_set = train_test_split(dataset, test_size=0.2)
# train
train_feature, train_labels = split_feature_class(training_set, "income")
# test
test_feature, test_labels = split_feature_class(test_set,"income")
# model
model = GaussianNB()
model.fit(train_feature,train_labels)
#predict
clf_predict = model.predict(test_feature)

print("Accuracy Score = ",accuracy_score(test_labels, clf_predict))