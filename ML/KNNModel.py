from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

iris_datasets = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_datasets['data'], iris_datasets['target'], test_size=0.4, random_state=0)

#model 
knn = KNeighborsClassifier(n_neighbors=1)

#training
knn.fit(x_train, y_train)

#prediction
pred = knn.predict(x_test)

print(classification_report(y_test, pred, target_names=iris_datasets['target_names']))
print("accuracy",accuracy_score(y_test, pred))
