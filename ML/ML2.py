from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()

#75%, 25%
x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0)

print(x_train.shape)