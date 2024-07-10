from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

def displayimage(x):
    plt.imshow(x.reshape(28,28),
               cmap=plt.cm.binary,
               interpolation="nearest")
    plt.show()

def displayPredict(clf,actually_y,x):
    print("Actually = ",actually_y)
    print("Prediction = ",clf.predict([x])[0])

mnist_raw = loadmat("ML/mnist-original.mat")
mnist = {
    "data" : mnist_raw["data"].T,
    "target" : mnist_raw["label"][0]
}
x,y = mnist["data"],mnist["target"]

print(mnist["data"].shape)
print(mnist["target"].shape)

# training set
# class 0-9
x_train, x_test ,y_train, y_test = x[:60000],x[60000:],y[:60000],y[60000:]

# class 0, ไม่ใช่ class 0
# ข้อมูล 5000 ค่า -> model -> class 0 หรือไม่ ? true : false
predict_number = 1000
y_train_0 = (y_train==0)
y_test_0 = (y_test==0)

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train,y_train_0)

score = cross_val_score(sgd_clf,x_train,y_train_0,cv=5,scoring="accuracy")
print(score)

displayPredict(sgd_clf,y_test_0[predict_number],x_test[predict_number])
displayimage(x_test[predict_number])
