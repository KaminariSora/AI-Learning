import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")

# train & test set
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

# train80%  test20%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# train algorithm
model = LinearRegression()
model.fit(x_train,y_train)

# test model
y_predict = model.predict(x_test)

#compare true data and predict data
df = pd.DataFrame({
        'Actually': y_test.flatten(),
        'Predict':y_predict.flatten()})

# df1 = df.head(20)
print("MAE",metrics.mean_absolute_error(y_test,y_predict))
print("MSE",metrics.mean_squared_error(y_test,y_predict))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print("ค่าความแม่นยำ Score",metrics.r2_score(y_test,y_predict))
# df1.plot(kind="bar",figsize=(16,10))
# plt.show()