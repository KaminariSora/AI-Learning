import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
rng = np.random

# การจำลองข้อมูล
x = rng.rand(50) * 10
a = rng.rand(10) * 100
c = rng.randn(50)
y = 2*x + c

# linear regression model
model = LinearRegression()
x_reshape = x.reshape(-1,1)

# train algorithm
model.fit(x_reshape,y)

# test model
xfit = np.linspace(-1,11)
xfit_reshape = xfit.reshape(-1,1)
yfit = model.predict(xfit_reshape)

# analysis model & result
plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()

