import pylab
from sklearn import datasets

digit_dataset = datasets.load_digits()

pylab.imshow(digit_dataset.images[:10], cmap=pylab.cm.gray_r)
pylab.show()