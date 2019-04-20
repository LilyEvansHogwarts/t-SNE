import numpy as np
from pylab import *

train_x = np.loadtxt('mnist2500_X.txt')
train_y = np.loadtxt('mnist2500_labels.txt')

def pca(X=np.array([]), ndim=2):
    mean = X.mean(axis=0)
    X = X - mean
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:,:ndim])
    return Y

tmp = pca(train_x, 2).real
print(tmp.shape)
print(train_y.shape)
scatter(tmp[:,0], tmp[:,1], 20, train_y)
show()
