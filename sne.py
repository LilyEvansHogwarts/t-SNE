import numpy as np
from pylab import *

class SNE:
    def __init__(self, train_x, train_y, no_dim=2, pca_dim=50):
        self.train_x = np.copy(train_x)
        self.train_y = np.copy(train_y)
        self.num_train, self.dim = self.train_x.shape
        self.no_dim = no_dim
        self.pca_dim = pca_dim
        self.train_x = self.pca(self.pca_dim)
        self.X_dist = self.distance(self.train_x)

    def pca(self, ndim=2):
        print('Preprocessing X with PCA...')
        mean = self.train_x.mean(axis=0)
        X = self.train_x - mean
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        Y = np.dot(X, M[:,:ndim])
        return Y.real

    def distance(self, X):
        X2_sum = np.square(X).sum(axis=1)
        return np.add(np.add(-2*np.dot(X, X.T), X2_sum).T, X2_sum)

    def Hbeta(self, D=np.array([]), beta=1.0):
        P = np.exp(-D * beta)
        P_sum = P.sum()
        H = np.log(P_sum) + beta*(D*P).sum()/P_sum
        P = P/P_sum
        return H, P

    def P_values(self, perplexity=20.0):
        log_p = np.log(perplexity)
        self.beta = np.ones((self.num_train))
        self.P = np.zeros((self.num_train, self.num_train))

        for i in range(self.num_train):
            if i % 500 == 0:
                print('Computing P-values for point %d of %d...' % (i, self.num_train))
            betamax = np.inf
            betamin = -np.inf
            Di = np.concatenate((self.X_dist[i,:i], self.X_dist[i,i+1:]))
            (H, thisP) = self.Hbeta(Di, self.beta[i])
            Hdiff = H - log_p
            tries = 0
            while np.abs(Hdiff) > 1e-5 and tries < 50:
                if Hdiff > 0:
                    betamin = self.beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        self.beta[i] = self.beta[i] * 2
                    else:
                        self.beta[i] = (self.beta[i] + betamax)/2
                else:
                    betamax = self.beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        self.beta[i] = self.beta[i] / 2.0
                    else:
                        self.beta[i] = (self.beta[i] + betamin)/2

                (H, thisP) = self.Hbeta(Di, self.beta[i])
                Hdiff = H - log_p
                tries += 1
            self.P[i,:i] = thisP[:i]
            self.P[i,i+1:] = thisP[i:]
        
        np.savetxt('sne_P.txt', self.P)
        print('Mean value of sigma: %f' % np.mean(np.sqrt(1/self.beta)))
        return self.P

    def train(self, perplexity=20.0):
        Y = np.random.randn(self.num_train, self.no_dim)
        dY = np.zeros((self.num_train, self.no_dim))
        iY = np.zeros((self.num_train, self.no_dim))
        initial_momentum = 0.5
        final_momentum = 0.8
        maxiter = 1000
        eta = 500
        min_gain = 0.01
        gains = np.ones((self.num_train, self.no_dim))

        self.P_values(perplexity)
        self.P = (self.P + self.P.T)/np.sum(self.P)
        self.P = self.P * 4
        self.P = np.maximum(self.P, 1e-12)

        for i in range(maxiter):
            Y_dist = self.distance(Y)
            num = 1.0 / (1.0 + Y_dist)
            num[range(self.num_train, self.num_train)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = self.P - Q
            for k in range(self.num_train):
                # dY[k,:] = -((PQ[k] * num[k])[:,None] * (Y - Y[k])).sum(axis=0)
                # dY[k,:] = -((PQ[:,k] * num[:,k])[:,None] * (Y - Y[k])).sum(axis=0)
                dY[k,:] = np.sum(np.tile(PQ[:,k]*num[:,k], (self.no_dim,1)).T * (Y[k,:]-Y),0)

            if i < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - Y.mean(axis=0)

            if (i + 1) % 10 == 0:
                C = np.sum(self.P * np.log(self.P / Q))
                print('Iteration %d: error is %f' % (i+1, C))

            if i == 100:
                self.P = self.P / 4

        return Y

            
if __name__ == '__main__':
    train_x = np.loadtxt('mnist2500_X.txt')
    train_y = np.loadtxt('mnist2500_labels.txt')
    model = SNE(train_x, train_y)
    Y = model.train()
    scatter(Y[:,0], Y[:,1], 20, train_y)
    show()
    
