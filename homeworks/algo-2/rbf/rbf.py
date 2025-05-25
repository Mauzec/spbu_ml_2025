import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RBFRegressor(BaseEstimator, RegressorMixin):
    '''
    RBF Kernel Regresor
    '''
    
    def __init__(self, bandwidth=1.0, a = 1.0):
        self.bandwidth = float(bandwidth)
        self.a = float(a)
    
    def _gauss(self, d):
        return np.exp(-.5 * (d/self.bandwidth)**2)
    
    def _kernel_matrix(self,x1,x2):
        d = x1[:,None,:] - x2[None,:,:]
        dists = np.linalg.norm(d, axis=2)
        return self._gauss(dists)
    
    def _kernel(self, d):
        u = d/self.bandwidth
        if self.kernel=='gaussian':
            return np.exp(-0.5*u**2)
        
    def fit(self,X,y):
        X, y = np.asarray(X,float), np.asarray(y,float).ravel()
        k = self._kernel_matrix(X, X)
        n=k.shape[0]
        
        A = k+self.a * np.eye(n)
        self.duel_coef = np.linalg.solve(A, y)
        self._X_train = X
        return self
    def predict(self,X):
        X = np.asarray(X, float)
        k = self._kernel_matrix(X, self._X_train)
        return k.dot(self.duel_coef)
        
