import numpy as np
from sklearn.base import BaseEstimator
import abc
import math

class Kernel(abc.ABC):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = float(bandwidth)
        
    @abc.abstractmethod
    def fit(self, X):
        raise NotImplemented
    @abc.abstractmethod
    def score_samples(self, P):
        raise NotImplemented
    
    
class TophatKernel(Kernel):
    
    def fit(self, X):
        self.X = np.asarray(X, float)
        self.n_samples, self.n_features = self.X.shape
        self.X_norm_sq = np.sum(self.X**2, axis=1)
        return self

    def score_samples(self, points):
        P = np.atleast_2d(points).astype(float)
        P_norm_sq = np.sum(P**2, axis=1)
        D2 = P_norm_sq[:, None] + self.X_norm_sq[None, :] - 2 * P.dot(self.X.T)
        K = (D2 <= self.bandwidth**2).astype(float)
        Vd = np.pi**(self.n_features/2) / math.gamma(self.n_features/2 + 1)
        norm = Vd * self.bandwidth**self.n_features * self.n_samples
        return K.sum(axis=1) / norm
    
class GaussianKernel(Kernel):
    
    def fit(self, X):
        self.X = np.asarray(X, float)
        self.n_samples, self.n_features = self.X.shape
        self.X_norm_sq = np.sum(self.X**2, axis=1)
        return self

    def score_samples(self, points):
        P = np.atleast_2d(points).astype(float)
        P_norm_sq = np.sum(P**2, axis=1)
        D2 = P_norm_sq[:, None] + self.X_norm_sq[None, :] - 2 * P.dot(self.X.T)
        K = np.exp(-0.5 * D2 / self.bandwidth**2)
        norm = (2*np.pi)**(self.n_features/2) * self.bandwidth**self.n_features * self.n_samples
        return K.sum(axis=1) / norm
    
class KDE(BaseEstimator):
    '''
    bandwidth: float
        The bandwidth of the kernel.
    '''
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.kernel_obj = None
        
    def set_kernel(self, kernel):
        self.kernel = kernel
        if kernel == 'gaussian':
            self.kernel_obj = GaussianKernel(self.bandwidth)
        elif kernel == 'tophat':
            self.kernel_obj = TophatKernel(self.bandwidth)
        else:
            raise ValueError(f'unknown kernel: {kernel}')
        # //TODO: implement other kernels
        
    
    def fit(self, X, y=None):
        '''
        Fit KDE model.
        
        X: DF
        y: array-like of target variable
        '''
        
        self.kernel_obj.fit(X)
        return self
        
    def score_samples(self, points):
        ''' 
        Compute density for points.
        returns array[m]
        '''
        return self.kernel_obj.score_samples(points)
        

        
        