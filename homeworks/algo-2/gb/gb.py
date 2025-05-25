import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DSRegressor(BaseEstimator, RegressorMixin):
    '''
    
    Decision Stump Regressor.
    divide the feature by a threshold and return constant in each part
    
    '''
    
    def __init__(self):
        self.ft_idx = None; 
        self.thr = None;
        self.lval = None;
        self.rval = None;
        

    def fit(self,X,y):
        X= np.asarray(X,float)
        y = np.asarray(y,float)
        n_samples, n_features = X.shape
        
        best_sse = np.inf
        for j in range(0,n_features):
            vals=X[:,j]
            un_vals = np.unique(vals)
            for thr in un_vals:
                lm = vals <= thr
                if (lm.all() or (~lm).all()): continue
                
                l_mean = y[lm].mean()
                r_mean = y[~lm].mean()
                sse = (
                    ((y[lm] - l_mean)**2).sum()+\
                    ((y[~lm] - r_mean)**2).sum()
                )
                if sse >= best_sse: continue
                
                best_sse = sse
                self.ft_idx = j
                self.thr = thr
                self.lval = l_mean
                self.rval = r_mean
                
                
                
        return self
      
      
    def predict(self,X):
        X=np.asarray(X,float)          
        vals=X[:,self.ft_idx]
        
        pr = np.where(
            vals <= self.thr,
            self.lval,
            self.rval
        )
        return pr
    def __str__(self):
        return f'DSRegressor(ft_idx={self.ft_idx}, thr={self.thr}, lval={self.lval}, rval={self.rval})'
    def __repr__(self):
        return self.__str__()
    
    
    
class GBRegressor(BaseEstimator, RegressorMixin):
    '''
    Gradient Boosting Regressor.
    
    '''
    
    def __init__(self, lr=.1, n_estimators=100):
        self.lr = float(lr)
        self.n_estimators = int(n_estimators)
    
    
    def fit(self,X,y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel
        n_samples = y.size

        self._first_pr = y.mean()
        
        f = np.full(n_samples, self._first_pr, float)
        self._estimators = []
        
        for _ in range(self.n_estimators):
            r = y-f
            dst = DSRegressor().fit(X, r)
            pr = dst.predict(X)
            
            f = f +self.lr * pr
            self._estimators.append(dst)
            
        return self
    

    def predict(self,X):
        X=np.asarray(X,float)
        f = np.full(X.shape[0], self._first_pr, float)
        for dst in self._estimators:
            f = f +self.lr * dst.predict(X)
        return f