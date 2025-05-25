import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DTRegressor(BaseEstimator, RegressorMixin):
    '''
    CART
    Decision Tree Regressor.
    divide the feature by a threshold and return constant in each part
    
    '''
    
    def __init__(self, md=1, min_s_split=2):
        self.max_depth = int(md)
        self.min_s_split = int(min_s_split)
        self.tree=None
        
        
    class Node:
    
        def __init__(self, ft=None, thr=None, l=None, r=None, v=None):
           self.ft = ft
           self.thr = thr;
           self.l = l
           self.r = r
           self.v = v 
        
    def fit(self,X,y):
        X=np.asarray(X,float,)
        y=np.asarray(y,float).ravel()
        self.tree=self.build_tree(X,y,0)
        return self
    def build_tree(self,X,y,d):
        n_samples,n_features=X.shape
        if (
            d >= self.max_depth or
            n_samples < self.min_s_split or
            np.unique(y).size == 1
        ):
            return DTRegressor.Node(v=y.mean())

        best_ft = None
        best_thr = None
        best_sse = np.inf
        best_splits = None
        
        for j in range(n_features):
            vals = X[:, j]
            
            un_vals = np.unique(vals)
            for thr in un_vals:
                lm = vals<=thr
                if (lm.all() or (~lm).all()):
                    continue
                l_y = y[lm]
                r_y = y[~lm]
                sse = (
                    ((l_y - l_y.mean()) ** 2).sum() + 
                    ((r_y - r_y.mean()) ** 2).sum()
                )
                if sse >= best_sse:
                    continue
                best_sse = sse
                best_ft = j
                best_thr = thr
                best_splits = (lm, ~lm)
                
        if best_ft is None:
            return DTRegressor.Node(v=y.mean())
        left_tree = self.build_tree(
            X[best_splits[0]], y[best_splits[0]], d + 1
        )
        right_tree = self.build_tree(
            X[best_splits[1]], y[best_splits[1]], d + 1
        )
        return DTRegressor.Node(
            ft=best_ft,
            thr=best_thr,
            l=left_tree,
            r=right_tree,
        )
        
    def pr1(self,x,n):
        if n.v is not None: return n.v
        if x[n.ft] <= n.thr: return self.pr1(x,n.l)
        return self.pr1(x,n.r)
    def predict(self, X):
        X= np.asarray(X, float)
        pr = np.array([self.pr1(x,self.tree) for x in X])
        return pr
        
    
    
    
class GBRegressor(BaseEstimator, RegressorMixin):
    '''
    Gradient Boosting Regressor.
    
    '''
    
    def __init__(self, lr=.1, n_estimators=100, md=1):
        self.lr = float(lr)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(md)
    
    
    def fit(self,X,y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n_samples = y.size

        self._first_pr = y.mean()
        
        f = np.full(n_samples, self._first_pr, float)
        self._estimators = []
        
        for _ in range(self.n_estimators):
            r = y-f
            dst = DTRegressor(md=self.max_depth).fit(X, r)
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