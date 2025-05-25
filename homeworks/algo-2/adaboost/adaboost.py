import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionStump(BaseEstimator, ClassifierMixin):


    def __init__(self):
        self.feature_idx = None;
        self.thr = None;
        self.polar = None;
        
    def fit(self, X, y,weights):
        '''
        Fit the decision stump;
        
        weights: array-like, shape (n_samples,)
            The weights for each sample.
        ''' 
        
        X = np.asarray(X) 
        y = np.asarray(y) 
        w = np.asarray(weights) 
        n_samples, n_features = X.shape
        min_err = np.inf
        
        for fidx in range(n_features):
            vals = X[:, fidx]
            
            for t in np.unique(vals) :
                for p in [-1, +1]:
                    # p X[fidx] < pt => -1
                    
                    pr = np.ones(  n_samples)
                    m = p*vals <p * t
                    pr[m] = -1 
                    err = np.sum(w[y != pr])
                     
                    if err < min_err:
                        min_err = err
                        self.feature_idx = fidx
                        self.thr = t
                        self.polar = p
         
        return self
    
    def predict(self, X):
        '''
        ''' 
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        pr = np.ones(n_samples)
        vals =  X[:,self.feature_idx]
        m = self.polar*vals<self.polar*self.thr
        pr[m] = -1
        return pr 
    
    def __str__(self): 
        return f'DecisionStump(feature_idx={self.feature_idx}, thr={self.thr}, polar={self.polar})'
    def __repr__(self):
        return self.__str__()


class AdaBoost(BaseEstimator, ClassifierMixin):
    '''
    AdaBoost Classifier
    
    Parameters:
    lr: float, default=1.0
        Learning rate for the model.
    n_estimators: int, default=100
        Number of estimators (decision stumps) to use in the ensemble.
    '''
    
    def __init__(self, lr=1.0, n_estimators=100):
        self.lr = float(lr)
        self.n_estimators = int(n_estimators)
        
    def fit(self,X,y):
        X=np.asarray(X)
        y=np.asarray(y)
        self._classes = np.unique(y)
        
        y1 = np.where(y==self._classes[0], -1, +1)
        n_samples = X.shape[0]
        w =np.full(n_samples,float(1)/n_samples)
        
        self._est, self._w_est = list(), list()
        for i in range(self.n_estimators):
            dst = DecisionStump().fit(X,y1,w)
            pr = dst.predict(X)
            
            err = np.dot(w, (y1 != pr))
            err = np.clip(err,1e-10,1.0 - 1e-10)
            
            alpha = self.lr * np.log((1.0 - err) / err) * .5
            
            w *= np.exp(- alpha * y1 * pr)
            w = w / np.sum(w)
            self._est.append(dst)
            self._w_est.append(alpha)
        return self
    

    def predict(self,X):
        X = np.asarray(X)
        agg = np.zeros(X.shape[0])
        for alpha, st in zip(self._w_est, self._est):
            agg += alpha * st.predict(X)
        y1 = np.sign(agg)
        y1 = np.where(y1 == -1, self._classes[0], self._classes[1])
        return y1
    def predict_proba(self,X):
        X=np.asarray(X)
        agg = np.zeros(X.shape[0])
        for alpha, st in zip(self._w_est, self._est):
            agg += alpha * st.predict(X)
        prob_pos=1.0/( 1.0 + np.exp(-2.0*agg))
        proba = np.vstack([1.0 - prob_pos, prob_pos]).T
        return proba
    def __str__(self):
        return f'AdaBoost(lr={self.lr}, n_estimators={self.n_estimators})'
    def __repr__(self):
        return self.__str__()

            