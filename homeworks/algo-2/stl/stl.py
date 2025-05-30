import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class STL(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        period:int, trend_wdw:float=.25, seasonal_wdw:float=.25, 
        inner_loops:int=2, outer_loops:int=1
    ):
        self.period = int(period)
        self.seasonal_wdw = float(seasonal_wdw)
        self.trend_wdw = float(trend_wdw)
        self.inner_loops = int(inner_loops)
        self.outer_loops = int(outer_loops)
        
    def fit(self, y, X=None):
        ''' 
        y = 1d array-like
        '''
        
        y = np.asarray(y, float)
        y = y.ravel()
        n = y.size
        p = self.period
        
        if self.trend_wdw >= 1:
            t_span = int(self.trend_wdw)
        else:
            t_span = max(int(np.ceil(self.trend_wdw * n)), 2)

        if self.seasonal_wdw >= 1:
            s_span = int(self.seasonal_wdw)
        else:
            s_span = max(int(np.ceil(self.seasonal_wdw * n)), 2)

        
        trend = self._loess(y,t_span)
        
        seasonal = np.zeros(n, float)
        for _ in range(self.outer_loops):
            for _ in range(self.inner_loops):
                detrended=y-trend
                # season_temp = np.array([
                #     detrended[j::p].mean()
                #     for j in range(p)
                # ])
                
                seasonal = np.zeros(n, float)
                for j in range(p):
                    seq = detrended[j::p]
                    span_j = min(s_span, seq.size)
                    smooth_j = self._loess(seq, span_j)
                    seasonal[j::p] = smooth_j
                seasonal -= seasonal.mean()
                
                trend = self._loess(y - seasonal, t_span)
        
        trend_lp = self._loess(y-seasonal,t_span)
        
        
        r = y-trend_lp - seasonal
        self._trend = trend_lp
        self._seasonal = seasonal
        self._residual = r
        return self
        
    def transform(self,X):
        return pd.DataFrame({
            'trend': self._trend,
            'seasonal': self._seasonal,
            'residual': self._residual
        })
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
    def _loess(self, series, span_ratio):
        ''' 
        span_ratio: fraction of input data to use for local regression
        '''
        n = series.size
        idx = np.arange(n)
        out = np.zeros(n, float)
        
        for i in range(n):
            # time dist
            d = np.abs(idx - i)
            
            # closest span points
            sel = np.argsort(d)[:span_ratio]
            dm = d[sel].max()

            w = (1 - (d[sel]/dm)**3)**3 if dm>0 else np.ones(span_ratio)
                
            A = np.stack([np.ones(span_ratio), idx[sel]-i], axis=1)
            W = np.diag(w)
            beta = np.linalg.pinv(A.T@W@A) @ (A.T@(w*series[sel]))
            out[i] = beta[0]
        return out