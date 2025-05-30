import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from stl import STL as STLMY


def test_stl_N_sm():
    np.random.seed(11)
    
    n = 240
    t = np.arange(n)
    
    trend_true = .1 *t
    seasonal_true = 5.0 * np.sin(2 * np.pi * t / 12)
    
    noise = np.random.randn(n) * .5
    
    y = trend_true + seasonal_true + noise
    
    stl = STLMY(period=12, trend_wdw=13, seasonal_wdw=13, inner_loops=5, outer_loops=2)
    
    df_stl = stl.fit_transform(y)
    
    stl_ref = STL(y, period=12, seasonal=13, trend=13, robust=False)
    stl_ref_res = stl_ref.fit()
    df_stl_ref = pd.DataFrame({
        'trend': stl_ref_res.trend,
        'seasonal': stl_ref_res.seasonal,
        'residual': stl_ref_res.resid
    })
    
    mse_trend = np.mean((df_stl['trend'] - df_stl_ref['trend'])**2)
    mse_seasonal = np.mean((df_stl['seasonal'] - df_stl_ref['seasonal'])**2)
    mse_residual = np.mean((df_stl['residual'] - df_stl_ref['residual'])**2)
   
    if mse_trend >= 2.0 or mse_seasonal >= 5.0 or mse_residual >= 1.0:
        print('mse_trend:', mse_trend)
        print('mse_seasonal:', mse_seasonal)
        print('mse_residual:', mse_residual)
        assert mse_trend < 2.0, 'STL and STL_ref are not equal' 
        assert mse_seasonal < 5.0, 'STL and STL_ref are not equal'
        assert mse_residual < 1.0, 'STL and STL_ref are not equal' 
    print('test_stl_N_sm passed')
    

def make_ts():
    np.random.seed(11)
    n=120
    t=np.arange(n)
    trend=.05 * t
    seasonal = 2.0 * np.sin(2*np.pi * t / 12)
    noise = np.random.randn(n) * .3
    y = trend + seasonal + noise
    return y,trend, seasonal, noise
def mse(a, b):
    return np.mean((a - b)**2)

def test_outlier_in_residual():
    y, trend, seasonal, noise = make_ts()
    y2 = y.copy()
    y2[50] += 10 #outlier
    stl = STLMY(period=12, seasonal_wdw=13, trend_wdw=25, inner_loops=2)
    df = stl.fit_transform(y2)
 
    resid = df['residual']
    assert resid.iloc[50] > 5 * resid.abs().mean()

    
if __name__ == '__main__':
    test_stl_N_sm()
    test_outlier_in_residual()