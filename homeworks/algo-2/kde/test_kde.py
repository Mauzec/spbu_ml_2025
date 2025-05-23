import numpy as np
import pandas as pd

from kde import KDE
from sklearn.neighbors import KernelDensity

def test_kde_N_skl():
    np.random.seed(0)
    X1 = np.random.randn(1000, 2) + np.array([[0, 0]])
    X2 = np.random.randn(2000, 2) + np.array([[5, 5]])
    X = np.vstack([X1, X2])
    
    i = np.random.permutation(len(X))
    spl = int(len(X) * .8)
    X_train, X_test = X[i[:spl]], X[i[spl:]]
    
    kde = KDE(bandwidth=1.0)
    kde.set_kernel('gaussian')
    kde.fit(X_train)
    kde_score = kde.score_samples(X_test)
    
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X_train)
    kde_skl_score = np.exp(kde_skl.score_samples(X_test))
    
    df = pd.DataFrame({'kde': kde_score, 'kde_skl': kde_skl_score})
    df['abs_err'] = np.abs(df['kde'] - df['kde_skl'])
    df['rel_err'] = df['abs_err'] / (df['kde_skl'] + 1e-8)
    
    # print('abs_err:', df['abs_err'].describe())
    # print('rel_err:', df['rel_err'].describe())
    # print('\nkde score:', kde_score,'\n kde_skl score:', kde_skl_score)
    if df['abs_err'].max() >= 1e-5:
        print('abs_err:', df['abs_err'].describe())
        print('rel_err:', df['rel_err'].describe())
        assert df['abs_err'].max() < 1e-5, 'kde and kde_skl are not equal'
    print('test_kde_N_skl passed')
    

def test_kde_N_skl_on_faithful():
    df = pd.read_csv('../../../data/faithful.tsv', sep='\t')
    X= df[['waiting', 'eruptions']].values
    
    np.random.seed(11)
    i = np.random.permutation(len(X))
    spl = int(len(X) * .8)
    X_train, X_test = X[i[:spl]], X[i[spl:]]
    
    kde = KDE(bandwidth=.5)
    kde.set_kernel('gaussian')
    kde.fit(X_train)
    kde_score = kde.score_samples(X_test)
    
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=.5).fit(X_train)
    kde_skl_score = np.exp(kde_skl.score_samples(X_test))
    
    df = pd.DataFrame({'kde': kde_score, 'kde_skl': kde_skl_score})
    df['abs_err'] = np.abs(df['kde'] - df['kde_skl'])
    df['rel_err'] = df['abs_err'] / (df['kde_skl'] + 1e-8)
    if df['abs_err'].max() >= 1e-5:
        print('abs_err:', df['abs_err'].describe())
        print('rel_err:', df['rel_err'].describe())
        assert df['abs_err'].max() < 1e-5, 'kde and kde_skl are not equal'
    print('test_kde_N_skl_on_faithful passed')

if __name__ == "__main__":
    test_kde_N_skl()
    test_kde_N_skl_on_faithful()