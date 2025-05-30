import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import pandas as pd

from rbf import RBFRegressor

def test_rgf_vs_skl():
    rng=np.random.RandomState(1)
    
    
    n= 200
    X=rng.uniform(-3,3,size=(n,1))
    y=np.sin(X[:,0])+.05 * rng.randn(n)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=42)
    
    bw = 1.0
    my_rbf = RBFRegressor(bandwidth=bw,a=1.0)
    my_rbf.fit(X_train, y_train)
    my_pr = my_rbf.predict(X_test)
    my_mse = np.mean((my_pr - y_test) ** 2)
    
    skl_rbf = KernelRidge(kernel='rbf', gamma=1/(2*bw**2), alpha=1.0)
    skl_rbf.fit(X_train, y_train)
    skl_pr = skl_rbf.predict(X_test)
    skl_mse = np.mean((skl_pr - y_test) ** 2)
    if abs(my_mse - skl_mse) > 0.02:
        print('my_mse:', my_mse)
        print('skl_mse:', skl_mse)
        raise AssertionError(
            f'my_mse={my_mse}, skl_mse={skl_mse}'
        )
    print('test_rgb_vs_skl passed')
    
def test_rgb_vs_skl_on_df():
    df = pd.read_csv('../../../data/breast-cancer-wisconsin.csv')
    y = df['diagnosis'].map({'M': 1, 'B': 0}).values
    X = df.drop(columns=['id', 'diagnosis']).values
    X_tr, X_te, y_tr,y_te = \
        train_test_split(X, y, test_size=.3, random_state=41, stratify=y)
    
    bw =1.0
    a=1.0
    
    my = RBFRegressor(bandwidth=bw, a=a)
    my.fit(X_tr, y_tr)
    my_pr = my.predict(X_te)
    my_acc = np.mean((my_pr > .5).astype(int) == y_te)
    
    skl = KernelRidge(kernel='rbf', gamma=1/(2*bw**2), alpha=a)
    skl.fit(X_tr, y_tr)
    skl_pr = skl.predict(X_te)
    skl_acc = np.mean((skl_pr > .5).astype(int) == y_te)
    
    print('...[test_rgb_vs_skl_on_df] my_acc:', my_acc)
    print('...[test_rgb_vs_skl_on_df] skl_acc:', skl_acc)
    if abs(my_acc - skl_acc) > 0.05:
        print('my_acc:', my_acc)
        print('skl_acc:', skl_acc)
        raise AssertionError(
            f'my_acc={my_acc}, skl_acc={skl_acc}'
        )
    print('test_rgb_vs_skl_on_df passed')
    
if __name__ == '__main__':
    test_rgf_vs_skl()
    test_rgb_vs_skl_on_df()
    print('all tests passed')