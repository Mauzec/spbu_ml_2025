import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from gb import GBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

def test_gb_vs_skl():
    np.random.seed(65)
    n =200
    X= np.random.uniform(-5,5,size=(n,1))
    y=np.sin(X[:,0])+ .1 * np.random.randn(n)
    
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:150],idx[150:]
    X_train, X_test = X[train_idx],X[test_idx]
    y_train, y_test = y[train_idx],y[test_idx]
    
    my_gb = GBRegressor(lr=.1, n_estimators=100, md=3)
    my_gb.fit(X_train, y_train)
    my_pr = my_gb.predict(X_test)
    my_mse = np.mean((my_pr - y_test) ** 2)
    
    skl_gb = GradientBoostingRegressor(
        learning_rate=.1,
        n_estimators=100,
        max_depth=3,
    )
    skl_gb.fit(X_train, y_train)
    skl_pr = skl_gb.predict(X_test)
    skl_mse = np.mean((skl_pr - y_test) ** 2)
    
    if abs(my_mse - skl_mse) > 0.1:
        print('my_mse:', my_mse)
        print('skl_mse:', skl_mse)
        raise AssertionError(
            f'my_mse={my_mse}, skl_mse={skl_mse}'
        )
    print('test_gb_vs_skl passed')
        
def test_gb_vs_skl_on_df():
    df =pd.read_csv('../../../data/breast-cancer-wisconsin.csv')
    y = df['diagnosis'].map({'M': 1, 'B': 0}).values
    X = df.drop(columns=['id', 'diagnosis']).values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=41, stratify=y)
        
    my_gb = GBRegressor(lr=.0666, n_estimators=55, md=3)
    my_gb.fit(X_train, y_train)
    my_pr = my_gb.predict(X_test)
    my_acc = np.mean(( my_pr > .5  ).astype(int) == y_test)

    skl_gb = GradientBoostingRegressor(
        learning_rate=.0666,
        n_estimators=55,
        max_depth=3,
    )
    skl_gb.fit(X_train, y_train)
    skl_pr = skl_gb.predict(X_test)
    skl_acc = np.mean(( skl_pr > .5  ).astype(int) == y_test)
    
    print('...[test_gb_vs_skl_on_df] my_acc:', my_acc)
    print('...[test_gb_vs_skl_on_df] skl_acc:', skl_acc)
    if abs(my_acc - skl_acc) > 0.05:
        print('my_acc:', my_acc)
        print('skl_acc:', skl_acc)
        raise AssertionError(
            f'my_acc={my_acc}, skl_acc={skl_acc}'
        )
    print('test_gb_vs_skl_on_df passed')
    
    

if __name__ == '__main__':
    test_gb_vs_skl()
    test_gb_vs_skl_on_df()
    print('all tests passed')