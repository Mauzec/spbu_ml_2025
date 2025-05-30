from adaboost import AdaBoost
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

def test_adaboost_vs_skl() :
    rng = np.random.RandomState(13)
    x1,x2 = \
        rng.randn(100,2)+np.array([2,2]), \
        rng.randn(100,2)+np.array([-2,-2])
    x = np.vstack((x1,x2))
    y = np.array([1]*100 + [0]*100)
    
    p = rng.permutation(len(y))
    x,y = x[p], y[p]
    
    my_clf = AdaBoost(lr=1.0, n_estimators=50)
    my_clf.fit(x, y)
    my_pr = my_clf.predict(x)
    my_acc = np.mean(my_pr == y)
    
    sk_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME'
    )
    sk_clf.fit(x, y)
    sk_pr = sk_clf.predict(x)
    sk_acc = np.mean(sk_pr == y)
    
    if abs(my_acc - sk_acc) > 0.05:
        raise AssertionError(
            f'acc mismatch: my_acc={my_acc}, sk_acc={sk_acc}'
        )
    print('test_adaboost_vs_skl passed')
        
        
        
def test_adaboost_vs_skl_on_df():
    df = pd.read_csv('../../../data/breast-cancer-wisconsin.csv')
    y = df['diagnosis'].map({'M': 1, 'B': 0}).values
    X = df.drop(columns=['id', 'diagnosis']).values
    X_train,X_test,y_train,y_test = \
        train_test_split(X,y,test_size=.3,random_state=41,stratify=y)
    my_clf = AdaBoost(lr=.666, n_estimators=55)
    my_clf.fit(X_train, y_train)
    my_acc = np.mean(my_clf.predict(X_test) == y_test)
    
    sk_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=55,
        learning_rate=.666,
        algorithm='SAMME'
    )
    sk_clf.fit(X_train, y_train)
    sk_acc = np.mean(sk_clf.predict(X_test) == y_test)
    
    if abs(my_acc - sk_acc) > 0.05:
        raise AssertionError(
            f'acc mismatch: my_acc={my_acc}, sk_acc={sk_acc}'
        )
    print(f'...[test_adaboost_vs_skl_on_df] my_acc={my_acc}, sk_acc={sk_acc}')
    print('test_adaboost_vs_skl_on_df passed')
    
if __name__ == '__main__':
    test_adaboost_vs_skl()
    test_adaboost_vs_skl_on_df()
    
    


        