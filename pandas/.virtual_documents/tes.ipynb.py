import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import randint
from matplotlib import pyplot as plt
import seaborn as sns


data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')


data_train.head(6)


def uplo(db):
    y = db['target']
    X = db.drop('target', axis=1)
    xt, xte, yt, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    print('The shape of dataset:', X.shape[0])
    print('The shape of the X dataset:', xt.shape[0])
    
    return xt,xte,yt,yte


X_train, X_test, y_train, y_test = uplo(data_train)


corrMatrix = data_train.corr()

print(corrMatrix['target'][:10].sort_values(ascending=False))


col = ['target','duration'               ,        
'credit_amount'                 ,
'checking_account_status_A12'   , 
'installment_rate'              , 
'present_residence'             , 
'dependents'                    ,
'existing_credits'             ,
'checking_account_status_A13'  , 
'age', 'checking_account_status_A14']


corr = data_train[col].corr()

ax = sns.heatmap(  corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 200, n=200),
    square=True)

ax.set_xticklabels(ax.get_xticklabels(),
    rotation=45, horizontalalignment='right');


fp = []
nu = [randint(2, 49) for p in range(2, 7)]
nam = list(X_train.columns[nu])
nam.extend(['duration','age'])
nam


def stree(fl,xt,xte,yt,yte):
    for i in xt.loc[:,nam]:
        df = xt.iloc[np.random.permutation(xt[i].values)]
        clf = DecisionTreeClassifier()
        clf = clf.fit(df,yt)
        
        #Predict the response for test dataset
        y_pred = clf.predict(xte)
        cm = confusion_matrix(yte, y_pred)
        print(cm[1,0])
        fl.append(cm[1,0])
   
    clf = clf.fit(xt,yt)
    y_base = clf.predict(xte)
    cm = confusion_matrix(yte, y_base)
    fl.append(cm[1,0])
    return fl


stree(fp,X_train,X_test,y_train,y_test)


re = nam.copy()
re.append('BASE')


fig, ax = plt.subplots(figsize = (9, 6))
ax.bar(re, fp)
plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=8)
plt.title('Train data: Selected variables')
plt.show()


X_train, X_test, y_train, y_test = uplo(data_test)


fpt = []
stree(fpt,X_train,X_test,y_train,y_test)


ret = nam.copy()
ret.append('TEST')


fig, ax = plt.subplots(figsize = (9, 6))
plt.bar(ret, fpt)
plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=8)
plt.title('Test data: Selected variables')
plt.show()


lt = pd.DataFrame(list(zip(re, fp)),columns =['Name', 'CM val'])
lt.index = lt.Name
lt.drop(columns='Name', inplace=True)

lt


Dict = {'fp': [fp[-1]],
        'most_important' : lt.loc[["purpose_A48", "age"]].to_dict(),
        'fp_most_important' : lt['CM val'].min()}
