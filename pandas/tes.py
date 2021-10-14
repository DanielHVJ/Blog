import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.tree import DecisionTreeClassifier #
from sklearn.model_selection import train_test_split # function
from sklearn.metrics import confusion_matrix
from random import randint
import matplotlib.pyplot as plot
from matplotlib import pyplot as plt 

data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

## MODEL ON DATA TRAIN

def uplo(db):
    y = db['target']
    X = db.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('The shape of X dataset', X.shape[0])
    print('The shape of X_train', X_train.shape[0])
    return X_train,X_test,y_train,y_test

X_train, X_test, y_train, y_test = uplo(data_train)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm
## CORRELATION MATRIX
corrMatrix = data_train.corr()
print(corrMatrix['target'].sort_values(ascending=False))

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

# Calling DataFrame constructor after zipping
lt = pd.DataFrame(list(zip(re, fp)),columns =['Name', 'CM val'])
lt

xg = lt['CM val']

fig, ax = plt.subplots(figsize = (7, 7))
ax.bar(re, xg)
plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=8)
plt.title('Train data: Selected variables')
plt.show()

## MODEL ON DATA TEST
y = data_test['target']
X = data_test.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('The shape of X dataset', X.shape[0])
print('The shape of X_train', X_train.shape[0])

X_test.shape

X_train, X_test, y_train, y_test = uplo(data_test)

fpt = []
stree(fpt,X_train,X_test,y_train,y_test)

ret = nam.copy()
ret.append('TEST')

# Calling DataFrame constructor after zipping
# xg = lt['CM val']

fig, ax = plt.subplots(figsize = (7, 7))
plt.bar(ret, fpt)
plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=8)
plt.title('Test data: Selected variables')
plt.show()
