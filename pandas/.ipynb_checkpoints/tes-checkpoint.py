import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier #
from sklearn.model_selection import train_test_split # function
from sklearn.metrics import classification_report, confusion_matrix


data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

y = data_train['target']
X = data_train.drop('target', axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('The shape of X dataset', X.shape[0])
print('The shape of X_train', X_train.shape[0])


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

## CORRELATION MATRIX

corrMatrix = data_train.corr()
print(corrMatrix['target'].sort_values(ascending=False))

# housing_A153                    0.098238
# present_employment_A72          0.108264
# checking_account_status_A12     0.116657
# credit_history_A31              0.135735
# property_A124                   0.148996
# credit_amount                   0.168680
# duration                        0.223492

fp = []

for i in X_train[['housing_A153','duration','age']]:
    df = X_train.iloc[np.random.permutation(X_train[i].values)]
    # df.head(4)
    clf = DecisionTreeClassifier()
    clf = clf.fit(df,y_train)
#Predict the response for test dataset
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm[1,0])
    fp.append(cm[1,0])

fp






