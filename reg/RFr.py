import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]


df = pd.read_csv("E://Gitrepo//Blog//rus2.csv", sep=",")

dt = df.select_dtypes(include=np.number)
clean_dataset(dt)
dt.head(6)

y = dt["price_usd"].values
x = dt.drop(["price_usd"], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

reg1 = RandomForestRegressor(n_estimators = 300)
reg1.fit(x_train, y_train)

pred1 = reg1.predict(x_test)
mse1 = round(mean_squared_error(y_test, pred1),2)
r1 = round(r2_score(y_test, pred1),2)
print(mse1, r1)

## N = 1000
reg2 = RandomForestRegressor(n_estimators = 1000)
reg2.fit(x_train, y_train)

pred2 = reg2.predict(x_test)
mse2 = round(mean_squared_error(y_test, pred2),2)
r2 = round(r2_score(y_test, pred2),2)
print(mse2, r2)

## N = 100
reg3 = RandomForestRegressor(n_estimators = 100)
reg3.fit(x_train, y_train)

pred3 = reg3.predict(x_test)
mse3 = round(mean_squared_error(y_test, pred3),2)
r3 = round(r2_score(y_test, pred3),2)
print(mse3, r3)



# Plot the impurity-based feature importances of the forest

feature_list = list(x.columns)
importances = list(reg3.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

t = df = pd.DataFrame(feature_importances,columns =['Name', 'val'])

plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), t.val, color="b", align="center")
plt.xticks(range(x.shape[1]), t.Name, fontsize=8, rotation=25)
plt.xlim([-1, x.shape[1]])
plt.show()


lw = 2

svrs = [reg1, reg2, reg3]
kernel_label = ['300', '1000', '100']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].scatter(x_train.iloc[:,5], svr.fit(x_train, y_train).predict(x_train), color=model_color[ix], lw=lw,
                  label='{} trees'.format(kernel_label[ix]))
    axes[ix].scatter(x_train.iloc[:,5], y_train, facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} predictions'.format(kernel_label[ix]))
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()



