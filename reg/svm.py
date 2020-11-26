import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]


df = pd.read_csv("rus2.csv", sep=",")

dt = df.select_dtypes(include=np.number)
clean_dataset(dt)
dt.head(6)

y = dt["price_usd"].values
x = dt.drop(["price_usd"], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.fit_transform(y_test.reshape(-1,1))

from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error

reg1 = SVR(kernel = "rbf", C=100, gamma=0.1, epsilon=.1)
reg1.fit(x_train, y_train.ravel())

pred1 = reg1.predict(x_test)
mse1 = round(mean_squared_error(y_test, pred1),2)
r1 = round(r2_score(y_test, pred1),2)

reg2 = SVR(kernel = "linear", C=100, gamma='auto')
reg2.fit(x_train, y_train.ravel())

pred2 = reg2.predict(x_test)
mse2 = round(mean_squared_error(y_test, pred2),2)
r2 = round(r2_score(y_test, pred2),2)

reg3 = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1, coef0=1)
reg3.fit(x_train, y_train.ravel())

pred3 = reg3.predict(x_test)
mse3 = round(mean_squared_error(y_test, pred3),2)
r3 = round(r2_score(y_test, pred3),2)


from prettytable import PrettyTable
t = PrettyTable()
t.field_names = ["Metrics","RBF", "Linear", "Poly(2)"]
t.add_row(['R-score', r1, r2, r3])
t.add_row(['MSE', mse1,mse2,mse3])
print(t)


# x_grid = np.arange(min(x[:,5]), max(x[:,5]), 0.1)
x_grid = x[:,5].reshape(len(x), 1)
plt.scatter(x[:,5], y, color = "red")
plt.scatter(x[:,5], reg1.predict(x), color = "blue")
plt.title("Regression model (SVR)")
plt.xlabel("Distance")
plt.ylabel("Price")
plt.show()


lw = 2

svrs = [reg1, reg2, reg3]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].scatter(x_train[:,5], svr.fit(x_train, y_train.ravel()).predict(x_train), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(x_train[svr.support_][:,5], y_train[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(x_train[np.setdiff1d(np.arange(len(x_train)), svr.support_)][:,5],
                     y_train[np.setdiff1d(np.arange(len(x_train)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
