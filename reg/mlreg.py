#%%
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]
#%%
df = pd.read_csv("D:\\GitRepo\\Blog\\rus2.csv", sep=",")

dt = df.select_dtypes(include=np.number)
clean_dataset(dt)
dt.head(6)

y = dt["price_usd"].values
x = dt.drop(["price_usd"], axis=1)


#%%
# x = sm.add_constant(x)
import statsmodels.api as sm

reg1 = sm.OLS(endog = y, exog = x).fit()
reg1.summary()

x2 = ['total_time','wait_time','surge_multiplier','driver_gender','distance_kms' ]
x = sm.add_constant(x)
reg2 = sm.OLS(endog = y, exog = x[x2]).fit()
reg2.summary()

x3 = ['wait_time','surge_multiplier','driver_gender','distance_kms' ]
# x = sm.add_constant(x)
reg3 = sm.OLS(endog = y, exog = x[x3]).fit()
reg3.summary()

#%%
from statsmodels.iolib.summary2 import summary_col
dict={'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[reg1,reg2,reg3], float_format='%0.3f',
                            stars = True, model_names=['Model 1',
                                         'Model 2',
                                         'Model 3'],
                            info_dict=dict, regressor_order= ['surge_multiplier','driver_gender','distance_kms','total_time','wait_time'])

results_table.add_title('Table 2 - OLS Regressions')
print(results_table)

#%%
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML

ht = Stargazer([reg1, reg2, reg3])
HTML(ht.render_html())

#%%
## PLOT

fix, ax = plt.subplots()
ax.scatter(dt['distance_kms'], reg3.predict(), alpha=0.5, label='predicted')
# Plot observed values
ax.scatter(dt['distance_kms'], dt['price_usd'], alpha=0.5, label='observed')

ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('distance_kms')
ax.set_ylabel('price_usd')
plt.show()
# %%
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_u, il_v = wls_prediction_std(reg3)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x.distance_kms, y, 'o', label="data")

ax.scatter(dt['distance_kms'], reg3.predict(), label='predicted', color='green',  s=100)
ax.plot(x.distance_kms, reg3.fittedvalues, 'r--.', label="OLS", alpha=0.35)

ax.legend(loc='best')
ax.set_title('OLS predicted values')
ax.set_xlabel('distance_kms')
ax.set_ylabel('price_usd')
plt.show()