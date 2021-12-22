import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests


df_data = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
df_dose = pd.read_csv("https://raw.githubusercontent.com/nychealth/covid-vaccine-data/main/doses/doses-by-day.csv")



df_ = df_data[["CASE_COUNT", "HOSPITALIZED_COUNT", "DEATH_COUNT" ]][df_dose.shape[0]:]
df = df_.reset_index(drop=True)
df_dose_cumulative = df_dose[["ADMIN_DOSE2_CUMULATIVE"]][-df.shape[0]:]
df_dose_cumulative = df_dose_cumulative.reset_index(drop=True)

df_final = pd.concat([df, df_dose_cumulative.reindex(df.index)], axis=1)

test_size = 20
df_train = df_final[:-test_size]
df_test =df_final[-test_size:]
df_differenced = df_train.diff().dropna()
df_differenced = df_differenced.diff().dropna()
model = VAR(df_differenced)
model_fitted = model.fit(8)
forecast_input = df_differenced.values[-8:]

fc = model_fitted.forecast(y=forecast_input, steps=test_size)
df_forecast = pd.DataFrame(fc, index=df_final.index[-test_size:], columns=df_final.columns + '_2d')

def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)    
case_list = df_results['CASE_COUNT_forecast']
textfile = open("case_count_forecast_20.txt", "w")
for element in case_list:
    textfile.write(str(int(element)) + "\n")
textfile.close()