import sys
import os
sys.path.append(os.getcwd())
from data.data_collector import data_collector
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = data_collector("Apple Inc.")
close_prices = df.values.reshape(-1, 1) 
df=pd.DataFrame(close_prices, columns=['Close Price'])
df['Date'] = pd.date_range(start='2020-01-02', periods=len(df), freq='B')
df.set_index('Date', inplace=True)
df_sample=df[df.index >'2026-01-01']
print(df_sample)
model = ExponentialSmoothing(
    df_sample['Close Price'],
    trend='mul',      # 'add' or 'mul' (additive vs multiplicative trend)
    seasonal='mul',   # 'add' or 'mul' (additive vs multiplicative seasonality)
    seasonal_periods=5
).fit()

forecast_6m = model.forecast(steps=126)

plt.plot(df_sample.index, df_sample['Close Price'], label='Actual')
plt.plot(forecast_6m.index, forecast_6m, label='Forecast', color='red')
plt.title('Apple Inc. Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()