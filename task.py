"""
Author: Nakka Manikanta 
Task: JPMorgan Natural Gas Storage Contract Pricing
Description: This script loads monthly gas prices, interpolates past data,
extrapolates for one year using STL decomposition, and provides price estimates
for any date between Oct 2020 and Sept 2025.
"""



import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import STL
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv("Nat_Gas.csv")
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df.set_index('Dates', inplace=True)

# Step 2: Interpolation
daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
interp_func = interp1d(df.index.astype(np.int64), df['Prices'], kind='cubic', fill_value='extrapolate')

# Step 3: STL decomposition for trend and seasonality
stl = STL(df['Prices'], period=12)
res = stl.fit()

# Step 4: Extrapolation to next 12 months
future_months = pd.date_range(df.index[-1] + MonthEnd(1), periods=12, freq='ME')
trend = res.trend
last_trend_slope = (trend.iloc[-1] - trend.iloc[-13]) / 12
future_trend = [trend.iloc[-1] + last_trend_slope * i for i in range(1, 13)]
seasonal_cycle = res.seasonal[-12:]
future_seasonal = seasonal_cycle.values.tolist()
future_prices = np.array(future_trend) + np.array(future_seasonal)
future_price_dict = dict(zip(future_months.strftime('%Y-%m-%d'), future_prices))

# Step 5: Define estimation function
def estimate_gas_price(date_str: str) -> float:
    target_date = pd.to_datetime(date_str)
    if target_date < df.index.min():
        raise ValueError("Date is earlier than available data (Oct 2020).")
    elif target_date <= df.index.max():
        return float(interp_func(target_date.to_datetime64().astype(np.int64)))
    elif target_date.strftime('%Y-%m-%d') in future_price_dict:
        return float(future_price_dict[target_date.strftime('%Y-%m-%d')])
    else:
        raise ValueError("Date is beyond extrapolation range (Sep 2025).")

# Step 6: Test the function
print("Estimated price on 2021-03-15:", estimate_gas_price("2021-03-15"))
print("Estimated price on 2025-06-30:", estimate_gas_price("2025-06-30"))

# Step 7: Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Prices'], label='Original', marker='o')
plt.plot(future_months, future_prices, label='Extrapolated', marker='x', linestyle='--')
plt.title("Natural Gas Prices: Historical and 1-Year Extrapolation")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
