"""
Author: Nakka Manikanta
JP Morgan Quantitative Research – Natural Gas Contract Pricing
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import STL
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
# Load historical gas price data
df = pd.read_csv("Nat_Gas.csv")
df['Dates'] = pd.to_datetime(df['Dates'], format="%m/%d/%y")
df.set_index('Dates', inplace=True)

# Interpolate daily prices
daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
interp_func = interp1d(df.index.astype(np.int64), df['Prices'], kind='cubic', fill_value='extrapolate')

# STL decomposition for extrapolation
stl = STL(df['Prices'], period=12)
res = stl.fit()
trend = res.trend

# Extend trend linearly
future_months = pd.date_range(df.index[-1] + MonthEnd(1), periods=12, freq='ME')
last_trend_slope = (trend.iloc[-1] - trend.iloc[-13]) / 12
future_trend = [trend.iloc[-1] + last_trend_slope * i for i in range(1, 13)]

# Add seasonal pattern
seasonal_cycle = res.seasonal[-12:]
future_seasonal = seasonal_cycle.values.tolist()
future_prices = np.array(future_trend) + np.array(future_seasonal)
future_price_dict = dict(zip(future_months.strftime('%Y-%m-%d'), future_prices))
def estimate_gas_price(date_str: str) -> float:
    target_date = pd.to_datetime(date_str)

    # Convert to nanoseconds for interpolation
    all_dates = df.index.tolist()
    all_prices = df['Prices'].tolist()

    # Extend dates with future extrapolated values
    for d, p in future_price_dict.items():
        all_dates.append(pd.to_datetime(d))
        all_prices.append(p)

    # Sort by date
    combined = sorted(zip(all_dates, all_prices))
    combined_dates = [d for d, _ in combined]
    combined_prices = [p for _, p in combined]

    # Interpolation
    interp_all = interp1d(
        pd.to_datetime(combined_dates).astype(np.int64),
        combined_prices,
        kind='cubic',
        fill_value='extrapolate'
    )

    if target_date < combined_dates[0] or target_date > combined_dates[-1]:
        raise ValueError("Date is beyond data range.")

    return float(interp_all(target_date.to_datetime64().astype(np.int64)))
def price_gas_storage_contract(
    injection_schedule,
    withdrawal_schedule,
    injection_rate_limit,
    withdrawal_rate_limit,
    max_storage_capacity,
    storage_cost_per_day
) -> float:
    all_dates = pd.date_range(
        start=min(min(pd.to_datetime([d for d, _ in injection_schedule]), default=pd.Timestamp.today()),
                  min(pd.to_datetime([d for d, _ in withdrawal_schedule]), default=pd.Timestamp.today())),
        end=max(max(pd.to_datetime([d for d, _ in injection_schedule]), default=pd.Timestamp.today()),
                max(pd.to_datetime([d for d, _ in withdrawal_schedule]), default=pd.Timestamp.today()))
    )

    storage = 0.0
    total_cost = 0.0
    total_revenue = 0.0
    storage_cost_total = 0.0

    inject_dict = {pd.to_datetime(d): v for d, v in injection_schedule}
    withdraw_dict = {pd.to_datetime(d): v for d, v in withdrawal_schedule}

    for day in all_dates:
        if day in inject_dict:
            volume = inject_dict[day]
            if volume > injection_rate_limit:
                raise ValueError(f"Injection rate exceeded on {day.date()}")
            if storage + volume > max_storage_capacity:
                raise ValueError(f"Storage capacity exceeded on {day.date()}")
            price = estimate_gas_price(str(day.date()))
            total_cost += price * volume
            storage += volume

        if day in withdraw_dict:
            volume = withdraw_dict[day]
            if volume > withdrawal_rate_limit:
                raise ValueError(f"Withdrawal rate exceeded on {day.date()}")
            if volume > storage:
                raise ValueError(f"Insufficient gas in storage to withdraw on {day.date()}")
            price = estimate_gas_price(str(day.date()))
            total_revenue += price * volume
            storage -= volume

        storage_cost_total += storage * storage_cost_per_day

    net_value = total_revenue - total_cost - storage_cost_total
    return net_value

"""
here is the sample input for to the test case :
if __name__ == "__main__":
    injections = [("2024-07-15", 200000), ("2024-08-15", 300000)]
    withdrawals = [("2025-01-15", 250000), ("2025-02-15", 250000)]

    contract_value = price_gas_storage_contract(
        injection_schedule=injections,
        withdrawal_schedule=withdrawals,
        injection_rate_limit=300000,
        withdrawal_rate_limit=300000,
        max_storage_capacity=500000,
        storage_cost_per_day=0.01
    )

    print(f"Estimated contract value: ${contract_value:,.2f}")

"""
