#importing and data checking
import pandas as pd

file_path = r"C:\Users\bonif\Desktop\southern indian\southern_india_daily_electricity.csv"
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d-%m-%Y")

print("Rows and columns:", df.shape)
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())
print("Duplicate dates:", df["date"].duplicated().sum())
print("Missing supply readings:", df["SR: EnergyMet"].isna().sum())

print("\nFirst five rows:")
print(df[["date", "SR: EnergyMet"]].head())

#-----------------------------------------------------------------------------------------
#taking data from 2024 to 2026

areas = [
    "Andhra Pradesh: EnergyMet",
    "Karnataka: EnergyMet",
    "Kerala: EnergyMet",
    "Tamil Nadu: EnergyMet",
    "Telangana: EnergyMet",
    "Puducherry: EnergyMet",
]

columns = ["date", "SR: EnergyMet"] + areas
study = df.loc[df["date"] >= "2024-01-01", columns].copy()
study = study.sort_values("date").reset_index(drop=True)

# Add the six areas. If any area is missing, the total stays missing.
study["six_area_total_gwh"] = study[areas].sum(axis=1, min_count=6)

print("Number of days:", len(study))
print("Date range:", study["date"].min().date(), "to", study["date"].max().date())

print("\nMissing readings:")
print(study[columns[1:]].isna().sum())

print("\nPublished and calculated totals for the last five days:")
print(study[["date", "SR: EnergyMet", "six_area_total_gwh"]].tail())

#------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

total = study["six_area_total_gwh"]

# 1. Daily supply and its average over the most recent 30 days
plt.figure(figsize=(11, 4))
plt.plot(study["date"], total, alpha=0.35, label="Daily supply")
plt.plot(study["date"], total.rolling(30).mean(), label="30-day average")
plt.title("Daily Electricity Supply Across the Six Areas")
plt.ylabel("Electricity supplied (GWh)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Average supply for each weekday
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

weekday_average = total.groupby(study["date"].dt.day_name()).mean()
weekday_average = weekday_average.reindex(weekdays)

weekday_average.plot(kind="bar", figsize=(9, 4), rot=0)
plt.title("Average Electricity Supply by Weekday")
plt.ylabel("Electricity supplied (GWh)")
plt.tight_layout()
plt.show()

# 3. Compare the six areas
area_average = study[areas].mean().sort_values()
area_average.index = area_average.index.str.replace(": EnergyMet", "", regex=False)

area_average.plot(kind="barh", figsize=(8, 4))
plt.title("Average Daily Electricity Supply by Area")
plt.xlabel("Electricity supplied (GWh)")
plt.tight_layout()
plt.show()

#model forcast
from statsmodels.tsa.statespace.sarimax import SARIMAX

# One electricity reading per day, indexed by date
supply = study.set_index("date")["six_area_total_gwh"].asfreq("D")

# Fit the model on the earlier days and test on the last 90 days
train = supply.iloc[:-90]
test = supply.iloc[-90:]

# Forecast 1: use the same weekday from last week
weekly_forecast = supply.shift(7).loc[test.index]

# Forecast 2: fit a starting SARIMA model with a seven-day cycle
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
fitted = model.fit(disp=False)

sarima_predictions = []

for i in range(len(test)):
    # Predict one day before seeing its actual value
    prediction = fitted.forecast(steps=1).iloc[0]
    sarima_predictions.append(prediction)

    # Add today's actual reading before predicting the next day
    fitted = fitted.append(test.iloc[i:i + 1], refit=False)

results = pd.DataFrame({
    "Actual": test,
    "Weekly naive": weekly_forecast,
    "SARIMA": sarima_predictions
})

weekly_mae = (results["Actual"] - results["Weekly naive"]).abs().mean()
sarima_mae = (results["Actual"] - results["SARIMA"]).abs().mean()
improvement = 100 * (weekly_mae - sarima_mae) / weekly_mae

print("Test period:", test.index[0].date(), "to", test.index[-1].date())
print(f"Weekly naive MAE: {weekly_mae:.2f} GWh")
print(f"SARIMA MAE:       {sarima_mae:.2f} GWh")
print(f"SARIMA improvement over baseline: {improvement:.2f}%")

print("\nLast five days:")
print(results.tail().round(2))

#-------------------------------------------------------------------------------
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

area_data = study.set_index("date")[areas].asfreq("D")
test_dates = results.index
area_forecast = pd.DataFrame(index=test_dates)

for area in areas:
    print("Forecasting", area)

    # Train using readings before the first test day
    train_area = area_data.loc[area_data.index < test_dates[0], area]
    test_area = area_data.loc[test_dates, area]

    model = SARIMAX(
        train_area,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    ).fit(disp=False)

    predictions = []

    for i in range(len(test_area)):
        predictions.append(model.forecast(steps=1).iloc[0])
        model = model.append(test_area.iloc[i:i + 1], refit=False)

    area_forecast[area] = predictions

# Add the six area predictions for each day
results["Bottom-up SARIMA"] = area_forecast.sum(axis=1, min_count=6)

comparison = results[["Actual", "SARIMA", "Bottom-up SARIMA"]].copy()
comparison = comparison.rename(columns={"SARIMA": "Direct SARIMA"})
comparison["Forecast gap"] = (
    comparison["Bottom-up SARIMA"] - comparison["Direct SARIMA"]
)

print("\nLast five test days (GWh)")
print(comparison.tail().round(2).to_string())

errors = comparison[["Direct SARIMA", "Bottom-up SARIMA"]].sub(
    comparison["Actual"], axis=0
).abs()

print("\nAverage forecast error over the 90 test days (GWh)")
print(errors.mean().round(2))

#minT

import numpy as np

total_name = "six_area_total_gwh"
series_names = [total_name] + areas

# 1. Recreate the seven SARIMA models on the training period
training = study.set_index("date")[series_names].asfreq("D")
training = training.loc[training.index < results.index[0]]

training_errors = pd.DataFrame(index=training.index)

for name in series_names:
    print("Estimating training errors for", name)

    fitted = SARIMAX(
        training[name],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    ).fit(disp=False)

    training_errors[name] = fitted.resid

# Skip the first 30 days, while the models are initializing
training_errors = training_errors.iloc[30:].dropna()
error_covariance = training_errors.cov().to_numpy()

# 2. Collect the forecasts you already made for the 90 test days
base = area_forecast[areas].copy()
base[total_name] = results["SARIMA"]
base = base[series_names]

# 3. Adjust the forecasts so total minus sum of areas equals zero
constraint = np.array([1] + [-1] * len(areas))
gap = base.to_numpy() @ constraint

weights = error_covariance @ constraint
weights = weights / (constraint @ weights)

mint = pd.DataFrame(
    base.to_numpy() - np.outer(gap, weights),
    index=base.index,
    columns=series_names
)

results["MinT SARIMA"] = mint[total_name]

# 4. Check that the adjusted forecasts add up
remaining_gap = mint[total_name] - mint[areas].sum(axis=1)
print("\nLargest gap after MinT (GWh):", round(remaining_gap.abs().max(), 8))

# 5. Compare average errors for the same regional total
models = ["SARIMA", "Bottom-up SARIMA", "MinT SARIMA"]
mae = results[models].sub(results["Actual"], axis=0).abs().mean()

print("\nAverage test error for the six-area total (GWh):")
print(mae.rename(index={"SARIMA": "Direct SARIMA"}).round(2))

# 6. See which areas gained or lost accuracy
area_actual = study.set_index("date").loc[mint.index, areas]

area_comparison = pd.DataFrame({
    "Before MinT": (area_actual - base[areas]).abs().mean(),
    "After MinT": (area_actual - mint[areas]).abs().mean()
})
area_comparison["Error reduced by"] = (
    area_comparison["Before MinT"] - area_comparison["After MinT"]
)

print("\nAverage test error by area (GWh):")
print(area_comparison.round(2))

#check the average of minT to all 90 days.

bottom_up_error = (results["Actual"] - results["Bottom-up SARIMA"]).abs()
mint_error = (results["Actual"] - results["MinT SARIMA"]).abs()

# Positive means MinT was closer to the actual supply.
daily_gain = bottom_up_error - mint_error

# Count differences smaller than 0.01 GWh as a tie.
tolerance = 0.01

print("Test days:", len(daily_gain))
print("Days MinT was closer:", (daily_gain > tolerance).sum())
print("Days bottom-up was closer:", (daily_gain < -tolerance).sum())
print("Days with almost the same error:", (daily_gain.abs() <= tolerance).sum())

print("\nDay MinT helped the most:", daily_gain.idxmax().date())
print(f"Error reduced by: {daily_gain.max():.2f} GWh")

print("\nDay bottom-up did best against MinT:", daily_gain.idxmin().date())
print(f"MinT's extra error that day: {-daily_gain.min():.2f} GWh")


# mint test to the narrow supply of the days
# Set the high-supply cutoff using only the first 963 days
cutoff = train.quantile(0.90)

# Label each test day using its actual electricity supply
high_supply = results["Actual"] >= cutoff

# Calculate the daily error for each forecast
errors = pd.DataFrame({
    "Bottom-up SARIMA": (results["Actual"] - results["Bottom-up SARIMA"]).abs(),
    "MinT SARIMA": (results["Actual"] - results["MinT SARIMA"]).abs()
})

print(f"High-supply cutoff: {cutoff:.2f} GWh")
print("High-supply test days:", high_supply.sum())
print("Other test days:", (~high_supply).sum())

print("\nAverage error by type of day (GWh):")
print(errors.groupby(high_supply.map({
    True: "High supply",
    False: "Other days"
})).mean().round(2))

#----------------------------------------------------
import matplotlib.pyplot as plt

# Chart 1: Compare actual supply with both forecasts
plt.figure(figsize=(12, 5))
plt.plot(results.index, results["Actual"], color="black",
         linewidth=2, label="Actual supply")
plt.plot(results.index, results["Bottom-up SARIMA"],
         linestyle="--", label="Bottom-up SARIMA")
plt.plot(results.index, results["MinT SARIMA"],
         linestyle=":", linewidth=2, label="MinT SARIMA")

plt.title("Actual and Forecast Electricity Supply Across Six Areas")
plt.ylabel("Daily electricity supplied (GWh)")
plt.xlabel("Test date")
plt.legend()
plt.tight_layout()
plt.show()

# Chart 2: A positive value means MinT was closer to the actual supply
bottom_up_error = (results["Actual"] - results["Bottom-up SARIMA"]).abs()
mint_error = (results["Actual"] - results["MinT SARIMA"]).abs()
daily_gain = bottom_up_error - mint_error

colors = ["seagreen" if gain >= 0 else "indianred" for gain in daily_gain]

plt.figure(figsize=(12, 4))
plt.bar(daily_gain.index, daily_gain, color=colors)
plt.axhline(0, color="black", linewidth=1)
plt.title("Daily Forecast Error Reduction from MinT")
plt.ylabel("Error reduction (GWh)")
plt.xlabel("Test date")
plt.tight_layout()
plt.show()

#------------------------------------------------------------
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

total_name = "six_area_total_gwh"
series_names = [total_name] + areas

history = study.set_index("date")[series_names].asfreq("D")
next_date = history.index[-1] + pd.Timedelta(days=1)

base_predictions = []
model_errors = pd.DataFrame(index=history.index)

# Fit one SARIMA model for the total and one for each area
for name in series_names:
    fitted = SARIMAX(
        history[name],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    ).fit(disp=False)

    base_predictions.append(fitted.forecast(steps=1).iloc[0])
    model_errors[name] = fitted.resid

# Use the models' past errors to share the MinT adjustment
covariance = model_errors.iloc[30:].dropna().cov().to_numpy()
constraint = np.array([1] + [-1] * len(areas))
base = np.array(base_predictions)

gap = base @ constraint
weights = covariance @ constraint
adjusted = base - gap * weights / (constraint @ weights)

forecast = pd.Series(
    adjusted,
    index=["Six-area total"] + [area.split(":")[0] for area in areas],
    name="Forecast (GWh)"
)

print("Forecast date:", next_date.date())
print(forecast.round(2).to_string())

#import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Forecast 30 days after the last actual reading
days_ahead = 30
forecast_dates = pd.date_range(
    start=history.index[-1] + pd.Timedelta(days=1),
    periods=days_ahead,
    freq="D"
)

# Forecast the combined total and each of the six areas
base_forecasts = pd.DataFrame(index=forecast_dates)

for name in series_names:
    fitted = SARIMAX(
        history[name],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    ).fit(disp=False)

    base_forecasts[name] = fitted.forecast(steps=days_ahead).to_numpy()

# Apply the MinT adjustment from Step 10 to each day
base = base_forecasts[series_names].to_numpy()
gaps = base @ constraint
adjustment = weights / (constraint @ weights)

monthly_forecast = pd.DataFrame(
    base - np.outer(gaps, adjustment),
    index=forecast_dates,
    columns=["Six-area total"] + [area.split(":")[0] for area in areas]
)
monthly_forecast.index.name = "Date"

print("Next 30 days of electricity supply forecasts (GWh)")
print(monthly_forecast.round(2).to_string())

total_for_30_days = monthly_forecast["Six-area total"].sum()
print(f"\nCombined electricity supplied over 30 days: {total_for_30_days:,.2f} GWh")