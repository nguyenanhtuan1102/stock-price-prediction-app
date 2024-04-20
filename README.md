# Stock Price Prediction Application

## Overview

## Usage

## Key Dependencies

## Data Preparation

### 1. Choose a Timeline

``` bash
# From 01-01-2021 to today
end_date = date.today()
start_date = end_date - reladate(years=3)
start_date = date(start_date.year,1,1)
end_date = end_date.strftime('%Y-%m-%d')
start_date = start_date.strftime('%Y-%m-%d')
```

### 2. Load Dataset

``` bash
scode = ['HPG'][0]
df_hpg = stock_historical_data(symbol=scode, start_date=start_date, end_date=end_date)
df_hpg.insert(0, column='scode', value=scode)
df_hpg = df_hpg.set_index('time')
df_hpg.index = pd.to_datetime(df_hpg.index)
```

### 3. Data Information

``` bash
df_hpg.info()
```

![info](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/ccc4742b-e015-49d9-86dc-da08628c01fb)

### 4. Describe Data

``` bash
df_hpg.describe()
```

![des](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/d52ad9eb-01a9-49a0-96a3-2692e49deacd)

### 5. Visualize HPG Stock

``` bash
scode = 'HPG'
df = df_hpg[df_hpg['scode'] == scode]

average_close = df['close'].mean()

fig = msplt(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price'), secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(window=90).mean(), marker_color='orange', name='90 Day MvA'))
fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), secondary_y=True)

fig.add_shape(
    type="line",
    x0=df.index.min(),
    y0=average_close,
    x1=df.index.max(),
    y1=average_close,
    line=dict(
        color="red",
        width=2,
        dash="dash"),
        name='3y avg'
)

fig.update_layout(title={'text': scode, 'x': 0.5})
fig.update_yaxes(range=[0, 100000000], secondary_y=True, visible=False)
fig.update_layout(xaxis_rangeslider_visible=False)
# Hiển thị biểu đồ
fig.show()
```

![hpg](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/cfff881d-e6c7-41b3-9caf-16dd72fe5fe9)

### 6. Train-test split

``` bash
train_data, test_data = train_test_split(df_hpg, test_size=0.3, random_state=42, shuffle=False)
```

## Modelling

### 1. Choose the Suitable Model With Auto Arima

``` bash
model = auto_arima(train_data["close"],
                   seasonal=False,
                   m=5,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   with_intercept="False",
                   d=0)
fitted = model.fit(train_data["close"])
print(fitted)
```

![choose](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/cf9b22c8-eed0-48b0-b3de-5d0848cad28a)

### 2. Training Model

``` bash
# training model
model = ARIMA(train_data["close"], order=(1, 0, 2), seasonal_order=(0, 0, 0, 0))
arima = model.fit()

# create index
index_range = test_data.index

# predict on test-set
forecast_arima = arima.forecast(steps=len(test_data))
forecast_arima.index = index_range
```

![pre](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/039fab47-dedb-4ed8-8886-7233eeef75c7)

### 3. Compare Predict Values with Past Values

``` bash
# Split plot to 2 part
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 2
ax1.grid(True)
ax1.set_xlabel('Dates')
ax1.set_ylabel('Closing Prices')
ax1.plot(train_data['close'], 'green', label='Train data')
ax1.plot(test_data['close'], 'blue', label='Test data')
ax1.plot(test_data.index, forecast_arima, 'red', label='Forecast')
ax1.set_title('Hoa Phat Group Stock Price Prediction')
ax1.legend()

# Plot 2
ax2.set_xlabel('Dates')
ax2.set_ylabel('Closing Prices')
ax2.plot(test_data['close'], 'blue', label='Test data')
ax2.plot(test_data.index, forecast_arima, 'red', label='Forecast')
ax2.set_title('Comparison of Predicted vs True Data')
ax2.legend()
ax2.grid(True)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
```

![tải xuống](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/7734f108-b3d0-47e2-896c-2673d6e10872)

### 4. Model Evaluation

``` bash
forecast_arima = pd.DataFrame(forecast_arima)
MSE = mean_squared_error(test_data['close'], forecast_arima['predicted_mean'])
print('MSE:',MSE)
```

![mse](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/60a1f5f0-59b4-4c68-8861-51bfc0d5f30f)

### 5. Predict For Next 30 Days

``` bash
# create index
start_next_index = df_hpg.index[-1] + timedelta(days=1)
end_next_index = start_next_index + + timedelta(days=30)
index_next_range = pd.date_range(start=start_next_index, end=end_next_index)

# predict for next 30 days
start_next = len(test_data) + len(train_data)
end_next = len(test_data) + len(train_data) + 30
forecast = arima.predict(start=start_next, end=end_next)
forecast.index = index_next_range
```

![pred](https://github.com/tuanng1102/stock-price-prediction-app/assets/147653892/c47fb181-323f-4ee2-a6b1-6822a66c2e33)

## Web Application

