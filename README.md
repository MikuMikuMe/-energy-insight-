# **energy-insight**

Certainly! Below is a Python program that simulates the `energy-insight` project, focusing on optimizing household energy consumption using real-time monitoring and predictive analytics. This program includes components for data collection, prediction using a machine learning model, and error handling.

```python
import random
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Simulate real-time data collection
class RealTimeDataCollector:
    def __init__(self):
        self.data = []

    def collect_data(self):
        # Simulated data: (timestamp, power_usage_in_kw)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        power_usage = round(random.uniform(0.5, 5.0), 2)  # Random power usage between 0.5kw and 5.0kw
        data_point = {"timestamp": current_time, "power_usage_kw": power_usage}
        self.data.append(data_point)
        print(f"Data collected: {data_point}")

    def get_data(self):
        return pd.DataFrame(self.data)


# Predictive Analytics using a simple Linear Regression Model
class EnergyPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def train(self, data):
        try:
            # Convert timestamps to ordinal
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['timestamp_ordinal'] = data['timestamp'].map(datetime.datetime.toordinal)

            # Define features and labels
            X = data[['timestamp_ordinal']]
            y = data['power_usage_kw']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            self.model.fit(X_train, y_train)
            self.trained = True

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Model trained. Mean Squared Error on test data: {mse:.2f}")

        except Exception as e:
            print(f"Error during model training: {e}")

    def predict(self, future_times):
        try:
            if not self.trained:
                raise Exception("Model is not trained yet.")

            # Convert future_times to ordinal
            future_times = [pd.to_datetime(ft).toordinal() for ft in future_times]
            X_future = np.array(future_times).reshape(-1, 1)

            # Predict future power usage
            predictions = self.model.predict(X_future)
            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            return []


def main():
    # Create instances of data collector and predictor
    data_collector = RealTimeDataCollector()
    energy_predictor = EnergyPredictor()

    try:
        # Simulate data collection
        for _ in range(10):  # Collect 10 data points
            data_collector.collect_data()

        # Retrieve and print the collected data
        collected_data = data_collector.get_data()
        print("Collected Data:\n", collected_data)

        # Train the predictive model
        energy_predictor.train(collected_data)

        # Make predictions for the next 5 days
        future_dates = [(datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]
        predictions = energy_predictor.predict(future_dates)

        # Display prediction results
        for date, prediction in zip(future_dates, predictions):
            print(f"Predicted power usage for {date}: {prediction:.2f} kW")

    except Exception as e:
        print(f"An error occurred in the energy insight tool: {e}")

if __name__ == "__main__":
    main()
```

### Explanation

- **RealTimeDataCollector**: Simulates real-time data collection for household energy consumption.
  
- **EnergyPredictor**: Implements a simple linear regression model for predictive analytics. It handles model training and making predictions based on future time points.

- **Error Handling**: Both the `train` and `predict` methods in the `EnergyPredictor` class include error handling to manage exceptions that could occur during these processes.

- **Main Function**: Orchestrates data collection, model training, and prediction, also handling any exceptions that might arise in the process.

This program is a simple conceptual framework and a starting point. In a real-world application, data collection would involve interfacing with smart meters or energy-use sensors, more sophisticated predictive modeling, possibly improved with machine learning techniques such as ARIMA, LSTM networks, or others specialized for time-series data.