import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime, timedelta

# Read data
df = pd.read_csv(r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\src\Weekly_Average_FoodWaste.csv')

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Define the chosen category for plotting
chosen_category = 'Fiber'
# print(df[chosen_category])

# Filter the data for the chosen category
selected_columns = ['Date', chosen_category]
filtered_data = df[selected_columns]
# print(filtered_data)

split_ratio = 0.97 # 0.93 = 2 points (test data)
split_index = int(len(filtered_data) * split_ratio)
df_train = filtered_data[:split_index+1]
df_test = filtered_data[split_index:]

# Extract features and target for the current state
# X = np.arange(len(df_train)).reshape(-1, 1)
# y = df_train[chosen_category].values
# print(df_train[chosen_category])

start = filtered_data['Date'].min()
end = filtered_data['Date'].max()
range_datetime = (end - start).days

# Normalize date and waste variables
reference_date = datetime(2023, 1, 1)
normalized_date = (df_train['Date'] - reference_date).dt.days.values.reshape(-1, 1) / range_datetime
normalized_waste = df_train[chosen_category].values.reshape(-1, 1) / np.max(filtered_data[chosen_category])
# normalized_date_test = (date_test - reference_date).days / range_datetime
X = normalized_date
y = normalized_waste    
# X_test = normalized_date_test.values.reshape(-1, 1)
# X_test = np.arange(len(df)).reshape(-1, 1)
# y_test = df[chosen_category].values

# Normalize date and waste variables
normalized_date_range = (df_test['Date'] - reference_date).dt.days.values.reshape(-1, 1) / range_datetime
normalized_waste_range = df_test[chosen_category].values.reshape(-1, 1) / np.max(df_train[chosen_category])

X_train_ =  normalized_date
X_test_ =  normalized_date_range

# Define different kernels
kernel_rbf = RBF(length_scale=20 ) #+ WhiteKernel(noise_level=10)
kernel_matern = Matern(length_scale=2.4, nu=5 ) #length_scale=2, nu=1.05, sigma=1 / length_scale=2.4, nu=1.03, sigma=2 / length_scale=2.4, nu=5 , split = 0.97
kernel_combined = kernel_rbf + kernel_matern

# Create Gaussian Process with different kernels
model = GaussianProcessRegressor(kernel=kernel_combined, n_restarts_optimizer=10)

# Fit the Gaussian Processes for the current category
model.fit(X, y)
model_params = model.get_params()

#prediction for train data
start_date = df_train['Date'].min()
end_date =   df_train['Date'].max() 
date_train = pd.date_range(start=start_date, end=end_date, freq='D')

# Normalize the date range
normalized_date_train = (date_train - reference_date).days / range_datetime
X_train = normalized_date_train.values.reshape(-1, 1)

# Make predictions for the date range using the GP model
y_pred_train, sigma_range_train = model.predict(X_train, return_std=True)

# Denormalize the predicted wastes
predicted_waste_train = y_pred_train * np.max(filtered_data[chosen_category])

###predict for test data
start_dates = df_test['Date'].min()
end_dates =   df_test['Date'].max()+ timedelta(days=30)
date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')

# Normalize the date range
normalized_date_test = (date_test - reference_date).days / range_datetime
X_test = normalized_date_test.values.reshape(-1, 1)

# Make predictions for the date range using the GP model
y_pred_test, sigma_range_test = model.predict(X_test, return_std=True)

# Denormalize the predicted wastes
predicted_waste_test = y_pred_test * np.max(filtered_data[chosen_category])

# Load the saved GP model
# model_filename = f"D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\src\savedModel\({chosen_category}).pkl"
# gp_model = joblib.load(model_filename)

# Generate test data for the current category
# X_test_category = np.arange(0, len(df), 0.1)[:, np.newaxis]

# Make predictions for the current state
# y_pred_model, sigma_model = model.predict(X_test_category, return_std=True)

# Visualization
plt.plot(filtered_data['Date'], filtered_data[chosen_category], marker='o' ,c='blue', label='Actual Waste')
plt.plot(date_train, predicted_waste_train, 'r', label='Predicted Waste (Train)')
plt.plot(date_test, predicted_waste_test, c='green', markersize=8, label='Predicted Waste (Test)')
plt.fill_between(date_train.ravel(),predicted_waste_train - 2 * sigma_range_train, predicted_waste_train + 2 * sigma_range_train, alpha=0.2, color='b')
plt.fill_between(date_test.ravel(),predicted_waste_test - 2 * sigma_range_test, predicted_waste_test + 2 * sigma_range_test, alpha=0.2, color='b')
plt.xlabel('Date')
plt.ylabel(chosen_category)
plt.title(f'Prediction waste for {chosen_category}')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

y_pred_train, _ = model.predict(X_train_, return_std=True)
predicted_waste_train_ = y_pred_train * np.max(filtered_data[chosen_category])
# print(len(predicted_waste_train_),len(df_train[chosen_category]))
# Calculate MSE and RMSE for the training data
mse_train = mean_squared_error(df_train[chosen_category], predicted_waste_train_)
rmse_train = np.sqrt(mse_train)
# print("predicted waste:", predicted_waste)
print(f"Training MSE: {mse_train:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
# print(predicted_waste_train_)

y_pred_test, _ = model.predict(X_test_, return_std=True)
predicted_waste = y_pred_test * np.max(filtered_data[chosen_category])
# print(predicted_waste)
# Calculate MSE and RMSE for the training data
mse_test = mean_squared_error(df_test[chosen_category], predicted_waste)
rmse_test = np.sqrt(mse_test)
# print("predicted waste:", predicted_waste)
print(f"Testing MSE: {mse_test:.4f}")
print(f"Testing RMSE: {rmse_test:.4f}")

# Create a dictionary to store results and parameters
results = {
    'Train_MSE': mse_train,
    'Train_RMSE': rmse_train,
    'Test_MSE': mse_test,
    'Test_RMSE': rmse_test,
    'Model_Params': model_params
}

# Save the results to a text file
with open(f'savedModel/{chosen_category}/model_resultsGPR {chosen_category}.txt', 'w') as file:
    for key, value in results.items():
        file.write(f'{key}: {value}\n')

# Save the trained GP model to a file
model_filename = f"savedModel/{chosen_category}/{chosen_category} model.pkl"
joblib.dump(model, model_filename)
