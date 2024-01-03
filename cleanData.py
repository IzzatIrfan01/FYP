import pandas as pd

# Read the CSV file
data = pd.read_csv(r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\src\Average_FoodWaste_Daily.csv')

# Rest of your data processing code remains the same
df = pd.DataFrame(data, columns=['Date', 'Carb', 'Fiber', 'Protein', 'Fat'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', dayfirst=True)

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Resample the data to weekly frequency, starting on the first day of the week (Monday)
weekly_average_nutrients = df.resample('W-Sun').mean().reset_index()

# Round the values to 2 decimal places
weekly_average_nutrients = weekly_average_nutrients.round(2)

# Display the DataFrame without the index number
print(weekly_average_nutrients.to_string(index=False))

# Save the processed DataFrame to a new CSV file with the same name in a different directory
output_file_path = r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\src\WeeklySun_Average_FoodWaste.csv'
weekly_average_nutrients.to_csv(output_file_path, index=False)



#**************************************** Calculating for total food waste per day ***************************************#

# # Calculating for total food waste per day
# # Include 'Carb' in the aggregation
# daily_total_nutrients = df.groupby('Date').agg({
#     'Carb': 'mean',    # Change 'mean' to 'sum' for total
#     'Fiber': 'mean',
#     'Protein': 'mean',
#     'Fat': 'mean'
# }).reset_index()

# # Round the values to 2 decimal places
# daily_total_nutrients = daily_total_nutrients.round(2)

# # Display the DataFrame without the index number
# print(daily_total_nutrients.to_string(index=False))

# # Save the processed DataFrame to a new CSV file with the same name in a different directory
# output_file_path = r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\FYP\Average_FoodWaste_Daily.csv'
# daily_total_nutrients.to_csv(output_file_path, index=False)


#**************************************** Calculating for total food waste per week ***************************************#

# # Extract week, year, and weekday from the date
# df['week'] = df['Date'].dt.isocalendar().week
# df['year'] = df['Date'].dt.year
# df['weekday'] = df['Date'].dt.weekday

# # Adjust the date to the starting day of the week (assuming Monday as the start of the week)
# df['Date'] = df['Date'] - pd.to_timedelta(df['weekday'], unit='D')

# # Calculating for total food waste per week
# # Include 'Carb' in the aggregation
# weekly_total_nutrients = df.groupby(['year', 'week', 'Date']).agg({
#     'Carb': 'mean',    # Change 'mean' to 'sum' for total
#     'Fiber': 'mean',
#     'Protein': 'mean',
#     'Fat': 'mean'
# }).reset_index()

# # Round the values to 2 decimal places
# weekly_total_nutrients = weekly_total_nutrients.round(2)

# # Reorder the columns (optional)
# weekly_total_nutrients = weekly_total_nutrients[['Date', 'Carb', 'Fiber', 'Protein', 'Fat']]

# # Display the DataFrame without the index number
# print(weekly_total_nutrients.to_string(index=False))

# # Save the processed DataFrame to a new CSV file with the same name in a different directory
# output_file_path = r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\FYP\Average_FoodWaste1.csv'
# weekly_total_nutrients.to_csv(output_file_path, index=False)

#**************************************** Calculating for average food waste per week ***************************************#

# # Extract year, month, and day from the date
# df['year'] = df['Date'].dt.year
# df['month'] = df['Date'].dt.month
# df['day'] = df['Date'].dt.day

# # Calculate the week number based on every 7 days within the same month
# df['week'] = (df['day'] - 1) // 7 + 1

# # Create a new DataFrame with necessary columns before assigning to the Date column
# new_date_df = df[['year', 'month', 'week']].assign(day=1)

# # Calculate the first day of each 7-day period within the same month
# df['Date'] = pd.to_datetime({'year': new_date_df['year'], 'month': new_date_df['month'], 'day': new_date_df['day']})

# # Calculating for average food waste per week within the same month
# # Include 'Carb' in the aggregation
# weekly_avg_nutrients = df.groupby(['year', 'month', 'week', 'Date']).agg({
#     'Carb': 'mean',
#     'Fiber': 'mean',
#     'Protein': 'mean',
#     'Fat': 'mean'
# }).reset_index()

# # Round the values to 2 decimal places
# weekly_avg_nutrients = weekly_avg_nutrients.round(2)

# # Reorder the columns (optional)
# weekly_avg_nutrients = weekly_avg_nutrients[['Date', 'Carb', 'Fiber', 'Protein', 'Fat']]

# # Display the DataFrame without the index number
# print(weekly_avg_nutrients.to_string(index=False))

# # Save the processed DataFrame to a new CSV file with the same name in a different directory
# output_file_path = r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\FYP\Average_FoodWaste1.csv'
# weekly_avg_nutrients.to_csv(output_file_path, index=False)