import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_nutrition_category(data, category):
    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot the selected nutrient category
    plt.plot(data['Date'], data[category], label=category, marker='o')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title(f'{category} Over Time')

    # Set x-axis ticks and labels for better readability
    x_ticks = np.arange(0, len(data['Date']), step=max(1, len(data['Date']) // 10))  # Adjust the step value as needed
    plt.xticks(x_ticks, data['Date'].iloc[x_ticks], rotation=45, ha='right')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Load the dataset into a DataFrame
file_path = r'D:\Bachelor of Computer Science (Artificial Intelligence)\FYP\FYP\Average_FoodWaste.csv'

# Load the dataset
nutrition_data = pd.read_csv(file_path)

# Specify the category you want to plot
chosen_category = 'Carb'  # You can change this to the desired category

# Plot the selected category
plot_nutrition_category(nutrition_data, chosen_category)
